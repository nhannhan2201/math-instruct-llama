import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
from src.config import BASE_MODEL, BASE_REVISION, TOKENIZER_REVISION, OUTPUT_ROOT, CHAT_TEMPLATE_KWARGS

from src.data_prep import build_prompt, tokenizer_provenance

class MathSolver:
    def __init__(self, method="qlora", adapter_path=None):
        print(f"Đang khởi tạo model ({method.upper()}) để dự đoán...")

        # Đường dẫn tới thư mục model bạn vừa train xong
        adapter_path = str(adapter_path or f"{OUTPUT_ROOT}/{method}_final")
        metadata_path = Path(adapter_path, "effective_data.json")
        if not metadata_path.exists():
            raise ValueError("Adapter has no Task 3B protocol provenance; legacy adapter requires review/retraining")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = metadata["tokenizer"]
        if expected.get("base_revision") != BASE_REVISION or expected.get("tokenizer_revision") != TOKENIZER_REVISION:
            raise ValueError("Adapter base/tokenizer revision differs from pinned revision")
        if metadata.get("method", method) != method:
            raise ValueError("Adapter method differs from requested method")
        adapter_config_path = Path(adapter_path, "adapter_config.json")
        if adapter_config_path.exists():
            adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
            if adapter_config.get("base_model_name_or_path") != BASE_MODEL or adapter_config.get("revision") != BASE_REVISION:
                raise ValueError("Adapter config base/revision differs from provenance")
        # Artifact preferred. Fallback only if its tokenizer is absent, using verified local cache.
        artifact_tokenizer = Path(adapter_path, "tokenizer_config.json").exists()
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path if artifact_tokenizer else BASE_MODEL,
            local_files_only=True, use_fast=True,
            **({} if artifact_tokenizer else {"revision": TOKENIZER_REVISION}))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if tokenizer_provenance(self.tokenizer) != expected:
            raise ValueError("Tokenizer/template/protocol differs from adapter provenance")
        precision = metadata.get("precision")
        if precision not in (None, "torch.float16", "torch.bfloat16"):
            raise ValueError("Unsupported artifact precision")
        if precision == "torch.bfloat16" and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            raise ValueError("Artifact requires a BF16-capable CUDA GPU")
        dtype = (torch.bfloat16 if precision == "torch.bfloat16" else torch.float16) if precision else (
            torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)

        # Chỉ load 4-bit nếu phương pháp là qlora
        if method == "qlora":
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, revision=BASE_REVISION,
                quantization_config=bnb_cfg, dtype=dtype,
                device_map="auto",
            )
        else:
            # Load 16-bit cho lora và tinylora
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, revision=BASE_REVISION,
                dtype=dtype,
                device_map="auto",
            )

        base_model.config.use_cache = True
        # Lắp "não" LoRA/TinyLoRA (adapter) vừa fine-tune vào Base Model
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()

        # Tạo pipeline sinh văn bản
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )
        print("Khởi tạo hoàn tất! Đã sẵn sàng giải toán.")

    def solve(self, question, max_new_tokens=150, temperature=0.7, do_sample=True):
        """
        Nhận câu hỏi, đưa vào prompt template và trả về câu trả lời.
        """
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if do_sample and (not isinstance(temperature, (int, float)) or not 0 < temperature <= 2):
            raise ValueError("temperature must be in (0, 2]")
        # Format câu hỏi theo đúng template lúc train (bỏ trống phần answer)
        prompt = self.tokenizer.apply_chat_template(
            build_prompt(question), tokenize=False, add_generation_prompt=True,
            **CHAT_TEMPLATE_KWARGS)

        # Sinh câu trả lời
        result = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **({"temperature": temperature} if do_sample else {}),
            add_special_tokens=False,
            eos_token_id=self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            return_full_text=False
        )

        return result[0]["generated_text"].strip()

# --- Đoạn code test nhanh (Chỉ chạy khi bạn gọi trực tiếp file này) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test inference mô hình đã train")
    parser.add_argument(
        "--method",
        type=str,
        choices=["lora", "qlora", "tinylora"],
        default="qlora",
        help="Chọn phương pháp model bạn muốn test"
    )
    parser.add_argument("--adapter-path", required=True, help="Explicit final/checkpoint artifact directory")
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    solver = MathSolver(method=args.method, adapter_path=args.adapter_path)
    test_q = "If 3 apples cost 90 cents, how much do 5 apples cost?"
    print(f"\nHỏi: {test_q}")
    print(f"Đáp: {solver.solve(test_q, do_sample=not args.greedy)}")
