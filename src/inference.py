import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
from src.config import BASE_MODEL, OUTPUT_ROOT, PROMPT_TEMPLATE

class MathSolver:
    def __init__(self, method="qlora"):
        print(f"Đang khởi tạo model ({method.upper()}) để dự đoán...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
        
        # Đường dẫn tới thư mục model bạn vừa train xong
        adapter_path = f"{OUTPUT_ROOT}/{method}_final"
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        # Chỉ load 4-bit nếu phương pháp là qlora
        if method == "qlora":
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_cfg,
                device_map="auto",
            )
        else:
            # Load 16-bit cho lora và tinylora
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                dtype=dtype,
                device_map="auto",
            )
        
        # Lắp "não" LoRA/TinyLoRA (adapter) vừa fine-tune vào Base Model
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # Tạo pipeline sinh văn bản
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )
        print("Khởi tạo hoàn tất! Đã sẵn sàng giải toán.")

    def solve(self, question, max_new_tokens=150, temperature=0.7):
        """
        Nhận câu hỏi, đưa vào prompt template và trả về câu trả lời.
        """
        # Format câu hỏi theo đúng template lúc train (bỏ trống phần answer)
        prompt = PROMPT_TEMPLATE.format(question=question, answer="")
        
        # Sinh câu trả lời
        result = self.generator(
            prompt, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            temperature=temperature, 
            return_full_text=False # Không in lại phần câu hỏi
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
    args = parser.parse_args()

    solver = MathSolver(method=args.method)
    test_q = "If 3 apples cost 90 cents, how much do 5 apples cost?"
    print(f"\nHỏi: {test_q}")
    print(f"Đáp: {solver.solve(test_q)}")