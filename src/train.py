import os
import torch
import mlflow
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TinyLoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

# Import các thiết lập từ config và module xử lý data
from src.config import (
    BASE_MODEL, OUTPUT_ROOT, LORA_TARGET_MODULES, 
    MAX_SEQ_LENGTH, SEED
)
from src.data_prep import prepare_data

def main():
    # 0. CẤU HÌNH ARGPARSE ĐỂ NHẬN LỆNH TỪ TERMINAL
    parser = argparse.ArgumentParser(description="Huấn luyện Llama 3.2 1B bằng PEFT")
    parser.add_argument(
        "--method", 
        type=str, 
        choices=["lora", "qlora", "tinylora"], 
        default="qlora", 
        help="Chọn phương pháp huấn luyện: lora, qlora, hoặc tinylora"
    )
    args = parser.parse_args()

    # 1. KHỞI TẠO MLFLOW TRACKING
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("llama-3-math-instruct")

    # Tên Run sẽ tự động thay đổi theo method bạn chọn (VD: qlora_training)
    with mlflow.start_run(run_name=f"{args.method}_training"):
        
        # 2. LOAD TOKENIZER & DATASET
        print("Đang tải Tokenizer và Chuẩn bị Dataset...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_ds, eval_ds = prepare_data(tokenizer)

        # 3. CẤU HÌNH VÀ LOAD MÔ HÌNH BASE TÙY THEO METHOD
        print(f"Đang tải Base Model cho phương pháp {args.method.upper()}...")
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        if args.method == "qlora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
            )
            model.config.use_cache = False
            model = prepare_model_for_kbit_training(model)
        else:
            # Dành cho lora và tinylora (Load full precision 16-bit)
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                dtype=dtype,
                device_map="auto",
            )
            model.config.use_cache = False

        # 4. ÁP DỤNG PEFT ADAPTER (LORA / TINYLORA)
        if args.method in ["lora", "qlora"]:
            peft_config = LoraConfig(
                r=4,
                lora_alpha=8,
                lora_dropout=0.05,
                target_modules=LORA_TARGET_MODULES,
                bias="none",
                task_type="CAUSAL_LM",
            )
        else:
            # Dành cho TinyLoRA
            peft_config = TinyLoraConfig(
                r=2,
                u=64,
                weight_tying=0.3,
                projection_seed=SEED,
                save_projection=True,
                init_v_bound=0.02,
                target_modules=LORA_TARGET_MODULES,
                tinylora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # 5. LOG CÁC THAM SỐ VÀO MLFLOW
        mlflow.log_params({
            "method": args.method,
            "learning_rate": 2e-4,
            "epochs": 1,
            "batch_size": 1
        })

        # 6. KHỞI TẠO SFT TRAINER
        # Output dir tự động lưu theo method (VD: models/qlora_checkpoints)
        training_args = SFTConfig(
            output_dir=f"{OUTPUT_ROOT}/{args.method}_checkpoints",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=32,
            num_train_epochs=1,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_steps=5,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=40,
            save_strategy="steps",
            save_steps=40,
            save_total_limit=1,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            optim="paged_adamw_8bit",
            report_to="mlflow", 
            remove_unused_columns=False,
            seed=SEED,
            max_length=MAX_SEQ_LENGTH,
        )

        def _format_one(example):
            return example["text"]

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            formatting_func=_format_one,
        )

        # 7. CHẠY HUẤN LUYỆN
        print(f"Bắt đầu huấn luyện với phương pháp {args.method.upper()}...")
        trainer.train()

        # 8. LƯU LẠI CHECKPOINT CUỐI CÙNG
        final_model_path = f"{OUTPUT_ROOT}/{args.method}_final"
        trainer.save_model(final_model_path)
        print(f"Hoàn thành! Model đã được lưu tại: {final_model_path}")

if __name__ == "__main__":
    main()