"""GPU training entry point. Smoke preserves full-mode hyperparameters, caps steps."""
import argparse
import json
import math
import time
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import mlflow
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainerCallback, set_seed)
from peft import LoraConfig, TinyLoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from src.config import (BASE_MODEL, BASE_REVISION, TOKENIZER_REVISION, OUTPUT_ROOT,
                        LORA_TARGET_MODULES, MAX_SEQ_LENGTH, SEED)
from src.data_prep import prepare_data, digest

CHECKPOINTING_KWARGS = {"use_reentrant": False}


def training_config(output_dir, *, smoke, bf16):
    return SFTConfig(
        output_dir=str(output_dir), per_device_train_batch_size=1,
        per_device_eval_batch_size=1, gradient_accumulation_steps=32,
        num_train_epochs=1, max_steps=2 if smoke else -1,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=5,
        logging_steps=1 if smoke else 10, eval_strategy="no" if smoke else "steps",
        eval_steps=40, save_strategy="no" if smoke else "steps", save_steps=40,
        save_total_limit=1, bf16=bf16, fp16=not bf16, optim="paged_adamw_8bit",
        report_to="none", remove_unused_columns=False, seed=SEED, data_seed=SEED,
        gradient_checkpointing=True, gradient_checkpointing_kwargs=CHECKPOINTING_KWARGS,
        max_length=MAX_SEQ_LENGTH, completion_only_loss=True,
        assistant_only_loss=False, packing=False, padding_free=False)


def smoke_subset(dataset, metadata):
    # Two optimizer steps × accumulation 32 × microbatch 1, one GPU only.
    if len(dataset) < 64:
        raise ValueError("Smoke requires at least 64 effective records")
    ids = metadata["effective"][:64]
    metadata["training_selection"] = {"mode": "smoke", "count": 64,
                                      "ids": ids, "ids_hash": digest(ids)}
    return dataset.select(range(64))


def save_protocol(path, tokenizer, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(path)
    (path / "effective_data.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class RunCallback(TrainerCallback):
    def __init__(self, tokenizer, metadata):
        self.tokenizer, self.metadata = tokenizer, metadata

    def on_log(self, args, state, control, logs=None, **kwargs):
        metrics = {k: float(v) for k, v in (logs or {}).items()
                   if isinstance(v, (int, float))}
        if any(not math.isfinite(v) for v in metrics.values()):
            raise ValueError("Nonfinite Trainer metric")
        mlflow.log_metrics(metrics, step=state.global_step)

    def on_save(self, args, state, control, **kwargs):
        save_protocol(Path(args.output_dir) / f"checkpoint-{state.global_step}",
                      self.tokenizer, self.metadata)

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        gradients = [p.grad.detach().float().norm() for p in model.parameters()
                     if p.requires_grad and p.grad is not None]
        if not gradients:
            raise ValueError("No adapter gradients")
        norm = torch.stack(gradients).norm().item()
        if not math.isfinite(norm):
            raise ValueError("Nonfinite adapter gradient")
        mlflow.log_metric("adapter_grad_norm", norm, step=state.global_step)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["lora", "qlora", "tinylora"], default="qlora")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Training requires exactly one visible CUDA GPU; select CUDA_VISIBLE_DEVICES=0")
    set_seed(SEED)  # Before model and adapter initialization, including TinyLoRA.
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    mode = "smoke" if args.smoke else "training"
    root = Path(OUTPUT_ROOT) / f"{args.method}_{mode}_{uuid4().hex[:12]}"
    config = training_config(root / "checkpoints", smoke=args.smoke, bf16=bf16)
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("llama-3-math-instruct")
    with mlflow.start_run(run_name=f"{args.method}_{mode}") as run:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=TOKENIZER_REVISION, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        train_ds, eval_ds, metadata = prepare_data(tokenizer, return_metadata=True)
        if args.smoke:
            train_ds = smoke_subset(train_ds, metadata)
        else:
            metadata["training_selection"] = {"mode": "full", "count": len(train_ds),
                                               "ids_hash": metadata["effective_ids_hash"]}
        metadata.update({"method": args.method, "base_model": BASE_MODEL,
                         "base_revision": BASE_REVISION, "precision": str(dtype),
                         "run_id": run.info.run_id, "training_config": config.to_dict(),
                         "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                                     "gpu": torch.cuda.get_device_name(0)},
                         "dependencies": {p: version(p) for p in
                            ("torch", "transformers", "trl", "peft", "accelerate",
                             "datasets", "bitsandbytes", "mlflow", "huggingface_hub")}})
        mlflow.log_params({"method": args.method, "mode": mode, "seed": SEED,
                           "base_revision": BASE_REVISION, "dataset_revision": metadata["revision"],
                           "precision": str(dtype), "gpu": torch.cuda.get_device_name(0),
                           "cuda": torch.version.cuda, "train_samples": len(train_ds),
                           "max_steps": config.max_steps, "gradient_accumulation_steps": 32})
        model_kwargs = {"revision": BASE_REVISION, "dtype": dtype, "device_map": {"": 0}}
        if args.method == "qlora":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)
        model.config.use_cache = False
        if args.method == "qlora":
            if not getattr(model, "is_loaded_in_4bit", False):
                raise ValueError("QLoRA base is not four-bit")
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs=CHECKPOINTING_KWARGS)
        if args.method in ("lora", "qlora"):
            adapter = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05,
                target_modules=LORA_TARGET_MODULES, bias="none", task_type="CAUSAL_LM", revision=BASE_REVISION)
        else:
            adapter = TinyLoraConfig(r=2, u=64, weight_tying=0.3, projection_seed=SEED,
                save_projection=True, init_v_bound=0.02, target_modules=LORA_TARGET_MODULES,
                tinylora_dropout=0.0, bias="none", task_type="CAUSAL_LM", revision=BASE_REVISION)
        model = get_peft_model(model, adapter)
        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if not trainable or any("lora" not in n.lower() for n in trainable):
            raise ValueError("Unexpected trainable base parameters")
        mlflow.log_metric("trainable_parameters", sum(p.numel() for p in trainable.values()))
        initial = {n: p.detach().cpu().clone() for n, p in trainable.items()} if args.smoke else None
        trainer = SFTTrainer(model=model, args=config, train_dataset=train_ds,
            eval_dataset=eval_ds, processing_class=tokenizer,
            callbacks=[RunCallback(tokenizer, metadata)])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        trainer.train()
        if args.smoke:
            if trainer.state.global_step != 2:
                raise ValueError("Smoke did not execute exactly two optimizer steps")
            delta = sum((p.detach().cpu().float() - initial[n].float()).square().sum().item()
                        for n, p in trainable.items()) ** 0.5
            if not math.isfinite(delta) or delta <= 0:
                raise ValueError("Smoke adapter did not update")
            mlflow.log_metric("adapter_update_norm", delta)
        metrics = trainer.evaluate()  # Required even when scheduled eval never fires.
        if not math.isfinite(metrics["eval_loss"]):
            raise ValueError("Nonfinite final validation loss")
        mlflow.log_metrics({f"final_{k}": float(v) for k, v in metrics.items()
                            if isinstance(v, (float, int))}, step=trainer.state.global_step)
        torch.cuda.synchronize()
        mlflow.log_metrics({"train_and_eval_seconds": time.perf_counter() - started,
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                           "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        final = root / "final"
        trainer.save_model(str(final))
        save_protocol(final, tokenizer, metadata)
        mlflow.log_dict(metadata, "effective_data.json")
        mlflow.log_artifacts(str(final), artifact_path="final")
        print(f"Saved adapter/tokenizer/protocol: {final}")


if __name__ == "__main__":
    main()
