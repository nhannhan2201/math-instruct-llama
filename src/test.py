"""Teacher-forced evaluation of the 500 held-out MathInstruct questions.

Run one method per process/GPU. This measures reference-token prediction, not
the correctness of answers generated independently by the model.
"""

import argparse
import json
import math
from importlib.metadata import version
from pathlib import Path

import mlflow
import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

from src.config import (BASE_MODEL, BASE_REVISION, CHAT_TEMPLATE_KWARGS,
                        DATASET_ID, DATASET_REVISION, DATASET_SNAPSHOT_HASH,
                        MAX_SEQ_LENGTH, MAX_TEST_SAMPLES, SEED,
                        SPLIT_MANIFEST_PATH, TOKENIZER_REVISION)
from src.data_prep import (cached_revision, digest, format_batch,
                           load_or_create_manifest, tokenizer_provenance)


def build_test_dataset(raw, tokenizer, manifest):
    """Return complete, untruncated test examples in manifest order."""
    refs = manifest["references"]["test"]
    if len(refs) != MAX_TEST_SAMPLES:
        raise ValueError(f"Expected {MAX_TEST_SAMPLES} test groups, got {len(refs)}")
    examples = []
    for ref in refs:
        index = ref["primary_reference"]
        row = raw[index]
        example = {key: value[0] for key, value in format_batch(
            {"instruction": [row["instruction"]], "output": [row["output"]]}, tokenizer).items()}
        prompt_ids = tokenizer.apply_chat_template(
            example["prompt"], tokenize=True, return_dict=False,
            add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS)
        ids = tokenizer.apply_chat_template(
            example["prompt"] + example["completion"], tokenize=True,
            return_dict=False, **CHAT_TEMPLATE_KWARGS)
        if ids[:len(prompt_ids)] != prompt_ids or len(ids) <= len(prompt_ids) + 1:
            raise ValueError(f"Invalid completion boundary for test row {index}")
        if ids[0] != tokenizer.bos_token_id or ids[-1] != tokenizer.convert_tokens_to_ids("<|eot_id|>"):
            raise ValueError(f"Invalid special tokens for test row {index}")
        examples.append(example)
    return Dataset.from_list(examples)


class EvaluationMetrics(TrainerCallback):
    """TRL 1.3 adds token accuracy to log events, not evaluate()'s return value."""

    def __init__(self):
        self.values = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.values.update({key: value for key, value in (logs or {}).items()
                            if key.startswith("eval_")})


def evaluate_test(trainer, collector):
    collector.values.clear()
    returned = trainer.evaluate()
    return {"test_" + key[len("eval_"):]: float(value)
            for key, value in {**returned, **collector.values}.items()
            if key.startswith("eval_") and isinstance(value, (float, int))}


def verify_adapter(path, method, tokenizer, manifest):
    if path is None:
        raise ValueError(f"--adapter-path is required for {method}")
    metadata_path = path / "effective_data.json"
    config_path = path / "adapter_config.json"
    if not metadata_path.is_file() or not config_path.is_file():
        raise ValueError("Adapter must contain effective_data.json and adapter_config.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {"method": method, "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
                "revision": DATASET_REVISION, "snapshot_hash": DATASET_SNAPSHOT_HASH,
                "manifest_hash": digest(manifest), "max_length": MAX_SEQ_LENGTH,
                "tokenizer": tokenizer_provenance(tokenizer)}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Adapter provenance mismatch: {key}")
    if (adapter_config.get("base_model_name_or_path") != BASE_MODEL
            or adapter_config.get("revision") != BASE_REVISION):
        raise ValueError("Adapter config base/revision mismatch")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("base", "lora", "qlora"), required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--tracking-uri", default="sqlite:///mlruns.db")
    parser.add_argument("--manifest-path", type=Path, default=SPLIT_MANIFEST_PATH)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Evaluation requires exactly one visible CUDA GPU")
    if args.method == "base" and args.adapter_path is not None:
        parser.error("Base model must not have an adapter")
    set_seed(SEED)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=TOKENIZER_REVISION, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset(DATASET_ID, revision=DATASET_REVISION)["train"]
    manifest = load_or_create_manifest(raw, cached_revision(raw), args.manifest_path)
    dataset = build_test_dataset(raw, tokenizer, manifest)
    metadata = verify_adapter(args.adapter_path, args.method, tokenizer, manifest) if args.method != "base" else None
    if metadata is not None and metadata.get("precision") != str(dtype):
        raise ValueError("Adapter training precision differs from evaluation precision")

    model_kwargs = {"revision": BASE_REVISION, "dtype": dtype, "device_map": {"": 0}}
    if args.method == "qlora":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)
    if args.method == "qlora" and not getattr(model, "is_loaded_in_4bit", False):
        raise ValueError("QLoRA base is not four-bit")
    if metadata is not None:
        model = PeftModel.from_pretrained(model, str(args.adapter_path), is_trainable=False)
    model.eval()
    lengths = [len(tokenizer.apply_chat_template(
        example["prompt"] + example["completion"], tokenize=True,
        return_dict=False, **CHAT_TEMPLATE_KWARGS)) for example in dataset]
    longest = max(lengths)
    if longest > model.config.max_position_embeddings:
        raise ValueError(f"Test length {longest} exceeds model context capacity")
    print(f"Test: {len(dataset)} complete samples; longest={longest} tokens; no truncation", flush=True)

    config = SFTConfig(
        output_dir=f"models/test_{args.method}", per_device_eval_batch_size=1,
        report_to="none", bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16, max_length=None,
        completion_only_loss=True, assistant_only_loss=False,
        packing=False, padding_free=False, remove_unused_columns=False)
    collector = EvaluationMetrics()
    # TRL 1.3 requires train_dataset even for evaluation. Never call train().
    trainer = SFTTrainer(model=model, args=config, train_dataset=dataset, eval_dataset=dataset,
                         processing_class=tokenizer, callbacks=[collector])
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("llama-3-math-instruct-test")
    with mlflow.start_run(run_name=f"{args.method}_test") as run:
        mlflow.log_params({
            "method": args.method, "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
            "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
            "snapshot_hash": DATASET_SNAPSHOT_HASH, "manifest_hash": digest(manifest),
            "test_groups": len(dataset), "training_max_length": MAX_SEQ_LENGTH,
            "test_max_length": longest, "test_truncation": "none",
            "accuracy_aggregation": "mean_per_example_batch_size_1_trl_1.3",
            "completion_only_loss": True, "seed": SEED, "precision": str(dtype),
            "gpu": torch.cuda.get_device_name(0),
            "adapter_path": str(args.adapter_path) if args.adapter_path else "none",
            "training_run_id": metadata.get("run_id", "unknown") if metadata else "none",
            "trl_version": version("trl"), "transformers_version": version("transformers"),
        })
        metrics = evaluate_test(trainer, collector)
        for key in ("test_loss", "test_mean_token_accuracy"):
            if key not in metrics or not math.isfinite(float(metrics[key])):
                raise ValueError(f"Missing or nonfinite metric: {key}")
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()
                            if isinstance(value, (int, float)) and math.isfinite(float(value))})
        mlflow.log_dict({
            "run_id": run.info.run_id, "method": args.method,
            "test_references": manifest["references"]["test"],
            "metrics": metrics,
            "interpretation": "Teacher-forced reference-token metrics; not answer accuracy",
        }, "test_protocol.json")
        print(json.dumps({"run_id": run.info.run_id, "method": args.method,
                          "test_loss": metrics["test_loss"],
                          "test_mean_token_accuracy": metrics["test_mean_token_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
