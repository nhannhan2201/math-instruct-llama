"""Offline CPU inspection of the actual prepared train set; never loads a model."""

import os

# Set before importing HF libraries; do not silently fall back to downloads.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import re
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from src.config import BASE_MODEL, DATASET_ID, MAX_SEQ_LENGTH, SEED
from src.data_prep import format_batch, prepare_data


def recover_metadata(train_ds, raw_ds, tokenizer):
    """Match formatted text, without repeating split/shuffle/sampling logic.

    Multiple identical texts can have different sources or original whitespace.
    Keep a representative raw row and all source candidates; never invent an ID.
    """
    wanted = set(train_ds["text"])
    matches = {}
    for start in range(0, len(raw_ds), 2000):
        batch = raw_ds[start:start + 2000]
        texts = format_batch(batch, tokenizer)["text"]
        for offset, text in enumerate(texts):
            if text not in wanted:
                continue
            source = batch.get("source", [None] * len(texts))[offset]
            if text not in matches:
                matches[text] = {
                    "raw_index": start + offset,
                    "instruction": batch["instruction"][offset],
                    "output": batch["output"][offset],
                    "sources": set(),
                    "raw_matches": 0,
                }
            matches[text]["sources"].add(source)
            matches[text]["raw_matches"] += 1
    if wanted - matches.keys():
        raise ValueError("Prepared text could not be matched to cached raw data.")
    return matches


def inspect_samples(train_ds, metadata, tokenizer, max_length):
    """Tokenizer-level truncation simulation, not SFTTrainer preprocessing."""
    if not tokenizer.is_fast or max_length <= 0:
        raise ValueError("Requires a fast tokenizer and positive max_length.")
    lengths, sources, examples = [], Counter(), []
    counts = Counter()
    example_categories = set()
    for index, row in enumerate(train_ds):
        text = row["text"]
        original = metadata[text]
        question = (original["instruction"] or "").strip()
        answer = (original["output"] or "").strip()
        # Reuse formatter for the boundary too; no independent template implementation.
        empty = format_batch({"instruction": [question], "output": [""]}, tokenizer)["text"][0]
        answer_start = len(empty) - len(tokenizer.eos_token)
        answer_end = answer_start + len(answer)
        full = tokenizer(text, return_offsets_mapping=True)
        cut = tokenizer(text, truncation=True, max_length=max_length,
                        return_offsets_mapping=True)
        offsets = full["offset_mapping"]
        cut_offsets = cut["offset_mapping"]
        before = sum(a < answer_end and b > answer_start for a, b in offsets)
        mask = [a < answer_end and b > answer_start for a, b in cut_offsets]
        after = sum(mask)
        # Locate appended EOS by its character span, not any EOS inside the answer.
        has_eos = any(t == tokenizer.eos_token_id and a >= answer_end and b > a
                      for t, (a, b) in zip(full["input_ids"], offsets))
        eos_mask = [t == tokenizer.eos_token_id and a >= answer_end and b > a
                    for t, (a, b) in zip(cut["input_ids"], cut_offsets)]
        retained_eos = any(eos_mask)
        lengths.append(len(full["input_ids"]))
        counts["answer_truncated"] += after < before
        counts["eos_lost"] += has_eos and not retained_eos
        counts["no_answer_after"] += after == 0
        counts["empty_answer_before"] += before == 0
        counts["eos_absent_before"] += not has_eos
        counts["ambiguous_raw_match"] += original["raw_matches"] > 1
        candidates = original["sources"]
        source = next(iter(candidates)) if len(candidates) == 1 else "AMBIGUOUS_SOURCE"
        sources[str(source)] += 1
        category = "lost" if before and not after else "cut" if after < before else "intact"
        # Three initial real rows, then up to two additional representative cases.
        if index < 3 or (len(examples) < 5 and category not in example_categories):
            example_categories.add(category)
            full_labels = list(cut["input_ids"])
            completion_labels = [t if answer_token or eos else -100
                                 for t, answer_token, eos in
                                 zip(cut["input_ids"], mask, eos_mask)]
            examples.append({
                "train_index": index, "representative_raw_index": original["raw_index"],
                "raw_match_count": original["raw_matches"],
                "source_candidates": sorted(str(s) for s in candidates),
                "instruction": original["instruction"], "output": original["output"],
                "formatted_text": text, "token_length": len(full["input_ids"]),
                "answer_tokens_before": before, "answer_tokens_after": after,
                "status": category, "eos_retained": retained_eos,
                "text_after_truncation": tokenizer.decode(cut["input_ids"]),
                "removed_tokens_decoded": tokenizer.decode(
                    full["input_ids"][max_length:] if tokenizer.truncation_side == "right"
                    else full["input_ids"][:-max_length]),
                "simulated_labels_full_sequence": full_labels,
                "simulated_labels_completion_only": completion_labels,
            })
    if not lengths:
        raise ValueError("Prepared train set is empty.")
    return {
        "samples_inspected": len(lengths), "max_length": max_length,
        "token_length": dict(zip(("min", "median", "p90", "p95", "p99", "max"),
                                 np.percentile(lengths, [0, 50, 90, 95, 99, 100]).tolist())),
        "percentile_method": "numpy linear interpolation",
        "over_threshold": {str(n): {"count": sum(x > n for x in lengths),
                                     "percent": 100 * sum(x > n for x in lengths) / len(lengths)}
                           for n in (128, 256, 512)},
        "counts": dict(counts), "source_distribution": dict(sorted(sources.items())),
        "examples": examples,
    }



def verify_actual_labels(train_ds, metadata, tokenizer, examples):
    """Run real TRL preprocessing/collation; never forward or train the tiny model."""
    import tempfile
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from trl import SFTConfig, SFTTrainer

    chosen = []
    for category in ("intact", "cut", "lost"):
        example = next((e for e in examples if e["status"] == category), None)
        if example is None:
            raise ValueError(f"No real example for {category}; cannot complete verification")
        chosen.append(example["train_index"])
    subset = train_ds.select(chosen)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
        max_position_embeddings=MAX_SEQ_LENGTH,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    ))
    with tempfile.TemporaryDirectory(prefix="math-label-check-") as output:
        args = SFTConfig(
            output_dir=output, use_cpu=True, bf16=False, fp16=False,
            report_to="none", gradient_checkpointing=False, optim="adamw_torch",
            per_device_train_batch_size=1, max_length=MAX_SEQ_LENGTH,
            remove_unused_columns=False, seed=SEED, disable_tqdm=True,
        )
        trainer = SFTTrainer(model=model, args=args, train_dataset=subset,
                             processing_class=tokenizer,
                             formatting_func=lambda example: example["text"])
        features = [trainer.train_dataset[i] for i in range(len(subset))]
        batch = trainer.data_collator(features)
        checks = {}
        checks["padding_ignored"] = bool(torch.all(batch["labels"][batch["attention_mask"] == 0] == -100))
        checks["padding_present"] = bool(torch.any(batch["attention_mask"] == 0))
        checks["cpu_only"] = all(v.device.type == "cpu" for v in batch.values() if isinstance(v, torch.Tensor))
        checks["completion_only_disabled"] = trainer.completion_only_loss is False
        records = []
        for row, train_index in enumerate(chosen):
            ids = batch["input_ids"][row].tolist()
            attention = batch["attention_mask"][row].tolist()
            labels = batch["labels"][row].tolist()
            text = subset[row]["text"]
            original = metadata[text]
            prefix = format_batch({"instruction": [original["instruction"]], "output": [""]}, tokenizer)["text"][0]
            start = len(prefix) - len(tokenizer.eos_token)
            end = start + len((original["output"] or "").strip())
            encoded = tokenizer(text, return_offsets_mapping=True)
            real_ids = [t for t, m in zip(ids, attention) if m]
            expected = encoded["input_ids"][:MAX_SEQ_LENGTH]
            matches = real_ids == expected
            checks[f"row_{row}_preprocessing_matches"] = matches
            if not matches:
                raise ValueError("Trainer preprocessing differs from expected tokenizer prefix; do not infer spans")
            prompt_positions = [j for j, (a, b) in enumerate(encoded["offset_mapping"][:len(real_ids)]) if b > a and b <= start]
            answer_positions = [j for j, (a, b) in enumerate(encoded["offset_mapping"][:len(real_ids)]) if a < end and b > start]
            checks[f"row_{row}_real_labels_equal_ids"] = all(l == t for t, l, m in zip(ids, labels, attention) if m)
            checks[f"row_{row}_prompt_unmasked"] = bool(prompt_positions) and all(labels[j] != -100 for j in prompt_positions)
            active = [j for j, l in enumerate(labels) if j > 0 and l != -100]
            records.append({
                "train_index": train_index, "input_ids": ids, "attention_mask": attention,
                "labels": labels, "ignored_positions": [j for j, l in enumerate(labels) if l == -100],
                "prompt_positions": prompt_positions, "answer_positions": answer_positions,
                "shifted_loss_target_positions": active,
                "decoded_input": tokenizer.decode(real_ids),
                "decoded_loss_targets": tokenizer.decode([ids[j] for j in active]),
                "prompt_targets_after_shift": sum(j in active for j in prompt_positions),
                "answer_targets_after_shift": sum(j in active for j in answer_positions),
                "padding_tokens": attention.count(0),
            })
        # Also check the actual training microbatch shape (one sample, no forced padding).
        singles = [trainer.data_collator([f]) for f in features]
        checks["microbatch_one_labels_match"] = all(
            b["labels"][0].tolist() == records[i]["labels"][:int(b["attention_mask"][0].sum())]
            for i, b in enumerate(singles))
        result = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "evidence_level": "actual SFTTrainer preprocessing and default collator",
            "versions": {n: version(n) for n in ("trl", "transformers", "torch", "peft", "accelerate")},
            "collator": type(trainer.data_collator).__name__,
            "effective_settings": {"completion_only_loss": trainer.completion_only_loss,
                                   "assistant_only_loss": args.assistant_only_loss,
                                   "packing": args.packing, "padding_free": args.padding_free,
                                   "max_length": args.max_length, "truncation_mode": args.truncation_mode},
            "checks": checks, "batch": records,
            "source_conclusion": "Plain text dataset selects full-sequence loss; collator masks padding with -100.",
            "limits": ["Tiny random Llama model; base weights and PEFT/quantization not loaded.",
                       "No forward, numerical loss, backward or training; shifted target positions describe loss eligibility.",
                       "Inspection overrides CPU/precision/tracking/optimizer/checkpointing settings only; source training files unchanged.",
                       "Three-row batch checks padding; single-row batches also checked to match training microbatch 1.",
                       "Applies to recorded local versions; recheck versions on Kaggle."],
        }
    return result

def normalize_whitespace(text):
    """Only normalize whitespace; preserve case, punctuation and math symbols."""
    return " ".join(text.split())


def inspect_duplicates(raw_ds, train_ds, eval_ds, tokenizer):
    """Count duplicates without removing records or reconstructing splits."""
    pairs, questions = {}, {}
    for index, row in enumerate(raw_ds):
        question = normalize_whitespace(row["instruction"])
        answer = normalize_whitespace(row["output"])
        pairs.setdefault((question, answer), []).append(index)
        questions.setdefault(question, set()).add(answer)

    def records(indices):
        return [{"raw_index": i, "source": raw_ds[i].get("source"),
                 "instruction": raw_ds[i]["instruction"], "output": raw_ds[i]["output"]}
                for i in indices]

    duplicate_pairs = [(key, ids) for key, ids in pairs.items() if len(ids) > 1]
    multiple_outputs = [q for q, answers in questions.items() if len(answers) > 1]
    # Recover both prepared sets together using the existing formatter matcher.
    metadata = recover_metadata({"text": list(train_ds["text"]) + list(eval_ds["text"])},
                                raw_ds, tokenizer)
    split_indices = {}
    for name, dataset in (("train", train_ds), ("validation", eval_ds)):
        grouped = {}
        for i, text in enumerate(dataset["text"]):
            question = normalize_whitespace(metadata[text]["instruction"])
            grouped.setdefault(question, []).append(i)
        split_indices[name] = grouped
    common = sorted(split_indices["train"].keys() & split_indices["validation"].keys())
    return {
        "normalization": '" ".join(text.split()); case/punctuation/math symbols preserved',
        "raw_records": len(raw_ds), "unique_questions": len(questions),
        "unique_question_output_pairs": len(pairs),
        "duplicate_pair_groups": len(duplicate_pairs),
        "duplicate_pair_records": sum(len(ids) for _, ids in duplicate_pairs),
        "surplus_pair_records": sum(len(ids) - 1 for _, ids in duplicate_pairs),
        "questions_with_multiple_outputs": len(multiple_outputs),
        "split_overlap": {
            "train_samples": len(train_ds), "validation_samples": len(eval_ds),
            "shared_questions": len(common),
            "train_samples_involved": sum(len(split_indices["train"][q]) for q in common),
            "validation_samples_involved": sum(len(split_indices["validation"][q]) for q in common),
        },
        "examples": {
            "duplicate_pairs": [{"raw_indices": ids, "records": records(ids)}
                                for _, ids in duplicate_pairs[:5]],
            "multiple_outputs": [{"normalized_question": q,
                                  "records": records([pairs[(q, a)][0] for a in sorted(questions[q])])}
                                 for q in multiple_outputs[:5]],
            "split_overlap": [{"normalized_question": q,
                               "train_indices": split_indices["train"][q],
                               "validation_indices": split_indices["validation"][q],
                               "records": records([pairs[(q, a)][0] for a in sorted(questions[q])])}
                              for q in common[:5]],
        },
        "limitations": [
            "Different outputs are not evidence of conflicting final answers.",
            "Only whitespace-normalized equality; semantic/near duplicates not measured.",
            "No test split exists; overlap covers current sampled train and validation only.",
            "Raw indices identify rows in this cached snapshot; split raw identity can be ambiguous.",
            "No samples removed; split, sampling, formatting and training settings unchanged.",
        ],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Offline real-data inspection")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--verify-labels", action="store_true", help="Verify actual TRL labels on CPU")
    modes.add_argument("--check-duplicates", action="store_true", help="Analyze duplicates and current split overlap")
    options = parser.parse_args()
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, local_files_only=True)
        if tokenizer.pad_token is None:  # Same adjustment as train.py.
            tokenizer.pad_token = tokenizer.eos_token
        raw_ds = load_dataset(DATASET_ID)["train"]
        train_ds, eval_ds = prepare_data(tokenizer)
    except (OSError, ConnectionError) as exc:
        raise SystemExit(f"Local cache unavailable; no downloads attempted: {exc}") from exc
    if options.check_duplicates:
        report = inspect_duplicates(raw_ds, train_ds, eval_ds, tokenizer)
        report["dataset_cache_files"] = raw_ds.cache_files
        report["dataset_revision_from_cache"] = sorted({part for entry in raw_ds.cache_files
            for part in Path(entry["filename"]).parts if re.fullmatch(r"[0-9a-f]{40}", part)})
        path = Path(__file__).resolve().parents[1] / "docs" / "data_duplicates_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Raw records: {report['raw_records']}; unique questions: {report['unique_questions']}")
        print(f"Duplicate pair groups: {report['duplicate_pair_groups']}; surplus records: {report['surplus_pair_records']}")
        print(f"Questions with different outputs: {report['questions_with_multiple_outputs']} (not necessarily conflicts)")
        print(f"Train/validation overlap: {report['split_overlap']}")
        print(f"JSON report saved: {path}")
        return
    metadata = recover_metadata(train_ds, raw_ds, tokenizer)
    report = inspect_samples(train_ds, metadata, tokenizer, MAX_SEQ_LENGTH)
    # Derive provenance from actual Arrow paths, not a hard-coded Hub ref.
    revisions = sorted({part for entry in raw_ds.cache_files
                        for part in Path(entry["filename"]).parts
                        if re.fullmatch(r"[0-9a-f]{40}", part)})
    report["dataset_revision_from_cache"] = revisions or "NEEDS VERIFICATION"
    report["dataset_cache_files"] = raw_ds.cache_files
    report["tokenizer"] = {"model_id": BASE_MODEL, "local_files_only": True,
                           "truncation_side": tokenizer.truncation_side,
                           "padding_side": tokenizer.padding_side,
                           "add_special_tokens": True}
    try:
        report["trl_version"] = version("trl")
    except PackageNotFoundError:
        report["trl_version"] = "NOT INSTALLED"
    report["limitations"] = [
        "Truncation is simulated using tokenizer calls; no SFTTrainer was instantiated.",
        "Labels in examples are simulations without padding, not actual Trainer labels.",
        "Actual SFTTrainer labels/completion-only loss: NEEDS VERIFICATION.",
        "Source recovery uses formatted-text matching; duplicate raw rows are not unique IDs.",
    ]
    if options.verify_labels:
        try:
            report["actual_labels_verification"] = verify_actual_labels(
                train_ds, metadata, tokenizer, report["examples"])
        except Exception as exc:
            report["actual_labels_verification"] = {
                "status": "FAIL", "error": f"{type(exc).__name__}: {exc}",
                "evidence_level": "Actual Trainer labels NOT VERIFIED; simulated labels remain separate"}
        report["limitations"] = [
            "Top-level truncation statistics and examples remain tokenizer simulations.",
            "Actual-label evidence, scope and CPU test-model limits are in actual_labels_verification."]
    path = Path(__file__).resolve().parents[1] / "docs" / "data_inspection_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Inspected {report['samples_inspected']} training samples.")
    print(f"JSON report saved: {path}")
    if options.verify_labels:
        result = report["actual_labels_verification"]
        print(f"Actual labels verification: {result['status']}")
        if result["status"] == "PASS":
            print("Loss targets cover prompt + answer; padding labels = -100. No training/forward.")
            for item in result["batch"]:
                print(f"Sample {item['train_index']}: prompt targets={item['prompt_targets_after_shift']}, "
                      f"answer targets={item['answer_targets_after_shift']}, padding={item['padding_tokens']}")
                print("Decoded input:", repr(item["decoded_input"]))
        else:
            print(result.get("error", result.get("checks")))
            raise SystemExit(1)
    else:
        print("Actual SFTTrainer labels: NEEDS VERIFICATION; use --verify-labels.")


if __name__ == "__main__":
    main()
