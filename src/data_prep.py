import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path

from datasets import load_dataset
from src.config import (
    BASE_MODEL, BASE_REVISION, TOKENIZER_REVISION, DATASET_ID, DATASET_REVISION, MAX_EVAL_SAMPLES, TRAIN_FRACTION,
    SEED, SYSTEM_MESSAGE, CHAT_TEMPLATE_KWARGS, MAX_SEQ_LENGTH, MAX_TEST_SAMPLES, SPLIT_MANIFEST_PATH
)

def build_prompt(question):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Empty or invalid question")
    return [{"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": question.strip()}]


def format_batch(examples, tokenizer):
    if len(examples["instruction"]) != len(examples["output"]):
        raise ValueError("Unequal instruction/output batch lengths")
    prompts, completions = [], []
    for question, answer in zip(examples["instruction"], examples["output"]):
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Empty or invalid answer")
        prompts.append(build_prompt(question))
        completions.append([{"role": "assistant", "content": answer.strip()}])
    return {"prompt": prompts, "completion": completions,
            "chat_template_kwargs": [dict(CHAT_TEMPLATE_KWARGS) for _ in prompts]}


def tokenizer_provenance(tokenizer):
    """Fingerprint the actual tokenizer and shared protocol, also saved with adapters."""
    return {"base_model": BASE_MODEL, "base_revision": BASE_REVISION,
            "tokenizer_revision": TOKENIZER_REVISION, "system_message": SYSTEM_MESSAGE,
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
            "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            "backend_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
            "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "assistant_eot_id": tokenizer.convert_tokens_to_ids("<|eot_id|>")}


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_whitespace(text):
    return " ".join(text.split())


def allocate_quota(sizes, total):
    """Hamilton allocation with integer arithmetic and lexical tie-breaking."""
    population = sum(sizes.values())
    if total < 0 or total > population:
        raise ValueError("Requested quota exceeds available population")
    if not population:
        return {key: 0 for key in sizes}
    quota = {key: total * size // population for key, size in sizes.items()}
    order = sorted(sizes, key=lambda key: (-(total * sizes[key] % population), key))
    for key in order[:total - sum(quota.values())]:
        quota[key] += 1
    return quota


def build_manifest(raw_ds, revision, seed=SEED, validation_size=MAX_EVAL_SAMPLES,
                   test_size=MAX_TEST_SAMPLES, train_fraction=TRAIN_FRACTION):
    """Index raw rows first; normalized strings are keys, never training text."""
    if not 0 <= train_fraction <= 1:
        raise ValueError("train_fraction must be between zero and one")
    pairs, groups, raw_hashes = {}, {}, []
    for index, row in enumerate(raw_ds):
        if not all(isinstance(row.get(k), str) for k in ("instruction", "output", "source")):
            raise ValueError(f"Invalid raw record at index {index}")
        q, a = normalize_whitespace(row["instruction"]), normalize_whitespace(row["output"])
        gid, pid = digest(q), digest([q, a])
        raw_hashes.append(digest({k: row[k] for k in ("instruction", "output", "source")}))
        pair = pairs.setdefault(pid, {"group_id": gid, "raw_indices": [], "source": row["source"]})
        pair["raw_indices"].append(index)
        group = groups.setdefault(gid, {"pair_ids": set(), "sources": set(), "split": "train"})
        group["pair_ids"].add(pid)
        group["sources"].add(row["source"])
    if len(groups) <= test_size + validation_size:
        raise ValueError("Not enough question groups for evaluation and training")
    strata = {}
    for gid, group in groups.items():
        group["pair_ids"] = sorted(group["pair_ids"])
        group["sources"] = sorted(group["sources"])
        strata.setdefault(canonical_json(group["sources"]), []).append(gid)
    initial_sizes = {key: len(ids) for key, ids in strata.items()}
    def ordered(ids, purpose):
        return sorted(ids, key=lambda identity: (digest([seed, purpose, identity]), identity))
    for split, size in (("test", test_size), ("validation", validation_size)):
        quotas = allocate_quota({key: len(ids) for key, ids in strata.items()}, size)
        for key in sorted(strata):
            ids = ordered(strata[key], split)
            chosen = ids[:quotas[key]]
            for gid in chosen:
                groups[gid]["split"] = split
            strata[key] = ids[quotas[key]:]
    train_pool = [pid for pid, pair in pairs.items() if groups[pair["group_id"]]["split"] == "train"]
    by_source = {}
    for pid in train_pool:
        by_source.setdefault(pairs[pid]["source"], []).append(pid)
    # Decimal avoids binary-float boundary surprises in floor(3% * N).
    subset_size = int(Decimal(str(train_fraction)) * len(train_pool))
    quotas = allocate_quota({key: len(ids) for key, ids in by_source.items()}, subset_size)
    subset = []
    for source in sorted(by_source):
        subset.extend(ordered(by_source[source], "subset")[:quotas[source]])
    subset = ordered(subset, "train_order")
    references = {"validation": [], "test": []}
    for gid in sorted(groups):
        group = groups[gid]
        if group["split"] in references:
            indices = sorted(pairs[pid]["raw_indices"][0] for pid in group["pair_ids"])
            references[group["split"]].append({"group_id": gid, "primary_reference": indices[0],
                                                "all_references": indices})
    return {
        "dataset_id": DATASET_ID, "revision": revision, "raw_count": len(raw_hashes),
        "snapshot_hash": digest(raw_hashes), "raw_hashes": raw_hashes,
        "normalization_version": "whitespace-v1", "normalization_rule": '" ".join(text.split())',
        "algorithm_version": "group-source-hamilton-sha256-v1",
        "settings": {"seed": seed, "validation_groups": validation_size,
                     "test_groups": test_size, "train_fraction": train_fraction},
        "pairs": dict(sorted(pairs.items())), "groups": dict(sorted(groups.items())),
        "training_subset": subset, "references": references,
        "stratum_sizes": dict(sorted(initial_sizes.items())),
    }


def load_or_create_manifest(raw_ds, revision, path, **settings):
    """Recompute expected structure to verify snapshot AND all manifest relations."""
    expected = build_manifest(raw_ds, revision, **settings)
    path = Path(path)
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual.get("snapshot_hash") != expected["snapshot_hash"] or actual.get("revision") != revision:
            raise ValueError("Dataset snapshot differs from manifest; refusing reuse/overwrite")
        if actual != expected:
            raise ValueError("Manifest integrity/settings mismatch; refusing reuse/overwrite")
        return actual
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(canonical_json(expected) + "\n", encoding="utf-8")
    temporary.replace(path)
    return expected


def cached_revision(raw_ds):
    # Datasets offline fallback may ignore the requested revision. Verify the
    # source URI recorded by the builder, never infer identities from Arrow paths.
    checksums = raw_ds.info.download_checksums or {}
    revisions = {match.group(1) for uri in checksums
                 if (match := re.fullmatch(
                     rf"hf://datasets/{re.escape(DATASET_ID)}@([0-9a-f]{{40}})/.+", uri))}
    if len(revisions) != 1:
        raise ValueError("Cannot establish a unique dataset revision from source metadata")
    revision = next(iter(revisions))
    if revision != DATASET_REVISION:
        raise ValueError("Resolved dataset revision differs from pinned revision")
    return revision


def prepared_from_manifest(raw_ds, tokenizer, manifest, *, return_metadata=False,
                           max_length=MAX_SEQ_LENGTH):
    from datasets import Dataset
    if type(max_length) is not int or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    selected = [{"pair_id": pid, "raw_index": manifest["pairs"][pid]["raw_indices"][0]}
                for pid in manifest["training_subset"]]
    validation = [{"group_id": ref["group_id"], "raw_index": ref["primary_reference"]}
                  for ref in manifest["references"]["validation"]]
    effective, excluded, lengths = [], [], []
    def formatted(identities, training):
        examples = []
        for identity in identities:
            row = raw_ds[identity["raw_index"]]
            batch = format_batch({"instruction": [row["instruction"]],
                                  "output": [row["output"]]}, tokenizer)
            example = {key: value[0] for key, value in batch.items()}
            prompt_ids = tokenizer.apply_chat_template(example["prompt"], tokenize=True, return_dict=False,
                add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS)
            ids = tokenizer.apply_chat_template(example["prompt"] + example["completion"],
                tokenize=True, return_dict=False, **CHAT_TEMPLATE_KWARGS)
            eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if ids[:len(prompt_ids)] != prompt_ids or len(ids) <= len(prompt_ids) + 1:
                raise ValueError(f"Invalid completion boundary/answer: {identity}")
            if ids[0] != tokenizer.bos_token_id or ids[-1] != eot:
                raise ValueError(f"Invalid BOS/assistant EOT: {identity}")
            if len(ids) > max_length:
                if not training:
                    raise ValueError(f"Validation exceeds max_length={max_length}: {identity}")
                excluded.append({**identity, "token_length": len(ids), "reason": "overlength"})
                continue
            if training:
                effective.append(identity)
                lengths.append(len(ids))
            examples.append(example)
        if not examples:
            raise ValueError("No valid effective samples")
        return Dataset.from_list(examples)
    eval_ds = formatted(validation, False)
    train_ds = formatted(selected, True)
    metadata = {"revision": manifest["revision"], "snapshot_hash": manifest["snapshot_hash"],
                "manifest_hash": digest(manifest), "max_length": max_length,
                "overlength_policy": "exclude_train_fail_validation",
                "tokenizer": tokenizer_provenance(tokenizer),
                "selected": selected, "effective": effective, "excluded": excluded,
                "validation": validation,
                "counts": {"selected": len(selected), "effective": len(effective),
                           "excluded": len(excluded), "validation": len(validation)},
                "effective_ids_hash": digest(effective),
                "effective_token_lengths": lengths, "token_lengths_hash": digest(lengths)}
    return (train_ds, eval_ds, metadata) if return_metadata else (train_ds, eval_ds)


def prepare_data(tokenizer, *, return_metadata=False):
    """Preserve manifest selections; build an explicit complete-answer effective subset."""
    raw_ds = load_dataset(DATASET_ID, revision=DATASET_REVISION)["train"]
    manifest = load_or_create_manifest(raw_ds, cached_revision(raw_ds), SPLIT_MANIFEST_PATH)
    result = prepared_from_manifest(raw_ds, tokenizer, manifest, return_metadata=return_metadata)
    print(f"Đã chuẩn bị MathInstruct: train={len(result[0])}, eval={len(result[1])}")
    return result


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=TOKENIZER_REVISION, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _, _, metadata = prepare_data(tokenizer, return_metadata=True)
    path = SPLIT_MANIFEST_PATH.parent / "effective_data.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Metadata: {path}")
