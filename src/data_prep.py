import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path

from datasets import load_dataset
from src.config import (
    DATASET_ID, MAX_EVAL_SAMPLES, TRAIN_FRACTION, 
    SEED, PROMPT_TEMPLATE, MAX_TEST_SAMPLES, SPLIT_MANIFEST_PATH
)

def format_batch(examples, tokenizer):
    """
    Format dataset theo cấu trúc Prompt Template đã định nghĩa.
    """
    texts = []
    for q, a in zip(examples["instruction"], examples["output"]):
        q = (q or "").strip()
        a = (a or "").strip()
        text = PROMPT_TEMPLATE.format(question=q, answer=a) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

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
    revisions = {part for entry in raw_ds.cache_files for part in Path(entry["filename"]).parts
                 if re.fullmatch(r"[0-9a-f]{40}", part)}
    if len(revisions) != 1:
        raise ValueError("Cannot establish a unique dataset revision from cache")
    return next(iter(revisions))


def prepared_from_manifest(raw_ds, tokenizer, manifest):
    train_indices = [manifest["pairs"][pid]["raw_indices"][0] for pid in manifest["training_subset"]]
    eval_indices = [item["primary_reference"] for item in manifest["references"]["validation"]]
    def formatted(indices):
        selected = raw_ds.select(indices)
        return selected.map(lambda batch: format_batch(batch, tokenizer), batched=True,
                            remove_columns=selected.column_names)
    return formatted(train_indices), formatted(eval_indices)


def prepare_data(tokenizer):
    """Dedup, group split, then sample; preserve the training-facing text schema."""
    raw_ds = load_dataset(DATASET_ID)["train"]
    manifest = load_or_create_manifest(raw_ds, cached_revision(raw_ds), SPLIT_MANIFEST_PATH)
    train_ds, eval_ds = prepared_from_manifest(raw_ds, tokenizer, manifest)
    print(f"Đã chuẩn bị MathInstruct: train={len(train_ds)}, eval={len(eval_ds)}, test groups={len(manifest['references']['test'])}")
    return train_ds, eval_ds
