"""Offline Task 3A analysis. Never changes production data or runs model forward."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import hashlib
import json
import tempfile
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from trl import SFTConfig, SFTTrainer
from src.config import BASE_MODEL, SPLIT_MANIFEST_PATH, PROMPT_TEMPLATE
from src.data_prep import digest, format_batch

ROOT = Path(__file__).resolve().parents[1]
LIMITS = [128, 256, 512, 1024, 2048]
SYSTEM = 'You are a helpful math tutor.\nSolve the problem with clear reasoning and give a concise final answer.'
CHAT_KWARGS = {'date_string': '09 Oct 2026'}


def stats(values):
    return dict(zip(['min', 'median', 'p90', 'p95', 'p99', 'max'],
                    map(float, np.percentile(values, [0, 50, 90, 95, 99, 100]))))


def representation(row, tokenizer, mode):
    q, a = row['instruction'].strip(), row['output'].strip()
    if mode == 'current':
        text = format_batch({'instruction': [q], 'output': [a]}, tokenizer)['text'][0]
        prefix = PROMPT_TEMPLATE.format(question=q, answer='')
        example = {'text': text}
        encoded = tokenizer(text, return_offsets_mapping=True)
        pids = tokenizer(prefix)['input_ids']
    else:
        prompt = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': q}]
        completion = [{'role': 'assistant', 'content': a}]
        example = {'prompt': prompt, 'completion': completion, 'chat_template_kwargs': CHAT_KWARGS}
        prefix = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, **CHAT_KWARGS)
        text = tokenizer.apply_chat_template(prompt + completion, tokenize=False, **CHAT_KWARGS)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        pids = tokenizer.apply_chat_template(prompt, tokenize=True, return_dict=True,
                    add_generation_prompt=True, **CHAT_KWARGS)['input_ids']
    start = len(prefix)
    assert text.startswith(prefix)
    end = start + len(a)
    ids = encoded['input_ids']
    spans = encoded['offset_mapping']
    answer_positions = [i for i, (x, y) in enumerate(spans) if x < end and y > start]
    prompt_positions = [i for i, (x, y) in enumerate(spans) if y > x and y <= start]
    stop_position = len(ids) - 1
    expected_stop = tokenizer.eos_token_id if mode == 'current' else tokenizer.convert_tokens_to_ids('<|eot_id|>')
    assert ids[-1] == expected_stop
    return {'ids': ids, 'pids': pids, 'answer_positions': answer_positions,
            'prompt_positions': prompt_positions, 'stop_position': stop_position,
            'prefix_match': ids[:len(pids)] == pids, 'example': example,
            'lengths': {'prompt': len(pids), 'answer': len(answer_positions),
                        'total': len(ids), 'prompt_separate': len(pids),
                        'other': len(ids) - len(pids) - len(answer_positions)}}


def summarize(records):
    n = len(records)
    output = {'records': n, 'lengths': {k: stats([r['lengths'][k] for r in records])
                                      for k in records[0]['lengths']},
              'prefix_mismatches': sum(not r['prefix_match'] for r in records),
              'empty_answer_tokens': sum(not r['answer_positions'] for r in records), 'limits': {}}
    for limit in LIMITS:
        counts = Counter()
        lens = []
        for r in records:
            positions = r['answer_positions']
            retained = sum(i < limit for i in positions)
            counts['sample_truncated'] += len(r['ids']) > limit
            counts['answer_partial'] += 0 < retained < len(positions)
            counts['answer_lost'] += bool(positions) and retained == 0
            counts['stop_lost'] += r['stop_position'] >= limit
            counts['answer_tokens_retained'] += retained
            counts['completion_targets_after_shift'] += sum(0 < i < limit for i in range(len(r['pids']), len(r['ids'])))
            lens.append(min(limit, len(r['ids'])))
        useful = sum(lens)
        batches = {}
        for batch_size in [1, 4, 8]:
            padded = sum(max(lens[i:i+batch_size]) * len(lens[i:i+batch_size])
                         for i in range(0, n, batch_size))
            batches[str(batch_size)] = {'padded_slots': padded, 'padding_slots': padded-useful,
                                       'padding_fraction': (padded-useful)/padded}
        output['limits'][str(limit)] = {'counts': dict(counts),
            'rates': {k: counts[k]/n for k in ['sample_truncated', 'answer_partial', 'answer_lost', 'stop_lost']},
            'retained_token_volume': useful, 'full_token_volume': sum(len(r['ids']) for r in records),
            'intact_records': sum(len(r['ids']) <= limit for r in records),
            'intact_only_token_volume': sum(len(r['ids']) for r in records if len(r['ids']) <= limit),
            'fixed_max_padding_slots': n*limit-useful,
            'fixed_max_padding_fraction': (n*limit-useful)/(n*limit),
            'ordered_dynamic_batches': batches}
    return output


def verify(records, tokenizer, mode):
    # Real examples within the supplied train/validation selection; never test.
    chosen = set([0, min(range(len(records)), key=lambda i: len(records[i]['ids'])),
                  max(range(len(records)), key=lambda i: len(records[i]['ids']))])
    for limit in LIMITS:
        for predicate in [lambda r: r['stop_position'] < limit,
                          lambda r: r['answer_positions'] and min(r['answer_positions']) < limit <= max(r['answer_positions']),
                          lambda r: r['answer_positions'] and min(r['answer_positions']) >= limit]:
            index = next((i for i, r in enumerate(records) if predicate(r)), None)
            if index is not None:
                chosen.add(index)
    chosen = sorted(chosen)
    data = Dataset.from_list([records[i]['example'] for i in chosen])
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tokenizer), hidden_size=8,
        intermediate_size=16, num_hidden_layers=1, num_attention_heads=1,
        num_key_value_heads=1, max_position_embeddings=4096,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id))
    evidence = []
    with tempfile.TemporaryDirectory(dir='/tmp', prefix='task3a-') as directory:
        args = SFTConfig(output_dir=directory, use_cpu=True, bf16=False, fp16=False,
            report_to='none', optim='adamw_torch', gradient_checkpointing=False,
            max_length=128, packing=False, padding_free=False,
            completion_only_loss=mode == 'chat', assistant_only_loss=False,
            dataset_num_proc=None, disable_tqdm=True)
        trainer = SFTTrainer(model=model, args=args, train_dataset=data,
            processing_class=tokenizer,
            formatting_func=(lambda example: example['text']) if mode == 'current' else None)
        features = [trainer.train_dataset[i] for i in range(len(chosen))]
        for i, feature in enumerate(features):
            r = records[chosen[i]]
            assert feature['input_ids'] == r['ids']
            if mode == 'chat':
                assert feature['completion_mask'] == [int(j >= len(r['pids'])) for j in range(len(r['ids']))]
        for truncation in ['keep_start', 'keep_end']:
            for limit in LIMITS:
                trainer.data_collator.max_length = limit
                trainer.data_collator.truncation_mode = truncation
                batch = trainer.data_collator(features)
                for i, index in enumerate(chosen):
                    r = records[index]
                    start = 0 if truncation == 'keep_start' else max(0, len(r['ids'])-limit)
                    expected = r['ids'][start:start+limit]
                    size = len(expected)
                    ids = batch['input_ids'][i].tolist()
                    labels = batch['labels'][i].tolist()
                    attention = batch['attention_mask'][i].tolist()
                    assert ids[:size] == expected and attention == [1]*size+[0]*(len(ids)-size)
                    wanted = [token if mode == 'current' or start+j >= len(r['pids']) else -100
                              for j, token in enumerate(expected)] + [-100]*(len(ids)-size)
                    assert labels == wanted
                    evidence.append({'selection_index': index, 'max_length': limit, 'truncation': truncation,
                        'input_ids': ids, 'attention_mask': attention, 'labels': labels,
                        'answer_targets_after_shift': sum(labels[j-start] != -100 for j in r['answer_positions']
                                                         if start < j < start+size),
                        'stop_target': r['stop_position'] >= start and r['stop_position'] < start+size
                                       and labels[r['stop_position']-start] != -100,
                        'padding_tokens': len(ids)-size})
                singles = [trainer.data_collator([f]) for f in features]
                assert all(b['labels'][0].tolist() == batch['labels'][i][:int(b['attention_mask'][0].sum())].tolist()
                           for i, b in enumerate(singles))
        incompatible_formatter_error = None
        if mode == 'chat':
            try:
                SFTTrainer(model=model, args=args, train_dataset=data,
                    processing_class=tokenizer, formatting_func=lambda example: 'unused')
            except ValueError as exc:
                incompatible_formatter_error = str(exc)
            assert incompatible_formatter_error and 'incompatible' in incompatible_formatter_error
    return {'status': 'PASS', 'selected_indices': chosen,
            'preprocessing_exact_ids': True, 'microbatch_one_matches': True,
            'formatting_func_with_completion_only_error': incompatible_formatter_error,
            'collator': type(trainer.data_collator).__name__, 'batches': evidence}


def main():
    assert version('trl') == '1.3.0' and version('transformers') == '5.7.0'
    manifest_bytes = SPLIT_MANIFEST_PATH.read_bytes()
    m = json.loads(manifest_bytes)
    # Read existing Arrow files directly: avoids cache lock writes and Hub resolution.
    cached = json.loads((ROOT/'docs/data_inspection_report.json').read_text())['dataset_cache_files']
    assert all(m['revision'] in entry['filename'] for entry in cached)
    raw = concatenate_datasets([Dataset.from_file(entry['filename']) for entry in cached])
    assert len(raw) == m['raw_count']
    hashes = [digest({k: r[k] for k in ['instruction', 'output', 'source']}) for r in raw]
    assert hashes == m['raw_hashes'] and digest(hashes) == m['snapshot_hash']
    indices = {'train': [m['pairs'][pid]['raw_indices'][0] for pid in m['training_subset']],
               'validation': [r['primary_reference'] for r in m['references']['validation']]}
    assert len(indices['train']) == 7359 and len(indices['validation']) == 100
    assert not set(indices['train']) & set(indices['validation'])
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    report = {'versions': {n: version(n) for n in ['trl', 'transformers', 'torch', 'datasets']},
        'provenance': {'revision': m['revision'], 'snapshot_hash': m['snapshot_hash'],
            'manifest_file_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
            'selected_raw_indices': indices, 'test_used': False, 'all_raw_hashes_verified': True,
            'tokenizer_id': BASE_MODEL, 'chat_template_kwargs': CHAT_KWARGS,
            'eos_token': tokenizer.eos_token, 'eos_token_id': tokenizer.eos_token_id,
            'pad_token_id': tokenizer.pad_token_id, 'padding_side': tokenizer.padding_side,
            'assistant_eot_id': tokenizer.convert_tokens_to_ids('<|eot_id|>'),
            'template_has_generation_mask_blocks': '{% generation' in tokenizer.chat_template,
            'chat_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            'tokenizer_backend_sha256': hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()},
        'definitions': {'prompt': 'Length of separately tokenized prompt, including BOS/header; full-sample prefix equality checked for every row.',
            'answer': 'Full-sample tokens overlapping original stripped answer character span.',
            'other': 'Remaining tokens, terminal EOS/EOT for these data; boundary overlaps assigned to answer.',
            'prompt_separate': 'Prompt separately tokenized; used by TRL completion mask, may differ at boundary.',
            'stop_lost': 'Appended eos_token for current; assistant eot_id for chat. Not necessarily the same token.',
            'padding': 'CPU arithmetic on keep_start capped lengths; dynamic batches in manifest order, no packing or pad multiple.',
            'gpu': 'No measured GPU VRAM or speed; token slots are workload estimates, not VRAM predictions.'},
        'formats': {}, 'actual_labels': {}}
    for mode in ['current', 'chat']:
        report['formats'][mode] = {}
        report['actual_labels'][mode] = {}
        for split, selection in indices.items():
            records = []
            for index in selection:
                row = raw[index]
                r = representation(row, tokenizer, mode)
                r['kind'] = 'CoT' if '/CoT/' in row['source'] else 'PoT' if '/PoT/' in row['source'] else 'UNKNOWN'
                records.append(r)
            report['formats'][mode][split] = {'all': summarize(records)}
            for kind in sorted({r['kind'] for r in records}):
                report['formats'][mode][split][kind] = summarize([r for r in records if r['kind'] == kind])
            print(mode, split, len(records), report['formats'][mode][split]['all']['lengths']['total'], flush=True)
            report['actual_labels'][mode][split] = verify(records, tokenizer, mode)
    assert SPLIT_MANIFEST_PATH.read_bytes() == manifest_bytes
    output = ROOT/'docs/task3a_analysis_report.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print('PASS: actual labels, raw snapshot, manifest unchanged. Report:', output, flush=True)


if __name__ == '__main__':
    main()
