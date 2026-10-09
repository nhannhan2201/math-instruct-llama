"""Offline integration check: PYTHONPATH=. python tests/verify_real_data.py.

Runs complete preparation twice and writes docs/data_preparation_report.json.
Requires the existing dataset/tokenizer cache; never trains or downloads.
"""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_DATASETS_OFFLINE']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from src.config import BASE_MODEL, SPLIT_MANIFEST_PATH
from src.data_prep import prepare_data, canonical_json, digest

t=AutoTokenizer.from_pretrained(BASE_MODEL,local_files_only=True,use_fast=True)
if t.pad_token is None: t.pad_token=t.eos_token
results=[]
for run in range(2):
 train, validation=prepare_data(t)
 m=json.loads(SPLIT_MANIFEST_PATH.read_text())
 groups=m['groups']; pairs=m['pairs']
 splits={name:{gid for gid,g in groups.items() if g['split']==name} for name in ('train','validation','test')}
 assert not splits['train'] & splits['test'] and not splits['train'] & splits['validation'] and not splits['test'] & splits['validation']
 assert len(splits['test'])==500 and len(splits['validation'])==100
 assert train.column_names==validation.column_names==['text']
 results.append({'manifest_hash':digest(m),'manifest_file_hash':__import__('hashlib').sha256(SPLIT_MANIFEST_PATH.read_bytes()).hexdigest(), 'train_text_hash':digest(list(train['text'])), 'validation_text_hash':digest(list(validation['text'])), 'subset_hash':digest(m['training_subset']), 'test_references_hash':digest(m['references']['test'])})
 print('Run',run+1,'PASS',len(train),len(validation),results[-1]['manifest_hash'],flush=True)
assert results[0]==results[1]
def distribution(ids):
 source=Counter(pairs[pid]['source'] for pid in ids)
 kinds=Counter('CoT' if '/CoT/' in s else 'PoT' if '/PoT/' in s else 'UNKNOWN' for pid in ids for s in [pairs[pid]['source']])
 return {'records':len(ids),'source':dict(sorted(source.items())),'CoT_PoT':dict(kinds)}
pools={name:[pid for pid,p in pairs.items() if p['group_id'] in splits[name]] for name in splits}
dists={name:distribution(ids) for name,ids in pools.items()}
dists['training_subset']=distribution(m['training_subset'])
for name in ('validation','test'):
 refs=m['references'][name]
 primary_indices={r['primary_reference'] for r in refs}
 dists[name+'_primary']=distribution([pid for pid,p in pairs.items() if p['raw_indices'][0] in primary_indices])
group_dist={name:dict(sorted(Counter(canonical_json(groups[gid]['sources']) for gid in gids).items())) for name,gids in splits.items()}
report={'raw_records':m['raw_count'],'deduplicated_records':len(pairs),'removed_duplicate_records':m['raw_count']-len(pairs),'question_groups':len(groups),'split_groups':{k:len(v) for k,v in splits.items()},'distributions':dists,'group_source_combinations':group_dist,'stratum_sizes':m['stratum_sizes'],'revision':m['revision'],'snapshot_hash':m['snapshot_hash'],'manifest_bytes':SPLIT_MANIFEST_PATH.stat().st_size,'verification':{'status':'PASS','runs':results,'zero_question_overlap':True,'schema_compatible':True}}
Path('docs/data_preparation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('distributions','group_source_combinations','verification','stratum_sizes')},indent=2))
print({k:v['CoT_PoT'] for k,v in dists.items()})
