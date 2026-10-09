"""Explicit offline integration: preserves historical reports, no training/weights.
Run: PYTHONPATH=. python -B tests/verify_real_data.py
"""
import os
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["HF_DATASETS_OFFLINE"]="1"
os.environ["CUDA_VISIBLE_DEVICES"]=""
import hashlib
import json
import tempfile
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from trl import SFTConfig, SFTTrainer
from src.config import BASE_MODEL, DATASET_ID, DATASET_REVISION, TOKENIZER_REVISION, SPLIT_MANIFEST_PATH, CHAT_TEMPLATE_KWARGS
from src.data_prep import prepare_data, digest, build_manifest

def main():
    original = SPLIT_MANIFEST_PATH.read_bytes()
    assert hashlib.sha256(original).hexdigest() == "b83ac64a3ea90090a11b12febba9d645bd2e7c1406eabd282bfa70ca90411647"
    m = json.loads(original)
    raw = load_dataset(DATASET_ID, revision=DATASET_REVISION)["train"]
    assert build_manifest(raw, m["revision"]) == m
    t = AutoTokenizer.from_pretrained(BASE_MODEL, revision=TOKENIZER_REVISION, local_files_only=True)
    train, validation, meta = prepare_data(t, return_metadata=True)
    assert digest(meta["effective"]) == "92dc3e8c1d8be4b86d502fb4e355b5ef913896563a2eabb976579c82b11a0fc2"
    assert (t.bos_token_id, t.eos_token_id, t.pad_token_id) == (128000, 128009, 128004)
    assert len(meta["selected"])==7359 and len(train)==7310 and len(meta["excluded"])==49 and len(validation)==100
    assert [x["pair_id"] for x in meta["selected"]]==m["training_subset"]
    assert [x["raw_index"] for x in meta["validation"]]==[x["primary_reference"] for x in m["references"]["validation"]]
    assert all(x["token_length"]>1024 for x in meta["excluded"])
    assert all("/CoT/" in m["pairs"][x["pair_id"]]["source"] for x in meta["excluded"])
    assert [x for x in meta["selected"] if x["pair_id"] not in {r["pair_id"] for r in meta["excluded"]}]==meta["effective"]
    from src.inference import MathSolver
    solver = MathSolver.__new__(MathSolver)
    solver.tokenizer = t
    prompts = []
    solver.generator = lambda prompt, **kwargs: prompts.append((prompt, kwargs)) or [{"generated_text": "unused"}]
    model=LlamaForCausalLM(LlamaConfig(vocab_size=len(t),hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2))
    with tempfile.TemporaryDirectory() as tmp:
        args=SFTConfig(output_dir=tmp,use_cpu=True,bf16=False,fp16=False,report_to="none",optim="adamw_torch",gradient_checkpointing=False,max_length=1024,completion_only_loss=True,assistant_only_loss=False,packing=False,padding_free=False,disable_tqdm=True)
        trainer=SFTTrainer(model=model,args=args,train_dataset=train,eval_dataset=validation,processing_class=t)
        for original_ds, processed in [(train,trainer.train_dataset),(validation,trainer.eval_dataset)]:
            for start in range(0,len(processed),8):
                features=[processed[i] for i in range(start,min(start+8,len(processed)))]
                batch=trainer.data_collator(features)
                for j,feature in enumerate(features):
                    example=original_ds[start+j]
                    prefix=t.apply_chat_template(example["prompt"],tokenize=True, return_dict=False,add_generation_prompt=True,**CHAT_TEMPLATE_KWARGS)
                    ids=t.apply_chat_template(example["prompt"]+example["completion"],tokenize=True, return_dict=False,**CHAT_TEMPLATE_KWARGS)
                    identity = meta["effective"][start+j] if original_ds is train else meta["validation"][start+j]
                    solver.solve(raw[identity["raw_index"]]["instruction"])
                    rendered, generation = prompts.pop()
                    assert t(rendered, add_special_tokens=False)["input_ids"] == prefix
                    assert generation["add_special_tokens"] is False and generation["eos_token_id"] == ids[-1]
                    assert feature["input_ids"]==ids and ids[:len(prefix)]==prefix
                    assert feature["completion_mask"] == [0]*len(prefix)+[1]*(len(ids)-len(prefix))
                    assert batch["input_ids"][j][:len(ids)].tolist() == ids
                    assert batch["attention_mask"][j].tolist() == [1]*len(ids)+[0]*(batch["input_ids"].shape[1]-len(ids))
                    assert ids[0]==t.bos_token_id and ids.count(t.bos_token_id)==1
                    assert ids[-1]==t.convert_tokens_to_ids("<|eot_id|>") and len(ids)<=1024
                    labels=batch["labels"][j].tolist()
                    assert labels==[-100]*len(prefix)+ids[len(prefix):]+[-100]*(len(labels)-len(ids))
                    assert any(x!=-100 for x in labels[1:len(ids)-1])
    assert SPLIT_MANIFEST_PATH.read_bytes()==original
    print(json.dumps({"status":"PASS","selected":len(meta["selected"]),"effective":len(train),"excluded":len(meta["excluded"]),"validation":len(validation),"manifest_sha256":hashlib.sha256(original).hexdigest(),"effective_ids_hash":digest(meta["effective"]),"tokenizer":meta["tokenizer"],"actual_label_samples":len(train)+len(validation)},indent=2))

if __name__=="__main__":
    main()
