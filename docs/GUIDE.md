# Math Instruct Llama — Guide

Cập nhật **2026-10-09**. Kết quả/approval/lịch sử nằm trong [PROGRESS.md](PROGRESS.md); [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) chỉ giữ lịch sử.

## Kiến trúc thực tế

`src/config.py` chứa identities và protocol; `data_prep.py` load raw snapshot → kiểm tra source revision → rebuild/validate manifest → chat prompt/completion → đo toàn sequence → loại train quá dài/fail validation. `train.py` đặt seed trước initialization, chuẩn bị dữ liệu, load base, gắn PEFT, chạy SFTTrainer và final validation, lưu adapter/tokenizer/provenance. `inference.py` kiểm tra artifact rồi dùng cùng prompt/stops. `app.py` load một solver qua lifespan và tuần tự hóa generation.

Dataset `TIGER-Lab/MathInstruct` pin `b4fdc323a7be1379c9c7c0b67b1de72dfee2111a`; base/tokenizer `unsloth/Llama-3.2-1B-Instruct` pin `5a8abab4a5d6f164389b1079fb721cfab8d7126c`. Pins đối chiếu snapshot/refs cache có sẵn; không tải pretrained weights trong implementation session. Source URI trong `Dataset.info.download_checksums` phải đúng pin. Offline Datasets có thể fallback cache mới nhất; revision/source và toàn manifest được kiểm tra để từ chối snapshot khác. Không phụ thuộc tên/path Arrow hoặc JSON reports lịch sử.

Dedup dùng whitespace-normalized question/output, giữ raw representative nhỏ nhất; distinct outputs cùng question nằm chung split. Test 500 groups, validation 100 groups, train sampling 3% sau split/dedup. Manifest vẫn rebuild từ raw rows để kiểm tra integrity; không đưa test vào Trainer. Snapshot: 262039 raw / 246002 pairs; selected **7359**, effective **7310**, excluded **49**, validation primary **100**. Manifest SHA256 `b83ac64a3ea90090a11b12febba9d645bd2e7c1406eabd282bfa70ca90411647`; effective IDs hash `92dc3e8c1d8be4b86d502fb4e355b5ef913896563a2eabb976579c82b11a0fc2`.

Chat gồm system tutor + user + assistant, fixed date `09 Oct 2026`; max length **1024**. Không truncate answer để vừa length: exclude train với IDs/reason/length; fail validation. TRL completion-only loss mask prompt/padding `-100`, answer/EOT có targets; BOS `128000`, assistant EOT/EOS `128009`, pad `128004`. Packing, padding-free và assistant-only loss tắt. Exact question isolation không chứng minh semantic leakage hoặc pretraining contamination; whitespace dedup cho PoT vẫn có rủi ro indentation.

## Môi trường và chuẩn bị dữ liệu

Chạy từ repository root, Python 3.10. [requirements.txt](../requirements.txt) pin direct dependencies theo installed metadata và CPU tests/imports đã PASS; không phải CUDA/transitive lockfile. Local runtime là **torch 2.11.0+cu130 / CUDA build 13.0**, chỉ chạy CPU vì không có GPU driver. **Không sao chép wheel này thành Kaggle recommendation.** Torch/CUDA wheel, driver, bitsandbytes kernels và khả năng đồng tồn tại packages Kaggle phải xác minh riêng; torch được provision trước requirements. Không tự chọn CUDA versions khi chưa có evidence. Local `pip check` phát hiện AWS packages có sẵn không tương thích: boto3 1.43.0 cần botocore>=1.43.0,<1.44.0 nhưng installed botocore1.42.91. Không thay environment trong session; MLflow local round-trip PASS, S3/clean dependency environment chưa verified.

```bash
conda activate math-instruct-llama
# Sau khi provision torch phù hợp cho host:
python -m pip install -r requirements.txt
python -m pip check
```

Kaggle cần quyền truy cập đúng model revision, raw snapshot hoặc HF cache, và working directory ghi được. Với offline transfer, copy cached dataset builder tree vào `$HF_DATASETS_CACHE`, tokenizer snapshot/blobs/refs vào `$HF_HUB_CACHE`, và `data/split_manifest.json` vào repository root. Không transfer legacy adapter để giả lập provenance. Verify SHA256 manifest trên host nhận. Nếu chưa có manifest, production tạo deterministically và không overwrite khi mismatch.

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
# Cache phải ghi được vì datasets cần lock/map cache:
# export HF_DATASETS_CACHE=/path/to/writable/dataset-cache
# export HF_HUB_CACHE=/path/to/hub-cache
python -B -m src.data_prep
python -B -m unittest discover -s tests -v
PYTHONPATH=. python -B tests/verify_real_data.py
```

Data preparation chỉ load tokenizer/data, ghi `data/effective_data.json` và manifest nếu thiếu; không model weights. Manifest-only tests không cần tokenizer cache; schema/label/parity tests và integration cần tokenizer cache. Integration cần toàn raw snapshot, kiểm tra rebuild manifest, counts/order/overlength và **7410** real label/parity samples; không ghi reports. Cache thiếu phải fail offline. Tests CPU có một smoke trên tiny random model, không pretrained weights; model đó xác nhận step cap và adapter update, không chứng minh real-model/CUDA loss.

Trong sandbox session đã dùng `/home/nhan/miniconda3/envs/math-instruct-llama/bin/python`, copy dataset cache vào `/tmp/math-instruct-hf/datasets` do original cache read-only. Đây là adaptation môi trường, integration không hardcode path này.

## Training và Kaggle smoke

Full mode giữ seed42, rank4/alpha8/dropout0.05, target q/k/v/o + gate/up/down projections, LR2e-4, epoch1, microbatch1, accumulation32, cosine/warmup5 và paged_adamw_8bit. Đây là starting config, chưa benchmark tối ưu. TinyLoRA vẫn là CLI lịch sử, ngoài comparison và chưa verified GPU.

Chỉ chấp nhận **một visible CUDA GPU**. T4 dùng FP16; chỉ chọn BF16 khi `is_bf16_supported(including_emulation=False)` xác nhận hỗ trợ native. Default True có thể tính cả emulation trên T4. QLoRA NF4/double quant có cùng compute dtype và phải thật sự loaded-in-4bit. `use_cache=False` khi train; gradient checkpointing explicit True, `use_reentrant=False` đồng nhất ở kbit helper và Trainer. Những settings này đã kiểm tra CPU contract, chưa bảo đảm VRAM/kernel/runtime Kaggle.

Trước load weights, chạy inventory/imports trên Kaggle và lưu kết quả:

```bash
nvidia-smi
python -m pip check
python - <<'PY'
from importlib.metadata import version
import torch, transformers, trl, peft, accelerate, datasets, bitsandbytes, mlflow
print({p: version(p) for p in ('torch','transformers','trl','peft','accelerate','datasets','bitsandbytes','mlflow')})
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported(including_emulation=False))
PY
```

Sau khi GPU/model access được cho phép, provision weights đúng BASE_REVISION trên Kaggle. Chỉ bật offline nếu cả weights đã cached; nếu cho phép Hub download ở lượt GPU, bỏ HF_HUB_OFFLINE/HF_DATASETS_OFFLINE trước training. Chạy từng method riêng (session implementation này **không chạy**):

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train --method lora --smoke
CUDA_VISIBLE_DEVICES=0 python -m src.train --method qlora --smoke
```

Smoke lấy deterministic prefix **64 effective IDs**, giữ accumulation32/microbatch1, `max_steps=2`, không scheduled save/eval, rồi final evaluate toàn 100 validation. Logging mỗi step. Warmup5 giữ nguyên nên step đầu LR0; smoke là correctness check, không quality comparison. Trainer step count phải đúng2, gradients/loss finite, adapter update norm >0, base frozen bằng trainable-parameter guard; QLoRA kiểm tra actual four-bit flag. CPU tests đã xác nhận 2 steps trên model ngẫu nhiên và wiring cả LoRA/QLoRA; CUDA optimizer và real weights còn mở.

Mỗi run có output riêng `models/<method>_<smoke|training>_<unique-id>/`; không overwrite legacy final hoặc run khác. Final nằm ở `final/`, checkpoints ở `checkpoints/checkpoint-N/`. Cả final/checkpoints có adapter config/weights, tokenizer/template và `effective_data.json`: identities/counts/lengths/hashes, revisions, dependencies, training config, method/precision, GPU/CUDA, MLflow run ID. Metadata population effective vẫn7310; `training_selection` chỉ rõ prefix64 ở smoke.

MLflow dùng một explicit run, SQLite `mlruns.db`; callbacks log loss/gradient và checkpoint provenance, không bật Trainer MLflow integration thứ hai. Log final loss, adapter update norm smoke, trainable count, GPU identity, thời gian **train + final eval** (không tính load/save), peak allocated/reserved trong cùng khoảng này. Final artifact upload vào run. `models/`, `data/`, `mlruns/`, DB/WAL/SHM đều gitignored. Download/copy cả artifact và DB/MLflow artifacts ra khỏi Kaggle trước khi session mất; không chỉ copy adapter weights.

Full training chỉ khi được duyệt riêng: bỏ `--smoke`; `max_steps=-1`, epoch1, scheduled eval/save40, logging10 và final evaluate vẫn bắt buộc. Không có answer accuracy evaluator/test benchmark trong scope này.

## Reload, API và Docker

Dùng chính path được train in ra, sau GPU smoke:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.inference --method lora \
  --adapter-path models/lora_smoke_<id>/final --greedy
# QLoRA tương tự, đổi method/path.
```

Artifact tokenizer được ưu tiên; fallback chỉ dùng pinned local base tokenizer và fingerprint khớp. Guard kiểm tra template/backend/token IDs, revisions, method và adapter config; legacy thiếu provenance bị từ chối, không tự tạo metadata giả. Saved precision được dùng khi reload; BF16 artifact cần GPU hỗ trợ. `use_cache=True`, model eval, cùng chat prompt/EOT stop, không append special tokens lần nữa. Demo default sampling150/.7 giữ nguyên; `--greedy` tắt sampling. Chưa chứng minh real-model reload logits/generation parity.

```bash
export MATH_METHOD=lora
export MATH_ADAPTER_PATH=/absolute/path/to/verified/final
uvicorn app:app --host 0.0.0.0 --port 8000
```

API lifespan load một model; `GET /` và `/solve` trả503 khi unavailable. Request blank/oversize hoặc tokens ngoài1..512/temperature ngoài(0,2] trả422. Generation có lock; lỗi trả500 với thông điệp chung, chi tiết chỉ server logs. API tests dùng mocks, chưa actual GPU serving/concurrency load test.

Docker dùng `--build-arg RUNTIME_IMAGE=<verified-pytorch-runtime-image>` trên serving host; image cần Python/dependencies tương thích. Không chọn CUDA image chưa xác minh; không bake models vào image, mount artifact/cache và truyền env như trên. Docker build/runtime chưa chạy. UI chưa có source; Gradio chỉ là dependency lịch sử.

Hai inspection scripts và bốn historical JSON reports đã xóa sau tests PASS: obsolete schema, cache paths không portable, dữ liệu cũ không phải current evidence. Summary/hashes/lịch sử giữ trong PROGRESS và Git; muốn tái tạo lịch sử phải checkout đúng archived commit. Không tạo reports phân tích mới trong docs.
