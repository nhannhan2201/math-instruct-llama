# Math Instruct Llama — Guide

Cập nhật **2026-10-09**. Kết quả/approval/lịch sử nằm trong [PROGRESS.md](PROGRESS.md); [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) chỉ giữ lịch sử.

## Kiến trúc thực tế

`src/config.py` chứa identities và protocol; `data_prep.py` load raw snapshot → kiểm tra source revision → rebuild/validate manifest → chat prompt/completion → đo toàn sequence → loại train quá dài/fail validation. `train.py` đặt seed trước initialization, chuẩn bị dữ liệu, load base, gắn PEFT, chạy SFTTrainer và final validation, lưu adapter/tokenizer/provenance. `inference.py` kiểm tra artifact rồi dùng cùng prompt/stops. `app.py` load một solver qua lifespan và tuần tự hóa generation.

Dataset `TIGER-Lab/MathInstruct` pin `b4fdc323a7be1379c9c7c0b67b1de72dfee2111a`; base/tokenizer `unsloth/Llama-3.2-1B-Instruct` pin `5a8abab4a5d6f164389b1079fb721cfab8d7126c`. Pins đối chiếu snapshot/refs cache có sẵn; không tải pretrained weights trong implementation session. Khi `Dataset.info.download_checksums` có source URI, revision trong URI phải đúng pin. Online JSON builder có thể trả null; khi đó requested revision chỉ được chấp nhận sau khi ordered raw-content hash khớp `DATASET_SNAPSHOT_HASH` đã pin độc lập. Content hash được kiểm tra trước cả tạo manifest mới hoặc reuse manifest cũ. Offline Datasets có thể fallback cache mới nhất; content pin và manifest integrity ngăn chấp nhận snapshot khác ngay cả khi source metadata thiếu. Không phụ thuộc tên/path Arrow hoặc JSON reports lịch sử.

Dedup dùng whitespace-normalized question/output, giữ raw representative nhỏ nhất; distinct outputs cùng question nằm chung split. Test 500 groups, validation 100 groups, train sampling 3% sau split/dedup. Manifest vẫn rebuild từ raw rows để kiểm tra integrity; không đưa test vào Trainer. Snapshot: 262039 raw / 246002 pairs; selected **7359**, effective **7310**, excluded **49**, validation primary **100**. Manifest SHA256 `b83ac64a3ea90090a11b12febba9d645bd2e7c1406eabd282bfa70ca90411647`; effective IDs hash `92dc3e8c1d8be4b86d502fb4e355b5ef913896563a2eabb976579c82b11a0fc2`.

Chat gồm system tutor + user + assistant, fixed date `09 Oct 2026`; max length **1024**. Không truncate answer để vừa length: exclude train với IDs/reason/length; fail validation. TRL completion-only loss mask prompt/padding `-100`, answer/EOT có targets; BOS `128000`, assistant EOT/EOS `128009`, pad `128004`. Packing, padding-free và assistant-only loss tắt. Exact question isolation không chứng minh semantic leakage hoặc pretraining contamination; whitespace dedup cho PoT vẫn có rủi ro indentation.

## Môi trường và chuẩn bị dữ liệu

Chạy từ repository root. CPU tests dùng Python 3.10; notebook Kaggle dùng Python 3.13.15. [requirements.txt](../requirements.txt) pin direct dependencies theo installed metadata và CPU tests/imports đã PASS; không phải CUDA/transitive lockfile. Local runtime là **torch 2.11.0+cu130 / CUDA build 13.0**, chỉ chạy CPU vì không có GPU driver. **Không sao chép wheel này thành Kaggle recommendation.** Kaggle đã chạy LoRA/QLoRA smoke trên torch 2.11.0+cu128 / CUDA build 12.8, driver 580.178.04, Tesla T4 capability 7.5, native BF16=False. NF4/paged optimizer đã chạy trong smoke; đây là evidence cho stack đã thử, chưa phải CUDA lockfile hoặc full training proof. Torch được provision trước requirements. Không tự chọn CUDA versions khi chưa có evidence. Local `pip check` phát hiện AWS packages có sẵn không tương thích: boto3 1.43.0 cần botocore>=1.43.0,<1.44.0 nhưng installed botocore1.42.91. Không thay environment trong session; MLflow local round-trip PASS, S3/clean dependency environment chưa verified.

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

## Lưu ý môi trường Kaggle đã thử

Cài dependencies bằng `python -m pip install -r requirements.txt`; training tự load data/model, không cần cài từng package riêng. Notebook đã gỡ **preinstalled torchao 0.10.0** bằng `python -m pip uninstall -y torchao`: PEFT 0.19.1 phát hiện optional torchao quá cũ, trong khi QLoRA của repo dùng bitsandbytes. Đây là workaround môi trường đã thực hiện, chưa được tự động hóa trong repo; không thêm torchao/CUDA pins mới. User đã yêu cầu dừng thay đổi dependencies. Fresh Kaggle clone + requirements trên image chưa chỉnh vẫn còn gate này.

Install returncode0 và imports PASS không đồng nghĩa `pip check` PASS. Kaggle còn conflicts: Diffusers với pinned huggingface_hub, PyOpenSSL với cryptography, cùng packages có sẵn như google-colab/bigframes/dopamine-rl/moviepy. Chưa kiểm chứng các integrations đó. Warnings HF_TOKEN, bitsandbytes FutureWarning và generation configuration trong successful smoke không làm run thất bại; không sửa dependencies chỉ để xóa warnings.

## Training và Kaggle smoke

Full mode giữ seed42, rank4/alpha8/dropout0.05, target q/k/v/o + gate/up/down projections, LR2e-4, epoch1, microbatch1, accumulation32, cosine/warmup5 và paged_adamw_8bit. Đây là starting config, chưa benchmark tối ưu. TinyLoRA vẫn là CLI lịch sử, ngoài comparison và chưa verified GPU.

Chỉ chấp nhận **một visible CUDA GPU**. T4 dùng FP16; chỉ chọn BF16 khi `is_bf16_supported(including_emulation=False)` xác nhận hỗ trợ native. Default True có thể tính cả emulation trên T4. QLoRA NF4/double quant có cùng compute dtype và phải thật sự loaded-in-4bit. TRL1.3.0 tự cast trainable quantized-model adapters sang BF16; ở FP16 mode, repo chuyển riêng trainable adapters về FP32 ngay sau Trainer initialization để FP16 GradScaler hoạt động. Frozen NF4 base không bị cast; BF16 native mode và LoRA không đổi. Snapshot kiểm tra adapter update lấy sau mọi dtype changes, để không tính rounding do cast thành optimizer update. Artifact metadata ghi `trainable_dtypes` riêng với compute precision. `use_cache=False` khi train; gradient checkpointing explicit True, `use_reentrant=False` đồng nhất ở kbit helper và Trainer. CPU contracts và Kaggle T4 runtime smoke đã PASS. Peak VRAM, resource metrics và full training feasibility chưa được trích xuất/đánh giá.

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

Sau khi GPU/model access được cho phép, provision weights đúng BASE_REVISION trên Kaggle. Chỉ bật offline nếu cả weights đã cached; nếu cho phép Hub download ở lượt GPU, bỏ HF_HUB_OFFLINE/HF_DATASETS_OFFLINE trước training. Chạy từng method riêng; user đã chạy cả hai smoke trên Kaggle, agent không chạy GPU local:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train --method lora --smoke
CUDA_VISIBLE_DEVICES=0 python -m src.train --method qlora --smoke
```

Smoke lấy deterministic prefix **64 effective IDs**, giữ accumulation32/microbatch1, `max_steps=2`, không scheduled save/eval, rồi final evaluate toàn 100 validation. Logging mỗi step. Warmup5 giữ nguyên nên step đầu LR0; smoke là correctness check, không quality comparison. Trainer step count phải đúng2, gradients/loss finite, adapter update norm >0, base frozen bằng trainable-parameter guard; QLoRA kiểm tra actual four-bit flag. CPU tests đã xác nhận 2 steps trên model ngẫu nhiên và wiring cả LoRA/QLoRA; Notebook mới xác nhận cả hai methods train2/2, evaluate100/100 và save trên real weights/T4; không phải benchmark chất lượng.

Mỗi run có output riêng `models/<method>_<smoke|training>_<unique-id>/`; không overwrite legacy final hoặc run khác. Final nằm ở `final/`, checkpoints ở `checkpoints/checkpoint-N/`. Cả final/checkpoints có adapter config/weights, tokenizer/template và `effective_data.json`: identities/counts/lengths/hashes, revisions, dependencies, training config, method/precision, GPU/CUDA, MLflow run ID. Metadata population effective vẫn7310; `training_selection` chỉ rõ prefix64 ở smoke.

MLflow dùng một explicit run, SQLite `mlruns.db`; callbacks log loss/gradient và checkpoint provenance, không bật Trainer MLflow integration thứ hai. Log final loss, adapter update norm smoke, trainable count, GPU identity, thời gian **train + final eval** (không tính load/save), peak allocated/reserved trong cùng khoảng này. Final artifact upload vào run. `models/`, `data/`, `mlruns/`, DB/WAL/SHM đều gitignored. Download/copy cả artifact và DB/MLflow artifacts ra khỏi Kaggle trước khi session mất; không chỉ copy adapter weights.

Full training chỉ khi được duyệt riêng: bỏ `--smoke`; `max_steps=-1`, epoch1, scheduled eval/save40, logging10 và final evaluate vẫn bắt buộc. Không có answer accuracy evaluator/test benchmark trong scope này.

## Reload, API và Docker

Dùng chính path được train in ra. Các path dưới đây thuộc notebook đã đọc, chỉ dùng khi artifacts vẫn còn trong Kaggle working directory. QLoRA path đã reload PASS; LoRA path mới nhất cần kiểm tra tiếp. Không gõ literal `<id>` trong shell vì dấu `<` là redirection:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.inference --method lora \
  --adapter-path models/lora_smoke_6077252db8b2/final --greedy
CUDA_VISIBLE_DEVICES=0 python -m src.inference --method qlora \
  --adapter-path models/qlora_smoke_c27830d96b37/final --greedy
```

Artifact tokenizer được ưu tiên; fallback chỉ dùng pinned local base tokenizer và fingerprint khớp. Guard kiểm tra template/backend/token IDs, revisions, method và adapter config; legacy thiếu provenance bị từ chối, không tự tạo metadata giả. Saved precision được dùng khi reload; BF16 artifact cần GPU hỗ trợ. `use_cache=True`, model eval, cùng chat prompt/EOT stop, không append special tokens lần nữa. Demo default sampling150/.7 giữ nguyên; `--greedy` tắt sampling. Notebook đã reload/generate QLoRA artifact mới và LoRA artifact cũ `models/lora_smoke_dd18ef5ce227/final`; chưa kiểm tra numerical logits tolerance trước/sau save. Demo một câu apple có final answer $1.50; QLoRA có lỗi diễn đạt trong reasoning, không suy ra answer accuracy.

```bash
export MATH_METHOD=lora
export MATH_ADAPTER_PATH=/absolute/path/to/verified/final
uvicorn app:app --host 0.0.0.0 --port 8000
```

API lifespan load một model; `GET /` và `/solve` trả503 khi unavailable. Request blank/oversize hoặc tokens ngoài1..512/temperature ngoài(0,2] trả422. Generation có lock; lỗi trả500 với thông điệp chung, chi tiết chỉ server logs. API tests dùng mocks, chưa actual GPU serving/concurrency load test.

Docker dùng `--build-arg RUNTIME_IMAGE=<verified-pytorch-runtime-image>` trên serving host; image cần Python/dependencies tương thích. Không chọn CUDA image chưa xác minh; không bake models vào image, mount artifact/cache và truyền env như trên. Docker build/runtime chưa chạy. UI chưa có source; Gradio chỉ là dependency lịch sử.

Hai inspection scripts và bốn historical JSON reports đã xóa sau tests PASS: obsolete schema, cache paths không portable, dữ liệu cũ không phải current evidence. Summary/hashes/lịch sử giữ trong PROGRESS và Git; muốn tái tạo lịch sử phải checkout đúng archived commit. Không tạo reports phân tích mới trong docs.

## Backup smoke artifacts (tùy chọn)

User xác nhận không cần giữ adapters smoke: chúng chỉ phục vụ kiểm tra pipeline, adapter cho ứng dụng chat sẽ lấy từ full training sau này. Có thể đóng Kaggle mà không backup. Notebook đã download và PROGRESS giữ evidence của lượt smoke; metrics/artifacts chưa inspect sẽ không còn kiểm chứng được nếu session mất.

Chỉ khi muốn giữ artifacts để debug hoặc kiểm tra thêm, chạy cell sau rồi tải ZIP trong Output:

```python
%cd /kaggle/working/math-instruct-llama
!zip -qr /kaggle/working/math-instruct-smoke-backup.zip models mlruns mlruns.db data/split_manifest.json
```

Archive giữ adapter/tokenizer/provenance, manifest và MLflow store; không cần đóng gói base weights/cache. Chưa kiểm tra ZIP/download hoặc khả năng mở MLflow khi chuyển host (artifact URIs có thể cần điều chỉnh). Section 15 của PROGRESS ghi đúng artifacts, commit, kết quả và các bước tiếp tục. Không tự chạy full training khi smoke PASS.

## Notebook gọn và CLI smoke/MLflow (2026-10-10)

`tune-peft.ipynb` chỉ clone/cập nhật HTTPS GitHub main, install/preflight pip check rồi gọi `python -m src.smoke_review`. CLI phải được commit/push lên main trước khi notebook clone; không pin commit. Notebook cũ có outputs vẫn được giữ ở `tune-peft.smoke-evidence.ipynb` để bảo toàn evidence. Bản notebook dài đã được thay thế.

`src.smoke_review` ghi actual HEAD, kiểm tra source sạch/không đổi, chạy LoRA/QLoRA smoke và reload qua CLI repo, rồi dùng `MlflowClient` đọc runs/params/metric histories/artifacts từ SQLite URI tuyệt đối. Bảng số liệu và verification.json lấy metrics trực tiếp từ MLflow, không parse loss/VRAM từ stdout hoặc tạo metrics giả. Metadata chỉ dùng lấy run ID và đối chiếu provenance. Verification reject missing/mismatch/FAILED/nonfinite/zero update; downloaded artifacts phải khớp SHA256 originals.

CLI export SQLite backup nhất quán, mlruns artifact store, model outputs, manifest, logs/results vào ZIP và kiểm tra checksums. Tải `/kaggle/working/smoke-review-*.zip` cùng `smoke-pip-check.log` trong Output trước reset. Artifact URIs gốc Kaggle được giữ trong database; restore cùng path hoặc relocate rõ ràng trên host khác. Nếu verification FAIL, CLI vẫn export khi store đầy đủ rồi exit lỗi. Train/reload/preflight failure dừng sớm và giữ logs, không tuyên bố PASS. Workaround torchao và pip conflicts không chứng minh clean environment; greedy không phải accuracy/numerical tolerance. Không chạy full training hoặc sửa production training config.

Nếu đã clone trước bản sửa preflight, chạy `git pull --ff-only origin main` từ repo root rồi gọi lại `CUDA_VISIBLE_DEVICES=0 python -m src.smoke_review`; không cần reset/install lại chỉ vì lỗi indentation. Preflight fail trước training nên lượt lỗi không tạo smoke run mới.
