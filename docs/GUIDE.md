# Math Instruct Llama — Guide

Cập nhật **2026-10-11**. Kết quả/approval/lịch sử nằm trong [PROGRESS.md](PROGRESS.md); [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) chỉ giữ lịch sử.

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

Chỉ chấp nhận **một visible CUDA GPU**. T4 dùng FP16; chỉ chọn BF16 khi `is_bf16_supported(including_emulation=False)` xác nhận hỗ trợ native. Default True có thể tính cả emulation trên T4. QLoRA NF4/double quant có cùng compute dtype và phải thật sự loaded-in-4bit. TRL1.3.0 tự cast trainable quantized-model adapters sang BF16; ở FP16 mode, repo chuyển riêng trainable adapters về FP32 ngay sau Trainer initialization để FP16 GradScaler hoạt động. Frozen NF4 base không bị cast; BF16 native mode và LoRA không đổi. Snapshot kiểm tra adapter update lấy sau mọi dtype changes, để không tính rounding do cast thành optimizer update. Artifact metadata ghi `trainable_dtypes` riêng với compute precision. `use_cache=False` khi train; gradient checkpointing explicit True, `use_reentrant=False` đồng nhất ở kbit helper và Trainer. CPU contracts và Kaggle T4 runtime smoke đã PASS. Peak VRAM/resource metrics smoke đã VERIFIED ở section kết quả mới; full training feasibility chưa VERIFIED.

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

## Kết quả smoke MLflow VERIFIED — 2026-10-10

Hai runs trên source `f2a11f5`, Tesla T4/FP16, train đúng **2 optimizer steps trên 64 samples**, final validation100, save/reload greedy và MLflow round-trip PASS. ZIP local `smoke-review-20261010T014250450690Z.zip` có SHA256 `284bfe45c392c806c7ddf3030ae75572139b4142adeef4d7ee89799f96a63a0b`; 44 file checksums PASS. Metrics dưới đây được đối chiếu trực tiếp SQLite với verification.json; không lấy từ demo output.

| Metric | LoRA | QLoRA |
|---|---:|---:|
| Run ID | `db8ecec18a5a46a084751a9067d40ead` | `007fe7181b6446e3a2e2acc3d0e90e9e` |
| final_eval_loss | 1.4786574841 | 1.4764468670 |
| adapter_update_norm | 0.0405102296 | 0.0401510722 |
| train + final eval (s) | 33.4980 | 44.0559 |
| peak allocated (GiB) | 3.67 | 3.35 |
| peak reserved (GiB) | 9.11 | 8.28 |
| trainable parameters | 2,818,048 | 2,818,048 |

Cả hai FINISHED, cùng selected/effective/excluded/validation7359/7310/49/100 và effective IDs hash `92dc3e8c1d8be4b86d502fb4e355b5ef913896563a2eabb976579c82b11a0fc2`. Artifact save/download hashes và reload PASS; numerical tolerance, answer accuracy, held-out test và **full training chưa VERIFIED**. `pip check` vẫn FAIL vì các conflicts ghi trong `smoke-pip-check.log`; không tuyên bố môi trường sạch. Bản local giải nén ở `mlruns/kaggle-smoke-review-20261010/` đã relocate database artifact URIs; ZIP gốc giữ nguyên.

## Full training trên Kaggle — hướng dẫn chuẩn bị, chưa chạy

### 1. Chuẩn bị

Chỉ chạy sau approval riêng cho full training. Bật Internet và một visible T4; dùng repo/stack đã chạy smoke. Session mới dùng clone/install ở notebook hiện tại, **không chạy cell `src.smoke_review`** nếu mục tiêu là full training. Session đang có repo chỉ pull khi working tree sạch. Ghi HEAD một lần trước hai methods, không pull giữa hai runs:

```python
%cd /kaggle/working/math-instruct-llama
!git log -1 --oneline
!nvidia-smi
!python -m pip check
```

Thực hiện inventory/import/native BF16 ở section môi trường trước khi train; giữ torch/CUDA đã provision, không đổi pins/config theo warnings. Chạy tuần tự LoRA rồi QLoRA trong cùng session/repo root để cùng SQLite store/experiment, mỗi lệnh là subprocess riêng giải phóng GPU khi kết thúc. Session mới không tự chứa smoke database cũ; nếu muốn cùng UI với smoke, cần restore store có chủ đích trước training.

### 2. Chạy từng method và lưu console log

Các lệnh sau là **full training**, không phải smoke. Chạy một cell mỗi method, kiểm tra exit code trước khi tiếp tục:

```python
%%bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python -u -m src.train --method lora 2>&1 | tee /kaggle/working/lora-full.log
```

```python
%%bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python -u -m src.train --method qlora 2>&1 | tee /kaggle/working/qlora-full.log
```

Giữ nguyên config: length1024, effective7310, validation100, epoch1, microbatch1, accumulation32, LR2e-4/cosine/warmup5, seed42, r4/alpha8/dropout0.05, paged_adamw_8bit, gradient checkpointing; T4 FP16. Khoảng **229 optimizer steps** nếu dữ liệu/counts hiện tại giữ nguyên. Eval/save mỗi40 steps, log mỗi10 steps; final eval luôn chạy. Không sử dụng held-out test hoặc thay config theo kết quả này.

**Tài nguyên:** smoke chỉ là prefix64, đỉnh memory3.67/3.35GiB allocated và9.11/8.28GiB reserved; full subset có sequences dài hơn nên không coi đây là giới hạn full VRAM. Chưa có cơ sở đủ chắc để hứa thời gian full run: smoke gồm startup, mới2 steps, full có scheduled validation/save. Theo dõi ETA sau vài chục steps và nvidia-smi. Chừa storage cho base/cache, manifest (~116MB), checkpoints và hai bản final (models + MLflow upload); chưa có đo tổng disk full run. Nếu OOM/session hết hạn, giữ logs/checkpoint và review trước khi thay config.

### 3. Checkpoint và MLflow

Ghi lại path `Saved adapter/tokenizer/protocol` cuối mỗi run. Output: `models/<method>_training_<id>/final`; checkpoints trong `checkpoints/checkpoint-N`. `save_total_limit=1` giữ checkpoint gần nhất, checkpoint cũ bị xoá. Final là weights **cuối training**, không tự chọn best validation checkpoint (`load_best_model_at_end` chưa bật). CLI hiện **không có resume flag**; không giả định chạy lại lệnh là resume, nó tạo run mới.

MLflow dùng experiment `llama-3-math-instruct`, run names `lora_training`/`qlora_training`. Database `mlruns.db`, artifact store `mlruns/`; params/step metrics/final validation/time/memory và final artifact được log. Checkpoints nằm trong models, **không tự upload lên MLflow**. `adapter_update_norm` chỉ có ở smoke; không yêu cầu metric này ở full. `src.smoke_review` chỉ kiểm tra smoke64/2 steps, không dùng để kiểm tra full runs.

Sau mỗi run, mở metadata final để lấy run ID, kiểm tra MLflow status FINISHED và final_eval_loss finite, final artifacts đủ adapter/tokenizer/provenance; reload CLI inference với đúng printed final path trước reset. Các metric quality trong UI là validation loss/token accuracy, chưa phải answer accuracy.

### 4. Save Version và tải outputs

Dùng `full-training.ipynb`: clone/install → full LoRA → full QLoRA. Chọn Save Version → Save & Run All trên Kaggle; notebook không có cell ZIP, export hoặc link download sau training. Khi version hoàn tất, tải outputs trực tiếp: `models/`, `mlruns/`, `mlruns.db` và WAL/SHM nếu có, `data/`. Giữ cấu trúc thư mục và database/artifacts cùng nhau để mở MLflow local; relocate artifact URIs Kaggle. Database riêng không chứa adapters/checkpoints. Kiểm tra cả hai runs thành công và files đã tải trước khi bỏ outputs.

Mỗi train tự validation định kỳ/final, không evaluator answer accuracy. Không đổi config/reset/pull giữa methods. Source runtime đã có trên GitHub; notebook full local chưa commit/push. Lượt hướng dẫn này không chạy full training.

### Local outputs sau cleanup

Theo yêu cầu user, smoke ZIP/log/evidence notebook và generated models/mlruns local đã xoá để nhận full outputs. Bảng VERIFIED giữ kết quả lịch sử; đường dẫn smoke archive/local UI nêu trên không còn tồn tại. Giữ data split và runnable notebooks/code. Khi tải full về, kiểm tra runs/metrics/artifacts trước khi đánh dấu full VERIFIED, rồi relocate database và mở UI mới.

## Full training Kaggle — verified outputs, 2026-10-11

User tải results.zip (SHA256 `8f7029c18941f8dc0959fc105aff6b757545f7d310afa837a51883b519e7d500`). ZIP CRC PASS (118 entries); extract chỉ64 output files models/mlruns/database/data vào `kaggle-full-results/math-instruct-llama/`, không ghi đè source. ZIP gốc giữ nguyên. SQLite integrity PASS; hai runs FINISHED, checkpoint-229/global_step229/epoch1; training_selection full7310, validation100, same dataset/manifest/tokenizer/effective hash. Final files SHA256 khớp MLflow artifacts, provenance khớp; local artifact URI relocation + actual MlflowClient download PASS. Validation có steps40/80/120/160/200/229, step229 ghi hai lần do final evaluation của Trainer và explicit evaluate; không suy ra đây là hai independent quality tests.

| Metric | LoRA | QLoRA |
|---|---:|---:|
| Run ID | bd847484833c4f2d9bac735de6034511 | 1f5aaf010bfc4050a8f281c71ba31b74 |
| train_loss | 0.7016379844 | 0.7252621359 |
| final_eval_loss | 0.7885810733 | 0.8134101033 |
| train_and_eval_seconds | 2389.7323 | 3234.1521 |
| peak_allocated_bytes | 4167497728 | 3841574912 |
| peak_reserved_bytes | 9491709952 | 6473908224 |

VERIFIED trong phạm vi full train/validation/save/MLflow archive và checkpoint state. Không actual GPU reload/generation adapter full, numerical tolerance, answer accuracy/base comparison/held-out test proof. Không chạy lại training hoặc evaluator local. Current environment/full provenance torch2.11.0+cu128/CUDA12.8/TeslaT4. Config full length1024/epoch1/accum32/save-eval40 giữ đúng metadata; checkpoint cuối229 thực tế được giữ. Local UI phải dùng DB mới trong kaggle-full-results, không smoke DB đã xoá. Next: user xem MLflow comparison; kiểm chứng reload full trên GPU trước dùng adapter, rồi chốt scope evaluator/base-vs-adapters nếu user yêu cầu. Chưa commit/push.

## Inference5 câu với full adapters (2026-10-11)

`inference.ipynb` đơn giản clone/install → liệt kê Input paths → gọi `python -m src.inference` cho LoRA và QLoRA lần lượt. `--demo-suite` thêm5 câu unit price/linear equation/fractions/quadratic/probability, expected finals $1.50/9/60 liters/8×12cm/10⁄21; `--greedy --max-new-tokens 512` dùng chung. Mỗi process load1 model/adapter, in question/generated answer/reference để review thủ công. Default CLI không --demo-suite vẫn1 câu/150 tokens như smoke cũ; không đổi train/data/config. Không chấm500 test, không claim accuracy.

Đã chuẩn bị local `models/math-instruct-adapters.zip` chỉ gồm hai folders `<method>_training_<id>/final/` từ verified full outputs, không base weights/checkpoints/MLflow DB. Upload ZIP này vào private Kaggle Dataset, đặt tên `math-instruct-adapters`, giữ folder hierarchy. Sau processing, kiểm tra dataset có hai final directories. Upload notebook, bật Internet/GPU, Add Input dataset. Chạy find cell rồi copy đúng path final nếu Kaggle mount/layout khác với paths ví dụ. Adapter path phải là cả final directory (weights/config/tokenizer/effective_data), không chỉ safetensors. Base weights tự download pinned revision. Không upload/run full-training notebook lần nữa để infer.

Có thể reuse outputs notebook training làm Input thay upload ZIP; tìm actual final paths và thay hai lệnh tương tự. Lưu outputs/Logs của5 câu mỗi method; GPU inference full vẫn pending đến chạy thực tế.
