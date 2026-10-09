from pathlib import Path


# --- THÔNG SỐ MODEL & DATASET ---
DATASET_ID = "TIGER-Lab/MathInstruct"
DATASET_REVISION = "b4fdc323a7be1379c9c7c0b67b1de72dfee2111a"
DATASET_SNAPSHOT_HASH = "6b438786f5ef69c39ac752b4d6eb7ccbfbfd69787ea61047f4a30702a475cc1a"
BASE_REVISION = "5a8abab4a5d6f164389b1079fb721cfab8d7126c"
TOKENIZER_REVISION = BASE_REVISION

BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"

# --- THÔNG SỐ HUẤN LUYỆN ---
MAX_SEQ_LENGTH = 1024
MAX_EVAL_SAMPLES = 100
MAX_TEST_SAMPLES = 500
SPLIT_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "split_manifest.json"
TRAIN_FRACTION = 0.03
SEED = 42

# --- THƯ MỤC OUTPUT ---
OUTPUT_ROOT = "models"

# --- CẤU HÌNH LORA ---
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Shared Task 3B protocol; pilot length, not a locked benchmark configuration.
SYSTEM_MESSAGE = (
    "You are a helpful math tutor.\n"
    "Solve the problem with clear reasoning and give a concise final answer."
)
CHAT_TEMPLATE_KWARGS = {"date_string": "09 Oct 2026"}
