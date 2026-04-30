import torch

# --- THÔNG SỐ MODEL & DATASET ---
DATASET_ID = "TIGER-Lab/MathInstruct"
BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"

# --- THÔNG SỐ HUẤN LUYỆN ---
MAX_SEQ_LENGTH = 128
MAX_EVAL_SAMPLES = 100
TRAIN_FRACTION = 0.03
SEED = 42

# --- THƯ MỤC OUTPUT ---
OUTPUT_ROOT = "models"

# --- CẤU HÌNH LORA ---
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", 
    "gate_proj", "up_proj", "down_proj"
]

# --- PROMPT TEMPLATE ---
PROMPT_TEMPLATE = (
    "You are a helpful math tutor.\n"
    "Solve the problem with clear reasoning and give a concise final answer.\n\n"
    "### Question:\n{question}\n\n"
    "### Answer:\n{answer}"
)