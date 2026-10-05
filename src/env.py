"""Tiny env/config helper: loads .env (if present) and resolves paths relative to repo root."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env():
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.split("#")[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf"))
    if not os.path.isabs(os.environ["HF_HOME"]):
        os.environ["HF_HOME"] = str(ROOT / os.environ["HF_HOME"])
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # project rule: single GPU


def path(env_name, default):
    p = Path(os.environ.get(env_name, default))
    return p if p.is_absolute() else ROOT / p


load_env()
DATA_DIR = path("DATA_DIR", "data")
OUTPUT_DIR = path("OUTPUT_DIR", "artifacts")
MODEL_ID = os.environ.get("MODEL_ID", "openai/whisper-large-v3-turbo")
