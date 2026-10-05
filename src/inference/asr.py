"""ASR wrappers. `make_asr(model_id, adapter)` returns a Whisper or Qwen3-ASR backend (zero-shot, or with a merged LoRA adapter)."""
import numpy as np
import soundfile as sf
import torch
from src.env import MODEL_ID, ROOT

SR = 16000


def load_audio(x, max_seconds=30):
    if isinstance(x, (str, bytes)) or hasattr(x, "__fspath__"):
        p = str(x)
        if not p.startswith("/") and (ROOT / p).exists():
            p = str(ROOT / p)
        a, sr = sf.read(p, dtype="float32")
    else:
        a, sr = x
        a = np.asarray(a, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    if sr != SR:
        import librosa
        a = librosa.resample(a, orig_sr=sr, target_sr=SR)
    return a[: SR * max_seconds]


def is_qwen(model_id):
    return "qwen3-asr" in model_id.lower()


class WhisperASR:
    def __init__(self, model_id, adapter=None, device="cuda", dtype=torch.bfloat16):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.model_id = model_id
        self.processor = WhisperProcessor.from_pretrained(model_id)
        m = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
        if adapter:
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
        self.model = m.to(device).eval()
        self.device, self.dtype = device, dtype
        # Whisper's default generation config suppresses symbol tokens (\ { } ^ _ ...), which makes LaTeX impossible to emit.
        # Fine-tuned (adapter) models must be decoded without that suppression; the zero-shot baseline keeps the default.
        self.gen_kwargs = dict(suppress_tokens=[], begin_suppress_tokens=[]) if adapter else {}

    @torch.no_grad()
    def transcribe(self, items, batch_size=24, max_new_tokens=200, num_beams=1):
        out = []
        for i in range(0, len(items), batch_size):
            arrs = [load_audio(x) for x in items[i:i + batch_size]]
            feats = self.processor.feature_extractor(arrs, sampling_rate=SR, return_tensors="pt").input_features.to(self.device, self.dtype)
            ids = self.model.generate(input_features=feats, language="en", task="transcribe", max_new_tokens=max_new_tokens, num_beams=num_beams, **self.gen_kwargs)
            out += [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]
        return out


class QwenASR:
    """Qwen3-ASR (Qwen3-Omni audio encoder + Qwen3 LLM decoder), Transformers-native."""
    def __init__(self, model_id, adapter=None, device="cuda", dtype=torch.bfloat16):
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        m = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=dtype)
        if adapter:
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
        self.model = m.to(device).eval()
        self.device, self.dtype = device, dtype

    @torch.no_grad()
    def transcribe(self, items, batch_size=32, max_new_tokens=256, num_beams=1):
        out = []
        for i in range(0, len(items), batch_size):
            arrs = [load_audio(x) for x in items[i:i + batch_size]]
            inp = self.processor.apply_transcription_request(arrs, language="English").to(self.device, self.dtype)
            ids = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, num_beams=num_beams)
            gen = ids[:, inp["input_ids"].shape[1]:]
            out += [t.strip() for t in self.processor.decode(gen, return_format="transcription_only")]
        return out


def make_asr(model_id=None, adapter=None, **kw):
    model_id = model_id or MODEL_ID
    return (QwenASR if is_qwen(model_id) else WhisperASR)(model_id, adapter=adapter, **kw)


ASR = make_asr   # backwards-compatible name
