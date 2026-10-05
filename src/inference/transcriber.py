"""High-level API: audio -> {generic transcript, rule-normalised LaTeX, MathSpeech LaTeX, readable Unicode text}."""
import os
from src.env import OUTPUT_DIR, MODEL_ID
from src.inference.asr import make_asr
from src.normalization.canonicalize import canonicalize
from src.rendering.latex_to_text import latex_to_text

DEFAULT_ADAPTER = os.environ.get("ADAPTER", str(OUTPUT_DIR / "model/qwen3asr-canonical/best"))


class MathTranscriber:
    def __init__(self, model_id=None, adapter=None, with_generic=True):
        self.model_id = model_id or MODEL_ID
        self.math = make_asr(self.model_id, adapter=adapter or DEFAULT_ADAPTER)
        self.generic = make_asr(self.model_id) if with_generic else None

    def __call__(self, audio_items, batch_size=16):
        items = audio_items if isinstance(audio_items, (list, tuple)) else [audio_items]
        latex = self.math.transcribe(items, batch_size=batch_size)
        gen = self.generic.transcribe(items, batch_size=batch_size) if self.generic else [None] * len(items)
        out = []
        for g, l in zip(gen, latex):
            rules = canonicalize(g)["latex"] if g else None
            out.append(dict(generic=g, generic_rules_latex=rules, generic_rules_text=latex_to_text(rules) if rules else None, latex=l, text=latex_to_text(l)))
        return out if isinstance(audio_items, (list, tuple)) else out[0]
