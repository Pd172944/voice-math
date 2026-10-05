"""MathSpeech demo (Gradio). Run: python app/app.py   (set CUDA_VISIBLE_DEVICES to pick one GPU)"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gradio as gr
from src.env import OUTPUT_DIR, MODEL_ID
from src.inference.asr import ASR
from src.normalization.canonicalize import canonicalize

ADAPTER = os.environ.get("ADAPTER", str(OUTPUT_DIR / "model/mathspeech-canonical/best"))
DEMO = OUTPUT_DIR / "demo"
_asr = {}


def get(kind):
    if kind not in _asr:
        _asr[kind] = ASR(MODEL_ID, adapter=ADAPTER if kind == "math" else None)
    return _asr[kind]


def tex_block(s):
    s = (s or "").strip()
    return f"$${s}$$" if s else "*(empty)*"


def run(audio):
    if audio is None:
        return "", "", "", "", ""
    generic = get("generic").transcribe([audio])[0]
    rules = canonicalize(generic)["latex"]
    math = get("math").transcribe([audio])[0]
    return generic, tex_block(rules), math, tex_block(math), f"`{math}`"


def build():
    ex = json.loads((DEMO / "examples.json").read_text()) if (DEMO / "examples.json").exists() else []
    with gr.Blocks(title="MathSpeech") as demo:
        gr.Markdown("# MathSpeech\nSpeech recognition that writes **mathematical notation**, not just words. "
                    "Upload or record a spoken equation (math, physics, statistics, ML, chemistry, CS).")
        with gr.Row():
            audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Spoken math")
            btn = gr.Button("Transcribe", variant="primary")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Normal ASR (Whisper large-v3-turbo, zero-shot)")
                generic = gr.Textbox(label="raw transcript", interactive=False)
                gr.Markdown("**+ rule-based math normaliser**")
                rules = gr.Markdown()
            with gr.Column():
                gr.Markdown("### MathSpeech (LoRA fine-tuned)")
                math = gr.Textbox(label="LaTeX", interactive=False)
                gr.Markdown("**Rendered**")
                rendered = gr.Markdown(latex_delimiters=[{"left": "$$", "right": "$$", "display": True}, {"left": "$", "right": "$", "display": False}])
                raw = gr.Markdown()
        outs = [generic, rules, math, rendered, raw]
        btn.click(run, audio, outs); audio.change(run, audio, outs)
        if ex:
            gr.Markdown("### Examples (held-out test set; click to load)")
            gr.Examples(examples=[[str(DEMO / e["audio"])] for e in ex], inputs=[audio], example_labels=[f"[{e['domain']}] {e['truth'][:60]}" for e in ex], examples_per_page=20)
    return demo


if __name__ == "__main__":
    build().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), allowed_paths=[str(DEMO)])
