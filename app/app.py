"""MathSpeech demo (Gradio). Run: python app/app.py   (CUDA_VISIBLE_DEVICES picks the GPU; default 0)"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gradio as gr
from src.env import OUTPUT_DIR
from src.inference.transcriber import MathTranscriber

DEMO = OUTPUT_DIR / "demo"
_t = {}


def get():
    if "t" not in _t:
        _t["t"] = MathTranscriber()
    return _t["t"]


def tex_block(s):
    s = (s or "").strip()
    return f"$${s}$$" if s else "*(empty)*"


def run(audio):
    if audio is None:
        return "", "", "", "", "", ""
    r = get()(audio)
    return r["generic"], r["generic_rules_text"] or "", r["text"], r["latex"], tex_block(r["latex"]), f"`{r['latex']}`"


def build():
    ex = json.loads((DEMO / "examples.json").read_text()) if (DEMO / "examples.json").exists() else []
    with gr.Blocks(title="MathSpeech") as demo:
        gr.Markdown("# MathSpeech\nSpeech recognition that writes **mathematical notation**, not just words (Qwen3-ASR + LoRA). "
                    "Upload or record a spoken equation (math, physics, statistics, ML, chemistry, CS).")
        with gr.Row():
            audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Spoken math")
            btn = gr.Button("Transcribe", variant="primary")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Normal ASR (zero-shot)")
                generic = gr.Textbox(label="raw transcript", interactive=False)
                rules = gr.Textbox(label="+ rule-based normaliser (as plain text)", interactive=False)
            with gr.Column():
                gr.Markdown("### MathSpeech (fine-tuned)")
                text = gr.Textbox(label="Plain text (Unicode, renders anywhere)", interactive=False, elem_classes=["big"])
                latex = gr.Textbox(label="LaTeX", interactive=False)
                gr.Markdown("**Typeset**")
                rendered = gr.Markdown(latex_delimiters=[{"left": "$$", "right": "$$", "display": True}, {"left": "$", "right": "$", "display": False}])
                raw = gr.Markdown()
        outs = [generic, rules, text, latex, rendered, raw]
        btn.click(run, audio, outs); audio.change(run, audio, outs)
        if ex:
            gr.Markdown("### Examples (held-out test sets; click to load)")
            gr.Examples(examples=[[str(DEMO / e["audio"])] for e in ex], inputs=[audio], example_labels=[f"[{e['domain']}] {e.get('text') or e['truth'][:60]}" for e in ex], examples_per_page=24)
    return demo


if __name__ == "__main__":
    build().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), allowed_paths=[str(DEMO)])
