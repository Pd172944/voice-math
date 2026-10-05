#!/usr/bin/env python
"""artifacts/evaluation/report.html (audio + generic ASR + rules + MathSpeech LaTeX/plain-text + truth + KaTeX render + error analysis) and artifacts/report.md."""
import html, json, random, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.env import DATA_DIR, OUTPUT_DIR, ROOT
from src.normalization.canonicalize import canonicalize
from src.evaluation.metrics import norm_latex
from src.rendering.latex_to_text import latex_to_text

EV = OUTPUT_DIR / "evaluation"
KATEX = ("<link rel=stylesheet href='https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css'><script defer src='https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js'></script>"
         "<script defer src='https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js' onload=\"renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true}],throwOnError:false})\"></script>")
CSS = """<style>body{font-family:system-ui,sans-serif;max-width:1150px;margin:2em auto;padding:0 1em;color:#222}table{border-collapse:collapse;margin:1em 0;font-size:.92em}td,th{border:1px solid #ccc;padding:4px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}.ex{border:1px solid #ddd;border-radius:10px;padding:12px 16px;margin:14px 0}.row{display:grid;grid-template-columns:150px 1fr;gap:6px;margin:4px 0}.lab{color:#666;font-size:.85em}
.ok{background:#e8f7e8}.bad{background:#fdeaea}code{font-size:.9em;word-break:break-all}.tag{font-size:.75em;background:#eee;border-radius:4px;padding:1px 6px}.r{overflow-x:auto}.txt{font-size:1.15em}</style>"""
NAMES = {"zeroshot": "A. Qwen3-ASR zero-shot", "rules": "C. A + rule normaliser", "canonical": "B. MathSpeech (Qwen3-ASR+LoRA)", "whisper_zeroshot": "Whisper v3-turbo zero-shot",
         "whisper_rules": "Whisper + rules", "whisper_canonical": "Whisper + LoRA (v1)"}
ROWS = ["exact_match", "latex_token_edit", "symbol_acc", "greek_acc", "operator_acc", "script_acc", "bracket_acc", "wer_vs_spoken"]
TITLES = {"test": "Test — unseen expressions (template families seen in training)", "test_comp": "Test-comp — unseen random compositions of known constructs",
          "test_hard": "Test-hard — 9 grammar templates never seen in training, unseen voices", "test_ood": "Test-OOD — 64 hand-written expressions, natural phrasing, unseen voices"}


def fmt(v): return "–" if v is None else f"{v:.3f}"


def main(n_per_split=14):
    (EV / "audio").mkdir(parents=True, exist_ok=True)
    body, md = [], ["# MathSpeech – experiment results\n"]
    rng = random.Random(0)
    for split, title in TITLES.items():
        mp, pf, mt = DATA_DIR / f"manifests/mathspeech_{split}.jsonl", EV / f"predictions_{split}.jsonl", EV / f"metrics_{split}.json"
        if not (mp.exists() and pf.exists() and mt.exists()): continue
        mf = {json.loads(l)["id"]: json.loads(l) for l in open(mp)}; preds = [json.loads(l) for l in open(pf)]; m = json.loads(mt.read_text())
        sy = [k for k in NAMES if k in m]
        t = "<table><tr><th>metric</th>" + "".join(f"<th>{NAMES[s]}</th>" for s in sy) + "</tr>"
        for r in ROWS: t += f"<tr><td>{r}</td>" + "".join(f"<td>{fmt(m[s].get(r))}</td>" for s in sy) + "</tr>"
        t += "<tr><td>n</td>" + "".join(f"<td>{m[s]['n']}</td>" for s in sy) + "</tr></table>"
        body.append(f"<h2>{title}</h2>{t}")
        md.append(f"\n## {title}\n\n| metric | " + " | ".join(NAMES[s] for s in sy) + " |\n|---|" + "---|" * len(sy))
        md += [f"| {r} | " + " | ".join(fmt(m[s].get(r)) for s in sy) + " |" for r in ROWS]
        pd = m.get("per_domain_exact", {})
        if pd:
            body.append("<h3>Exact match by domain</h3><table><tr><th>domain</th><th>n</th><th>zero-shot</th><th>+rules</th><th>MathSpeech</th></tr>" +
                        "".join(f"<tr><td>{d}</td><td>{v['n']}</td><td>{v.get('zeroshot',0):.2f}</td><td>{v.get('rules',0):.2f}</td><td>{v.get('canonical',0):.2f}</td></tr>" for d, v in sorted(pd.items())) + "</table>")
        sel = [p for p in preds if p["id"] in mf and "canonical" in p and "zeroshot" in p]
        fails = [p for p in sel if norm_latex(p["canonical"]) != norm_latex(mf[p["id"]]["target_text"])]
        oks = [p for p in sel if p not in fails]; rng.shuffle(oks); rng.shuffle(fails)
        shown = fails[: n_per_split // 2 + 2] + oks[: n_per_split]
        body.append(f"<h3>Examples — {len(fails[:n_per_split//2+2])} failures (error analysis, all failures of this kind are listed in predictions_{split}.jsonl) + {len(oks[:n_per_split])} random successes</h3>")
        for p in shown:
            r = mf[p["id"]]; dst = EV / "audio" / (p["id"] + ".flac")
            if not dst.exists(): shutil.copy(ROOT / r["audio"], dst)
            rules = canonicalize(p["zeroshot"])["latex"]; okb = norm_latex(p["canonical"]) == norm_latex(r["target_text"])
            body.append(f"<div class='ex'><span class=tag>{r['domain']}</span> <span class=tag>difficulty {r['difficulty']}</span> <span class=tag>voice {r['speaker_id']}</span><br><audio controls src='audio/{p['id']}.flac'></audio>"
                        f"<div class=row><span class=lab>Generic ASR</span><span>{html.escape(p['zeroshot'])}</span></div>"
                        f"<div class=row><span class=lab>Generic + rules</span><span class=txt>{html.escape(latex_to_text(rules))}</span></div>"
                        f"<div class='row {'ok' if okb else 'bad'}'><span class=lab>MathSpeech (text)</span><span class=txt>{html.escape(latex_to_text(p['canonical']))}</span></div>"
                        f"<div class='row {'ok' if okb else 'bad'}'><span class=lab>MathSpeech (LaTeX)</span><code>{html.escape(p['canonical'])}</code></div>"
                        f"<div class=row><span class=lab>Ground truth</span><code>{html.escape(r['target_text'])}</code></div>"
                        f"<div class='row r'><span class=lab>Typeset</span><span>$${html.escape(p['canonical'])}$$</span></div></div>")
    for nm, ttl in (("external_aaai_mathspeech.json", "Real human speech (evaluation only; AAAI2025/MathSpeech, n=1101)"),):
        f = EV / nm
        if f.exists():
            m = json.loads(f.read_text()); sy = [k for k in ("zeroshot", "rules", "mathspeech") if k in m]
            body.append(f"<h2>{ttl}</h2><table><tr><th>metric</th>" + "".join(f"<th>{s}</th>" for s in sy) + "</tr>" + "".join(f"<tr><td>{r}</td>" + "".join(f"<td>{fmt(m[s].get(r))}</td>" for s in sy) + "</tr>" for r in ROWS) + "</table>")
            md.append(f"\n## {ttl}\n\n| metric | " + " | ".join(sy) + " |\n|---|" + "---|" * len(sy)); md += [f"| {r} | " + " | ".join(fmt(m[s].get(r)) for s in sy) + " |" for r in ROWS]
    for tag, d in (("Qwen3-ASR LoRA (final)", "qwen3asr-canonical"),):
        ts = OUTPUT_DIR / f"model/{d}/training_summary.json"
        if ts.exists():
            t = json.loads(ts.read_text())
            md.append(f"\n## Training — {tag}\n\n- model: `{t['model']}` + LoRA, target_mode=`{t['target_mode']}`\n- steps: {t['steps']}, examples: {t['train_examples']}, best val loss: {t['best_val_loss']:.4f}\n- GPU: {t['gpu']}, wall time: {t['train_seconds']/60:.1f} min")
    (EV / "report.html").write_text(f"<!doctype html><meta charset=utf-8><title>MathSpeech evaluation</title>{KATEX}{CSS}<h1>MathSpeech – evaluation report</h1>" + "".join(body))
    (OUTPUT_DIR / "report.md").write_text("\n".join(md) + "\n\nSee `artifacts/evaluation/report.html` for audio + typeset examples.\n")
    print(f"report -> {EV / 'report.html'}")


if __name__ == "__main__":
    main()
