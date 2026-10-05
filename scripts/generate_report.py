#!/usr/bin/env python
"""Step 12: artifacts/evaluation/report.html (audio + generic ASR + rules + MathSpeech + truth + KaTeX render) and artifacts/report.md."""
import html, json, random, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.env import DATA_DIR, OUTPUT_DIR
from src.normalization.canonicalize import canonicalize
from src.evaluation.metrics import norm_latex

EV = OUTPUT_DIR / "evaluation"
KATEX = ("<link rel=stylesheet href='https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css'><script defer src='https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js'></script>"
         "<script defer src='https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js' onload=\"renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true}],throwOnError:false})\"></script>")
CSS = """<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;color:#222}table{border-collapse:collapse;margin:1em 0}td,th{border:1px solid #ccc;padding:4px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}.ex{border:1px solid #ddd;border-radius:10px;padding:12px 16px;margin:14px 0}.row{display:grid;grid-template-columns:150px 1fr;gap:6px;margin:4px 0}.lab{color:#666;font-size:.85em}
.ok{background:#e8f7e8}.bad{background:#fdeaea}code{font-size:.9em;word-break:break-all}.tag{font-size:.75em;background:#eee;border-radius:4px;padding:1px 6px}.r{overflow-x:auto}</style>"""


def mtable(m):
    rows = ["exact_match", "latex_token_edit", "symbol_acc", "greek_acc", "operator_acc", "script_acc", "bracket_acc", "wer_vs_spoken"]
    names = {"zeroshot": "A. Generic Whisper (raw)", "rules": "C. Generic + rule normaliser", "canonical": "B. MathSpeech (canonical LoRA)", "spoken_ft_rules": "D. Spoken-LoRA + rule normaliser"}
    sy = [k for k in names if k in m]
    t = "<table><tr><th>metric</th>" + "".join(f"<th>{names[s]}</th>" for s in sy) + "</tr>"
    for r in rows:
        t += f"<tr><td>{r}</td>" + "".join(f"<td>{m[s].get(r):.3f}</td>" if m[s].get(r) is not None else "<td>–</td>" for s in sy) + "</tr>"
    return t + f"<tr><td>n</td>" + "".join(f"<td>{m[s]['n']}</td>" for s in sy) + "</tr></table>"


def main(n_per_split=60):
    (EV / "audio").mkdir(parents=True, exist_ok=True)
    body, md = [], ["# MathSpeech – experiment results\n"]
    rng = random.Random(0)
    for split in ("test", "test_hard"):
        mf_path = DATA_DIR / f"manifests/mathspeech_{split}.jsonl"; pf = EV / f"predictions_{split}.jsonl"; mp = EV / f"metrics_{split}.json"
        if not (mf_path.exists() and pf.exists() and mp.exists()): continue
        mf = {json.loads(l)["id"]: json.loads(l) for l in open(mf_path)}; preds = [json.loads(l) for l in open(pf)]
        m = json.loads(mp.read_text())
        title = {"test": "Test (unseen expressions, partly unseen voices)", "test_hard": "Hard test (unseen grammar templates, unseen voices)"}[split]
        body.append(f"<h2>{title}</h2>" + mtable(m))
        md.append(f"\n## {title}\n\n| metric | " + " | ".join(k for k in ("zeroshot", "rules", "canonical", "spoken_ft_rules") if k in m) + " |\n|---|" + "---|" * len([k for k in ("zeroshot", "rules", "canonical", "spoken_ft_rules") if k in m]))
        for r in ("exact_match", "latex_token_edit", "symbol_acc", "greek_acc", "operator_acc", "script_acc", "bracket_acc", "wer_vs_spoken"):
            md.append(f"| {r} | " + " | ".join(("%.3f" % m[k][r]) if m[k].get(r) is not None else "–" for k in ("zeroshot", "rules", "canonical", "spoken_ft_rules") if k in m) + " |")
        pd = m.get("per_domain_exact", {})
        if pd:
            body.append("<h3>Exact match by domain</h3><table><tr><th>domain</th><th>n</th><th>generic</th><th>generic+rules</th><th>MathSpeech</th></tr>" +
                        "".join(f"<tr><td>{d}</td><td>{v['n']}</td><td>{v.get('zeroshot',0):.2f}</td><td>{v.get('rules',0):.2f}</td><td>{v.get('canonical',0):.2f}</td></tr>" for d, v in sorted(pd.items())) + "</table>")
            md.append("\n| domain | n | generic | generic+rules | MathSpeech |\n|---|---|---|---|---|")
            md += [f"| {d} | {v['n']} | {v.get('zeroshot',0):.2f} | {v.get('rules',0):.2f} | {v.get('canonical',0):.2f} |" for d, v in sorted(pd.items())]
        sel = [p for p in preds if p["id"] in mf and "canonical" in p]
        rng.shuffle(sel); sel = sel[:n_per_split] if split == "test" else sel[:n_per_split]
        body.append(f"<h3>Examples ({len(sel)} random, not cherry-picked)</h3>")
        for p in sel:
            r = mf[p["id"]]; dst = EV / "audio" / (p["id"] + ".flac")
            if not dst.exists(): shutil.copy(r["audio"], dst)
            rules = canonicalize(p["zeroshot"])["latex"]
            okb = norm_latex(p["canonical"]) == norm_latex(r["target_text"])
            body.append(f"<div class='ex'><span class=tag>{r['domain']}</span> <span class=tag>difficulty {r['difficulty']}</span> <span class=tag>voice {r['speaker_id']}</span><br><audio controls src='audio/{p['id']}.flac'></audio>"
                        f"<div class=row><span class=lab>Generic ASR</span><span>{html.escape(p['zeroshot'])}</span></div>"
                        f"<div class=row><span class=lab>Generic + rules</span><code>{html.escape(rules)}</code></div>"
                        f"<div class='row {'ok' if okb else 'bad'}'><span class=lab>MathSpeech</span><code>{html.escape(p['canonical'])}</code></div>"
                        f"<div class=row><span class=lab>Ground truth</span><code>{html.escape(r['target_text'])}</code></div>"
                        f"<div class='row r'><span class=lab>Rendered (MathSpeech)</span><span>$${html.escape(p['canonical'])}$$</span></div>"
                        f"<div class='row r'><span class=lab>Rendered (truth)</span><span>$${html.escape(r['target_text'])}$$</span></div></div>")
    ts = OUTPUT_DIR / "model/mathspeech-canonical/training_summary.json"
    if ts.exists():
        t = json.loads(ts.read_text())
        md.append(f"\n## Training\n\n- model: `{t['model']}` + LoRA, target_mode=`{t['target_mode']}`\n- steps: {t['steps']}, examples: {t['train_examples']}, best val loss: {t['best_val_loss']:.4f}\n- GPU: {t['gpu']}, wall time: {t['train_seconds']/60:.1f} min")
    (EV / "report.html").write_text(f"<!doctype html><meta charset=utf-8><title>MathSpeech evaluation</title>{KATEX}{CSS}<h1>MathSpeech – evaluation report</h1>" + "".join(body))
    (OUTPUT_DIR / "report.md").write_text("\n".join(md) + "\n\nSee `artifacts/evaluation/report.html` for audio + rendered examples.\n")
    print(f"report -> {EV / 'report.html'}")


if __name__ == "__main__":
    main()
