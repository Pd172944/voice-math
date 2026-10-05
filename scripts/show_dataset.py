#!/usr/bin/env python
"""Dump random dataset examples (audio path, spoken transcript, LaTeX, source, domain) to the terminal and to an HTML page with KaTeX rendering."""
import argparse, html, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.env import DATA_DIR, OUTPUT_DIR


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="train"); ap.add_argument("-n", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domain", default=None); ap.add_argument("--html", default=str(OUTPUT_DIR / "datasets/sample.html"))
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(DATA_DIR / f"manifests/mathspeech_{a.split}.jsonl")]
    if a.domain: rows = [r for r in rows if r["domain"] == a.domain]
    rows = random.Random(a.seed).sample(rows, min(a.n, len(rows)))
    cards = []
    for r in rows:
        print(f"[{r['domain']}] d={r['difficulty']} voice={r['speaker_id']} src={r['source']}\n  audio : {r['audio']}\n  spoken: {r['spoken_transcript']}\n  latex : {r['target_text']}\n")
        cards.append(f"<div class=c><audio controls src='file://{html.escape(r['audio'])}'></audio><p><b>{html.escape(r['domain'])}</b> · {html.escape(r['source'])} · {html.escape(r['license'])}</p>"
                     f"<p>{html.escape(r['spoken_transcript'])}</p><code>{html.escape(r['target_text'])}</code><div class=m>$${html.escape(r['target_text'])}$$</div></div>")
    Path(a.html).parent.mkdir(parents=True, exist_ok=True)
    Path(a.html).write_text("<meta charset=utf-8><link rel=stylesheet href='https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css'><script defer src='https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js'></script>"
                            "<script defer src='https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js' onload=\"renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true}]})\"></script>"
                            "<style>body{font-family:sans-serif;max-width:800px;margin:2em auto}.c{border:1px solid #ccc;border-radius:8px;padding:12px;margin:10px 0}</style>" + "".join(cards))
    print(f"HTML -> {a.html}")


if __name__ == "__main__":
    main()
