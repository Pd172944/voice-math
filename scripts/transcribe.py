#!/usr/bin/env python
"""Transcribe audio files: python scripts/transcribe.py a.wav b.flac [--json]  ->  generic ASR, MathSpeech LaTeX, and readable Unicode text."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference.transcriber import MathTranscriber


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("audio", nargs="+"); ap.add_argument("--json", action="store_true"); ap.add_argument("--adapter", default=None); ap.add_argument("--no_generic", action="store_true")
    a = ap.parse_args()
    res = MathTranscriber(adapter=a.adapter, with_generic=not a.no_generic)(a.audio)
    for f, r in zip(a.audio, res):
        if a.json: print(json.dumps(dict(file=f, **r), ensure_ascii=False)); continue
        print(f"{f}\n  generic    : {r['generic']}\n  generic+rules: {r['generic_rules_text']}\n  MathSpeech : {r['text']}\n  LaTeX      : {r['latex']}\n")


if __name__ == "__main__":
    main()
