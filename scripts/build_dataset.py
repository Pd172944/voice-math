#!/usr/bin/env python
"""Step 5: merge synthetic (+ real, if present) into manifests and a HF DatasetDict (mathspeech_{train,validation,test,test_hard,test_comp,test_ood}).
Audio is stored as a *relative path string* (portable, no duplicated bytes); decode with `ds.cast_column("audio", datasets.Audio())`."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.env import DATA_DIR, ROOT
from datasets import Dataset, DatasetDict

SYN_LICENSE = "synthetic: grammar-generated text (MIT) + Kokoro-82M TTS (Apache-2.0)"
SPLITS = {"train": "mathspeech_train", "validation": "mathspeech_validation", "test": "mathspeech_test", "test_hard": "mathspeech_test_hard",
          "test_comp": "mathspeech_test_comp", "test_ood": "mathspeech_test_ood"}


def rel(p):
    try: return str(Path(p).resolve().relative_to(ROOT))
    except ValueError: return str(p)


def load_synth():
    man = {}
    for l in open(DATA_DIR / "manifests/tts_manifest.jsonl"):
        d = json.loads(l); man[d["id"]] = d
    rows = []
    for f in sorted((DATA_DIR / "synthetic").glob("spoken*.jsonl")):
        if ".tts." in f.name: continue
        for l in open(f):
            r = json.loads(l)
            if r["id"] not in man: continue
            src = "mathspeech-synthetic-compositional+kokoro" if f.name == "spoken_comp.jsonl" else "mathspeech-handwritten-ood+kokoro" if f.name == "spoken_ood.jsonl" else "mathspeech-synthetic-grammar+kokoro"
            rows.append(dict(id=r["id"], audio=rel(DATA_DIR / f"synthetic/audio/{r['split']}/{r['id']}.flac"), spoken_transcript=r["spoken_text"], target_text=r["latex"],
                             domain=r["domain"], synthetic=True, speaker_id=r["voice"], source=src, license=SYN_LICENSE, difficulty=r["difficulty"], template=r["template"],
                             split=r["split"], duration=man[r["id"]]["duration"], confidence=1.0, speed=r["speed"]))
    return rows


def load_real():
    f, rows = DATA_DIR / "real/real_manifest.jsonl", []
    if f.exists():
        for l in open(f):
            if not l.strip(): continue
            r = json.loads(l)
            if r.get("accepted"):
                rows.append(dict(id=r["id"], audio=rel(r["audio"]), spoken_transcript=r["asr_transcript"], target_text=r["canonical"], domain=r["domain"], synthetic=False,
                                 speaker_id=r.get("speaker_id", "unknown"), source=r["source"], license=r["license"], difficulty=r.get("difficulty", 3), template="real",
                                 split=r["split"], duration=r["duration"], confidence=r["confidence"], speed=1.0))
    return rows


def main():
    rows = load_synth() + load_real()
    out = DATA_DIR / "processed"; out.mkdir(parents=True, exist_ok=True)
    man = DATA_DIR / "manifests"; man.mkdir(parents=True, exist_ok=True)
    dd = {}
    for split, name in SPLITS.items():
        rs = [{k: v for k, v in r.items() if k != "split"} for r in rows if r["split"] == split]
        if not rs: continue
        dd[name] = Dataset.from_list(rs)
        with open(man / f"{name}.jsonl", "w") as f:
            for r in rs: f.write(json.dumps(r) + "\n")
    DatasetDict(dd).save_to_disk(str(out / "mathspeech"))
    print({k: len(v) for k, v in dd.items()}); print(f"real examples: {sum(1 for r in rows if not r['synthetic'])}")


if __name__ == "__main__":
    main()
