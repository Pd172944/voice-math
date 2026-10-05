#!/usr/bin/env python
"""Step 5: merge synthetic (+ real, if present) into manifests and a HF DatasetDict: mathspeech_{train,validation,test}."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.env import DATA_DIR, OUTPUT_DIR
from datasets import Dataset, DatasetDict, Audio

SYN_LICENSE = "synthetic: grammar-generated text (MIT) + Kokoro-82M TTS (Apache-2.0)"


def load_synth():
    man = {json.loads(l)["id"]: json.loads(l) for l in open(DATA_DIR / "manifests/tts_manifest.jsonl")}
    rows = []
    for l in open(DATA_DIR / "synthetic/spoken.jsonl"):
        r = json.loads(l)
        if r["id"] not in man:
            continue
        sp = {"test_hard": "test_hard"}.get(r["split"], r["split"])
        rows.append(dict(id=r["id"], audio=str(DATA_DIR / f"synthetic/audio/{r['split']}/{r['id']}.flac"), spoken_transcript=r["spoken_text"],
                         target_text=r["latex"], domain=r["domain"], synthetic=True, speaker_id=r["voice"], source="mathspeech-synthetic-grammar+kokoro",
                         license=SYN_LICENSE, difficulty=r["difficulty"], template=r["template"], split=sp, duration=man[r["id"]]["duration"],
                         confidence=1.0, speed=r["speed"]))
    return rows


def load_real():
    f = DATA_DIR / "real/real_manifest.jsonl"
    rows = []
    if f.exists():
        for l in open(f):
            r = json.loads(l)
            if r.get("accepted"):
                rows.append(dict(id=r["id"], audio=r["audio"], spoken_transcript=r["asr_transcript"], target_text=r["canonical"], domain=r["domain"],
                                 synthetic=False, speaker_id=r.get("speaker_id", "unknown"), source=r["source"], license=r["license"], difficulty=r.get("difficulty", 3),
                                 template="real", split=r["split"], duration=r["duration"], confidence=r["confidence"], speed=1.0))
    return rows


def main():
    rows = load_synth() + load_real()
    out = DATA_DIR / "processed"; out.mkdir(parents=True, exist_ok=True)
    dd = {}
    for split in ("train", "validation", "test", "test_hard"):
        rs = [dict(r) for r in rows if r["split"] == split]
        if not rs:
            continue
        for r in rs: r.pop("split")
        ds = Dataset.from_list(rs).cast_column("audio", Audio(sampling_rate=16000))
        dd[{"train": "mathspeech_train", "validation": "mathspeech_validation", "test": "mathspeech_test", "test_hard": "mathspeech_test_hard"}[split]] = ds
    DatasetDict(dd).save_to_disk(str(out / "mathspeech"))
    # jsonl manifests (paths only, cheap to inspect/diff)
    man = DATA_DIR / "manifests"
    for name, ds in dd.items():
        with open(man / f"{name}.jsonl", "w") as f:
            for r in ds.remove_columns("audio") if False else ds.cast_column("audio", Audio(decode=False)):
                r = dict(r); r["audio"] = r["audio"]["path"]; f.write(json.dumps(r) + "\n")
    print({k: len(v) for k, v in dd.items()})
    print(f"real examples: {sum(1 for r in rows if not r['synthetic'])}")


if __name__ == "__main__":
    main()
