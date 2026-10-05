#!/usr/bin/env python
"""Step 3: synthesize audio with Kokoro-82M (multi-voice, speed-varied) -> 16kHz FLAC. Resumable; parallel workers share GPU 0."""
import argparse, json, os, random, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from src.env import DATA_DIR, ROOT


def worker(shard, nshards, inp, audio_dir, sr, limit, pitch_aug=0.0):
    import numpy as np, soundfile as sf, librosa, torch
    from kokoro import KPipeline
    from src.data.tts_text import to_tts
    pipe = KPipeline(lang_code="a", device="cuda", repo_id="hexgrad/Kokoro-82M")
    rows = [json.loads(l) for l in open(inp)]
    rows = rows[:limit] if limit else rows
    mine = rows[shard::nshards]
    ok = fail = 0; t0 = time.time(); man = open(f"{inp}.tts.{shard}.jsonl", "a")
    for n, r in enumerate(mine):
        path = audio_dir / r["split"] / f"{r['id']}.flac"
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = random.Random(sum(map(ord, r["id"])) * 7919)
        for attempt in range(3):      # retry (fall back to plain text on later attempts)
            try:
                txt = to_tts(r["spoken_text"], rng, pauses=attempt == 0)
                chunks = [a for _, _, a in pipe(txt, voice=r["voice"], speed=r["speed"])]
                wav = np.concatenate([c.numpy() if hasattr(c, "numpy") else c for c in chunks])
                fac = rng.uniform(0.93, 1.07) if (r['split'] == 'train' and rng.random() < pitch_aug) else 1.0   # pitch+tempo shift by resampling trick
                wav16 = librosa.resample(wav.astype("float32"), orig_sr=int(24000 * fac), target_sr=sr)
                wav16 = wav16 / max(1e-6, abs(wav16).max()) * 0.9
                sf.write(path, wav16, sr, format="FLAC")
                man.write(json.dumps(dict(id=r["id"], duration=len(wav16) / sr, tts_text=txt)) + "\n"); ok += 1
                break
            except Exception as e:
                if attempt == 2:
                    fail += 1; print(f"[tts-fail] {r['id']}: {e}", flush=True)
        if n % 200 == 0:
            print(f"[shard {shard}] {n}/{len(mine)} ok={ok} fail={fail} {time.time()-t0:.0f}s", flush=True)
    man.close()


def main():
    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=str(DATA_DIR / "synthetic/spoken.jsonl")); ap.add_argument("--workers", type=int, default=cfg["tts"]["workers"])
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--shard", type=int, default=-1); ap.add_argument("--pitch_aug", type=float, default=None)
    a = ap.parse_args()
    audio_dir = DATA_DIR / "synthetic/audio"; sr = cfg["tts"]["sample_rate"]
    pa = a.pitch_aug if a.pitch_aug is not None else (0.0 if Path(a.inp).name == "spoken.jsonl" else cfg["tts"].get("pitch_aug_prob", 0.0))
    if a.shard >= 0:
        worker(a.shard, a.workers, a.inp, audio_dir, sr, a.limit, pa); return
    procs = [subprocess.Popen([sys.executable, __file__, "--inp", a.inp, "--workers", str(a.workers), "--limit", str(a.limit), "--shard", str(s), "--pitch_aug", str(pa)],
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")}) for s in range(a.workers)]
    for p in procs: p.wait()
    # merge shard manifests
    man = DATA_DIR / "manifests/tts_manifest.jsonl"; man.parent.mkdir(parents=True, exist_ok=True)
    durs = {json.loads(l)["id"]: json.loads(l) for l in open(man)} if man.exists() else {}   # merge with earlier runs
    for s in range(a.workers):
        f = Path(f"{a.inp}.tts.{s}.jsonl")
        if f.exists():
            for l in open(f):
                d = json.loads(l); durs[d["id"]] = d
    with man.open("w") as f:
        for d in durs.values(): f.write(json.dumps(d) + "\n")
    print(f"{len(durs)} clips; manifest -> {man}")


if __name__ == "__main__":
    main()
