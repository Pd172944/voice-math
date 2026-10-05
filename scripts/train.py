#!/usr/bin/env python
"""LoRA fine-tuning of Whisper for canonical-math (target_mode=canonical) or plain spoken (target_mode=spoken) transcription.
Resumable (adapter + optimizer + step saved every --save_steps and each epoch), deterministic seed, W&B logging if a key is available."""
import argparse, json, math, os, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import soundfile as sf
import yaml
from src.env import DATA_DIR, OUTPUT_DIR, ROOT, MODEL_ID


def args():
    cfg = yaml.safe_load((ROOT / "configs/training.yaml").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID); ap.add_argument("--dataset", default=str(DATA_DIR / "processed/mathspeech"))
    ap.add_argument("--output_dir", default=str(OUTPUT_DIR / "model/mathspeech-canonical")); ap.add_argument("--epochs", type=float, default=cfg["epochs"])
    ap.add_argument("--learning_rate", type=float, default=float(cfg["learning_rate"])); ap.add_argument("--batch_size", type=int, default=cfg["batch_size"])
    ap.add_argument("--gradient_accumulation", type=int, default=cfg["gradient_accumulation"]); ap.add_argument("--max_audio_seconds", type=float, default=cfg["max_audio_seconds"])
    ap.add_argument("--target_mode", choices=["canonical", "spoken"], default=cfg["target_mode"]); ap.add_argument("--seed", type=int, default=cfg["seed"])
    ap.add_argument("--lora_r", type=int, default=cfg["lora"]["r"]); ap.add_argument("--eval_steps", type=int, default=200); ap.add_argument("--save_steps", type=int, default=300)
    ap.add_argument("--max_steps", type=int, default=-1); ap.add_argument("--warmup_steps", type=int, default=cfg["warmup_steps"]); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--run_name", default=None); ap.add_argument("--max_train_examples", type=int, default=0)
    return ap.parse_args(), cfg


class Collate:
    def __init__(self, model_id, mode, max_s):
        self.model_id, self.mode, self.max_s = model_id, mode, max_s
        self.fe = self.tok = None

    def _init(self):
        from transformers import WhisperProcessor
        p = WhisperProcessor.from_pretrained(self.model_id)
        self.fe, self.tok = p.feature_extractor, p.tokenizer
        self.tok.set_prefix_tokens(language="en", task="transcribe")

    def __call__(self, batch):
        if self.fe is None: self._init()
        arrs = []
        for b in batch:
            au = b["audio"]
            import io
            a, sr = sf.read(io.BytesIO(au["bytes"]) if au.get("bytes") else au["path"], dtype="float32")
            arrs.append(a[: int(16000 * self.max_s)])
        feats = self.fe(arrs, sampling_rate=16000, return_tensors="pt").input_features
        key = "target_text" if self.mode == "canonical" else "spoken_transcript"
        ids = [self.tok(b[key]).input_ids for b in batch]
        sot = self.tok.convert_tokens_to_ids("<|startoftranscript|>")
        ids = [x[1:] if x and x[0] == sot else x for x in ids]      # decoder_start_token (sot) is prepended by the model's right-shift
        L = max(map(len, ids))
        labels = torch.full((len(ids), L), -100, dtype=torch.long)
        for i, x in enumerate(ids): labels[i, : len(x)] = torch.tensor(x)
        return dict(input_features=feats, labels=labels)


def main():
    a, cfg = args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    from datasets import load_from_disk, Audio, concatenate_datasets
    from transformers import WhisperForConditionalGeneration, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model, PeftModel
    dd = load_from_disk(a.dataset)
    tr = dd["mathspeech_train"].cast_column("audio", Audio(decode=False))
    va = dd["mathspeech_validation"].cast_column("audio", Audio(decode=False))
    tr = tr.filter(lambda d: d["duration"] <= a.max_audio_seconds)
    if a.max_train_examples: tr = tr.shuffle(seed=a.seed).select(range(min(a.max_train_examples, len(tr))))
    va = va.shuffle(seed=0).select(range(min(400, len(va))))
    print(f"train={len(tr)} val={len(va)} mode={a.target_mode}")

    model = WhisperForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.float32)
    model.config.use_cache = False
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=lc["alpha"], lora_dropout=lc["dropout"], target_modules=lc["target_modules"], bias="none"))
    model.print_trainable_parameters()
    model.cuda()

    coll = Collate(a.model, a.target_mode, a.max_audio_seconds)
    g = torch.Generator(); g.manual_seed(a.seed)
    dl = torch.utils.data.DataLoader(tr, batch_size=a.batch_size, shuffle=True, collate_fn=coll, num_workers=a.workers, drop_last=True, generator=g, persistent_workers=True, prefetch_factor=4)
    vdl = torch.utils.data.DataLoader(va, batch_size=a.batch_size, collate_fn=coll, num_workers=4)
    steps_per_epoch = len(dl) // a.gradient_accumulation
    total = int(steps_per_epoch * a.epochs) if a.max_steps < 0 else a.max_steps
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.learning_rate, weight_decay=0.01, betas=(0.9, 0.98))
    sch = get_cosine_schedule_with_warmup(opt, a.warmup_steps, total)

    step = 0
    st = out / "state.json"
    if st.exists():                                        # ---- resume
        s = json.loads(st.read_text()); step = s["step"]
        model.load_adapter(str(out / "last"), adapter_name="default", is_trainable=True) if False else None
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(str(out / "last/adapter_model.safetensors")))
        o = torch.load(out / "optim.pt", map_location="cuda"); opt.load_state_dict(o["opt"]); sch.load_state_dict(o["sch"])
        print(f"[resume] from step {step}")

    wb = None
    if os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") != "disabled":
        try:
            import wandb
            wandb.login(key=os.environ["WANDB_API_KEY"])
            wb = wandb.init(project=os.environ.get("WANDB_PROJECT", "voiceTrain"), entity=os.environ.get("WANDB_ENTITY"), name=a.run_name or out.name,
                            config={**vars(a), **cfg}, resume="allow", id=(out.name + "-" + str(a.seed)).replace("/", "_"))
        except Exception as e:
            print(f"[warn] wandb disabled: {e}")
    log = open(out / "train_log.jsonl", "a")

    def evaluate():
        model.eval(); tot = n = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for b in vdl:
                o = model(input_features=b["input_features"].cuda(), labels=b["labels"].cuda())
                k = (b["labels"][:, 1:] != -100).sum().item(); tot += o.loss.item() * k; n += k
        model.train(); return tot / max(1, n)

    def save(tag="last"):
        model.save_pretrained(str(out / tag))
        if tag == "last":
            torch.save({"opt": opt.state_dict(), "sch": sch.state_dict()}, out / "optim.pt")
            st.write_text(json.dumps({"step": step}))

    model.train(); t0 = time.time(); best = 1e9; micro = 0; run_loss = 0.0; run_n = 0
    done = step >= total
    ep = 0
    while not done:
        for b in dl:
            if micro % a.gradient_accumulation == 0 and False: pass
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_features=b["input_features"].cuda(non_blocking=True), labels=b["labels"].cuda(non_blocking=True)).loss / a.gradient_accumulation
            loss.backward(); micro += 1; run_loss += loss.item() * a.gradient_accumulation; run_n += 1
            if micro % a.gradient_accumulation: continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 10 == 0:
                rec = dict(step=step, epoch=step / steps_per_epoch, loss=run_loss / run_n, lr=sch.get_last_lr()[0], elapsed=time.time() - t0)
                run_loss = run_n = 0; log.write(json.dumps(rec) + "\n"); log.flush()
                if wb: wb.log({"train/loss": rec["loss"], "train/lr": rec["lr"], "train/epoch": rec["epoch"]}, step=step)
                if step % 50 == 0: print(rec, flush=True)
            if step % a.eval_steps == 0 or step == total:
                vl = evaluate(); rec = dict(step=step, val_loss=vl); log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
                if wb: wb.log({"val/loss": vl}, step=step)
                if vl < best: best = vl; save("best")
            if step % a.save_steps == 0: save("last")
            if step >= total: done = True; break
        ep += 1
    save("last"); save("final")
    meta = dict(model=a.model, target_mode=a.target_mode, steps=step, train_examples=len(tr), best_val_loss=best, train_seconds=time.time() - t0,
                gpu=torch.cuda.get_device_name(0), args=vars(a))
    (out / "training_summary.json").write_text(json.dumps(meta, indent=1)); print(meta)
    if wb: wb.finish()


if __name__ == "__main__":
    main()
