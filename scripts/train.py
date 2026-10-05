#!/usr/bin/env python
"""LoRA fine-tuning for math-aware ASR. Backends (auto-detected from --model): Qwen3-ASR (default) and Whisper.
target_mode=canonical -> LaTeX, target_mode=spoken -> plain spoken words. Resumable (adapter+optimizer every --save_steps), seeded, W&B logging if a key exists.
Train-time augmentation (random gain, additive noise) is applied on the fly."""
import argparse, io, json, os, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import soundfile as sf
import yaml
from src.env import DATA_DIR, OUTPUT_DIR, ROOT, MODEL_ID

QWEN_TARGETS = (r".*(language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
                r"|audio_tower\..*self_attn\.(q_proj|k_proj|v_proj|out_proj)|multi_modal_projector\.linear_[12])")


def args():
    cfg = yaml.safe_load((ROOT / "configs/training.yaml").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID); ap.add_argument("--dataset", default=str(DATA_DIR / "processed/mathspeech"))
    ap.add_argument("--output_dir", default=str(OUTPUT_DIR / "model/qwen3asr-canonical")); ap.add_argument("--epochs", type=float, default=cfg["epochs"])
    ap.add_argument("--learning_rate", type=float, default=float(cfg["learning_rate"])); ap.add_argument("--batch_size", type=int, default=cfg["batch_size"])
    ap.add_argument("--gradient_accumulation", type=int, default=cfg["gradient_accumulation"]); ap.add_argument("--max_audio_seconds", type=float, default=cfg["max_audio_seconds"])
    ap.add_argument("--target_mode", choices=["canonical", "spoken"], default=cfg["target_mode"]); ap.add_argument("--seed", type=int, default=cfg["seed"])
    ap.add_argument("--lora_r", type=int, default=cfg["lora"]["r"]); ap.add_argument("--eval_steps", type=int, default=250); ap.add_argument("--save_steps", type=int, default=250)
    ap.add_argument("--max_steps", type=int, default=-1); ap.add_argument("--warmup_steps", type=int, default=cfg["warmup_steps"]); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--run_name", default=None); ap.add_argument("--max_train_examples", type=int, default=0); ap.add_argument("--augment", type=float, default=cfg.get("augment", 0.5))
    return ap.parse_args(), cfg


def read_audio(path, max_s, aug, rng):
    p = Path(path); p = p if p.is_absolute() else ROOT / p
    a, sr = sf.read(str(p), dtype="float32")
    a = a[: int(16000 * max_s)]
    if aug and rng.random() < aug:                       # gain + additive noise (SNR 15-35 dB)
        a = a * 10 ** (rng.uniform(-6, 6) / 20)
        if rng.random() < 0.7:
            snr = rng.uniform(15, 35); noise = np.random.default_rng(rng.randrange(2**31)).standard_normal(len(a)).astype("float32")
            a = a + noise * (np.sqrt((a ** 2).mean() + 1e-9) / (10 ** (snr / 20)) / (noise.std() + 1e-9))
        a = np.clip(a, -1.0, 1.0)
    return a


class WhisperCollate:
    def __init__(self, model_id, mode, max_s, aug):
        self.model_id, self.mode, self.max_s, self.aug = model_id, mode, max_s, aug; self.fe = self.tok = None

    def __call__(self, batch):
        if self.fe is None:
            from transformers import WhisperProcessor
            p = WhisperProcessor.from_pretrained(self.model_id); self.fe, self.tok = p.feature_extractor, p.tokenizer; self.tok.set_prefix_tokens(language="en", task="transcribe")
        rng = random.Random()
        feats = self.fe([read_audio(b["audio"], self.max_s, self.aug, rng) for b in batch], sampling_rate=16000, return_tensors="pt").input_features
        key = "target_text" if self.mode == "canonical" else "spoken_transcript"
        ids = [self.tok(b[key]).input_ids for b in batch]
        sot = self.tok.convert_tokens_to_ids("<|startoftranscript|>")
        ids = [x[1:] if x and x[0] == sot else x for x in ids]
        L = max(map(len, ids)); labels = torch.full((len(ids), L), -100, dtype=torch.long)
        for i, x in enumerate(ids): labels[i, : len(x)] = torch.tensor(x)
        return dict(input_features=feats, labels=labels)


class QwenCollate:
    def __init__(self, model_id, mode, max_s, aug):
        self.model_id, self.mode, self.max_s, self.aug = model_id, mode, max_s, aug; self.proc = None

    def __call__(self, batch):
        if self.proc is None:
            from transformers import AutoProcessor
            self.proc = AutoProcessor.from_pretrained(self.model_id)
        rng = random.Random()
        inp = self.proc.apply_transcription_request([read_audio(b["audio"], self.max_s, self.aug, rng) for b in batch], language="English")
        key = "target_text" if self.mode == "canonical" else "spoken_transcript"
        tok = self.proc.tokenizer; pad = tok.pad_token_id if tok.pad_token_id is not None else 151643
        seqs, labs = [], []
        for i, b in enumerate(batch):
            prompt = inp["input_ids"][i][inp["attention_mask"][i].bool()]          # prompt incl. audio placeholders + "language English<asr_text>"
            tgt = torch.tensor(tok(b[key] + "<|im_end|>", add_special_tokens=False).input_ids)
            seqs.append(torch.cat([prompt, tgt])); labs.append(torch.cat([torch.full_like(prompt, -100), tgt]))
        L = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), L), pad, dtype=torch.long); lab = torch.full((len(seqs), L), -100, dtype=torch.long); am = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, (x, y) in enumerate(zip(seqs, labs)):
            ids[i, : len(x)] = x; lab[i, : len(y)] = y; am[i, : len(x)] = 1
        return dict(input_ids=ids, attention_mask=am, labels=lab, input_features=inp["input_features"], input_features_mask=inp["input_features_mask"])


def build_model(a, cfg):
    from peft import LoraConfig, get_peft_model
    lc = cfg["lora"]
    if "qwen3-asr" in a.model.lower():
        from transformers import AutoModelForMultimodalLM
        m = AutoModelForMultimodalLM.from_pretrained(a.model, dtype=torch.bfloat16); m.config.use_cache = False
        m = get_peft_model(m, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=lc["dropout"], target_modules=QWEN_TARGETS, bias="none"))
        return m, QwenCollate
    from transformers import WhisperForConditionalGeneration
    m = WhisperForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.float32); m.config.use_cache = False
    m = get_peft_model(m, LoraConfig(r=a.lora_r, lora_alpha=lc["alpha"], lora_dropout=lc["dropout"], target_modules=lc["whisper_target_modules"], bias="none"))
    return m, WhisperCollate


def to_cuda(b, model_is_qwen):
    b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
    if model_is_qwen: b["input_features"] = b["input_features"].to(torch.bfloat16)
    return b


def main():
    a, cfg = args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    from datasets import load_from_disk
    from transformers import get_cosine_schedule_with_warmup
    dd = load_from_disk(a.dataset)
    tr = dd["mathspeech_train"].filter(lambda d: d["duration"] <= a.max_audio_seconds)
    va = dd["mathspeech_validation"].shuffle(seed=0); va = va.select(range(min(500, len(va))))
    if a.max_train_examples: tr = tr.shuffle(seed=a.seed).select(range(min(a.max_train_examples, len(tr))))
    tr_rows, va_rows = tr.to_list(), va.to_list()
    print(f"train={len(tr_rows)} val={len(va_rows)} mode={a.target_mode} model={a.model}")

    model, Coll = build_model(a, cfg); is_q = Coll is QwenCollate
    model.print_trainable_parameters(); model.cuda()
    if is_q: model.enable_input_require_grads() if False else None
    dl = torch.utils.data.DataLoader(tr_rows, batch_size=a.batch_size, shuffle=True, collate_fn=Coll(a.model, a.target_mode, a.max_audio_seconds, a.augment), num_workers=a.workers,
                                     drop_last=True, generator=torch.Generator().manual_seed(a.seed), persistent_workers=True, prefetch_factor=4)
    vdl = torch.utils.data.DataLoader(va_rows, batch_size=a.batch_size, collate_fn=Coll(a.model, a.target_mode, a.max_audio_seconds, 0.0), num_workers=4)
    steps_per_epoch = len(dl) // a.gradient_accumulation
    total = int(steps_per_epoch * a.epochs) if a.max_steps < 0 else a.max_steps
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.learning_rate, weight_decay=0.01, betas=(0.9, 0.98))
    sch = get_cosine_schedule_with_warmup(opt, a.warmup_steps, total)

    step = 0; st = out / "state.json"
    if st.exists():                                        # ---- resume
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        step = json.loads(st.read_text())["step"]
        set_peft_model_state_dict(model, load_file(str(out / "last/adapter_model.safetensors")))
        o = torch.load(out / "optim.pt", map_location="cuda"); opt.load_state_dict(o["opt"]); sch.load_state_dict(o["sch"]); print(f"[resume] from step {step}")

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
                b = to_cuda(b, is_q); o = model(**b)
                k = (b["labels"][:, 1:] != -100).sum().item(); tot += o.loss.item() * k; n += k
        model.train(); return tot / max(1, n)

    def save(tag="last"):
        model.save_pretrained(str(out / tag))
        if tag == "last":
            torch.save({"opt": opt.state_dict(), "sch": sch.state_dict()}, out / "optim.pt"); st.write_text(json.dumps({"step": step}))

    model.train(); t0 = time.time(); best = 1e9; micro = 0; run_loss = 0.0; run_n = 0; done = step >= total
    while not done:
        for b in dl:
            b = to_cuda(b, is_q)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**b).loss / a.gradient_accumulation
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
    save("last")
    meta = dict(model=a.model, target_mode=a.target_mode, steps=step, train_examples=len(tr_rows), best_val_loss=best, train_seconds=time.time() - t0,
                gpu=torch.cuda.get_device_name(0), args=vars(a))
    (out / "training_summary.json").write_text(json.dumps(meta, indent=1)); print(meta)
    if wb: wb.finish()


if __name__ == "__main__":
    main()
