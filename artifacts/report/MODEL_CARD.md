# MathSpeech (Whisper large-v3-turbo + LoRA) — model card

* **Base:** `openai/whisper-large-v3-turbo` (MIT). **Adapter:** LoRA r=32, α=64, dropout 0.05 on q/k/v/out_proj, fc1/fc2 and `proj_out` (29.6 M trainable params).
* **Task:** English speech → canonical LaTeX for spoken mathematics (`target_mode=canonical`). A sibling adapter (`target_mode=spoken`) is the plain-ASR ablation.
* **Training data:** 14,917 synthetic clips (Kokoro-82M TTS, 8 voices, speed 0.85–1.15) of 3,701 distinct LaTeX expressions from a 68-template grammar. No real recordings.
* **Training:** 1,864 steps, effective batch 32, lr 1.5e-4 cosine, bf16 autocast, seed 1234, 1×H100, 39 min. W&B: `prithvidixit05-/voiceTrain`.
* **Decoding:** greedy; must run with Whisper's `suppress_tokens` / `begin_suppress_tokens` disabled (see `src/inference/asr.py`), otherwise `\ { } ^ _` cannot be emitted.
* **Results:** test exact match 0.912 (generic Whisper 0.0; + rules 0.503); unseen-template test 0.111; real human speech probe token-edit 0.357 vs 0.654 (generic + rules). See README.
* **Intended use:** research/demo of math-aware ASR; dictating simple formulas. **Not** for transcription where errors matter without human review.
* **Limitations:** trained on clean synthetic speech; weak on structures outside the grammar; output follows one LaTeX style; may emit confident but wrong symbols (b/d/p/t confusions).
* **Licence:** code MIT; base model MIT; Kokoro-82M Apache-2.0. Adapter weights inherit these terms.
