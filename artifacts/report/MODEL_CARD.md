# MathSpeech (Qwen3-ASR-1.7B + LoRA) — model card

* **Base:** `Qwen/Qwen3-ASR-1.7B-hf` (Apache-2.0, released June 2026). **Adapter:** LoRA r=32, α=64, dropout 0.05 on the language-model attention+MLP, audio-tower attention and multimodal projector (41 M trainable params).
* **Task:** English speech → canonical LaTeX for spoken mathematics (`target_mode=canonical`). A readable Unicode rendering is produced on the fly by `src/rendering/latex_to_text.py`.
* **Training data:** 27,243 synthetic clips (35.9 h): 68-template grammar + compositional random trees, Kokoro-82M TTS (14 train voices, speed and pitch variation) with on-the-fly gain/noise augmentation. No real recordings.
* **Training:** 2,550 steps, effective batch 32, lr 1e-4 cosine, bf16 autocast, seed 1234, 1×H100, 25.5 min. W&B: `prithvidixit05-/voiceTrain`.
* **Results:** exact match 0.975 (unseen expressions), 0.940 (unseen compositions), 0.797 (hand-written OOD set), 0.481 (unseen templates); generic Qwen3-ASR scores 0.0 on all and generic+rules 0.48 / 0.22 / 0.41 / 0.56. On 1,101 real human recordings (eval-only) token-edit 0.264 vs 0.527 (generic+rules). See README.
* **Intended use:** research/demo of math-aware ASR; dictating formulas with human review. **Not** for settings where an unnoticed error is costly.
* **Limitations:** trained on clean synthetic speech; weaker on structures far outside the grammar; one LaTeX style; capitalization (X vs x) is not audible and is guessed.
* **Licence:** code MIT; base model Apache-2.0; Kokoro-82M Apache-2.0.
