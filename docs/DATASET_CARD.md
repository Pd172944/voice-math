# MathSpeech-synthetic — dataset card

Splits (HF `DatasetDict` saved to `data/processed/mathspeech`; manifests in `data/manifests/*.jsonl`): `mathspeech_train` 14,917 (13.8 h) · `mathspeech_validation` 184 · `mathspeech_test` 433 · `mathspeech_test_hard` 54.

Fields: `audio` (16 kHz FLAC), `spoken_transcript`, `target_text` (canonical LaTeX), `domain`, `synthetic`, `speaker_id`, `source`, `license`, `difficulty` (1–5), `template`, `duration`, `confidence`, `speed`.

* **Generation:** hand-written probabilistic grammar (`src/data/grammar.py`, 68 templates, 10 domains). Spoken text and LaTeX come from the same random draws; spoken variants include 15% lead-in fillers ("so", "we have", …).
* **Audio:** Kokoro-82M (Apache-2.0) via `scripts/generate_tts.py`; single-letter symbols and Greek/jargon words get explicit phoneme markup. Train voices `af_heart, af_bella, af_nicole, af_sarah, am_adam, am_michael, am_puck, af_kore`; held-out `af_sky, am_fenrir`.
* **Leakage control:** splits by MD5 of whitespace-stripped LaTeX (verified: 0 overlap); `test_hard` uses 9 templates excluded from train/val/test, spoken only by held-out voices.
* **Licence:** synthetic text (MIT, this repo) + TTS output; no third-party audio included. Real-audio source for the optional pipeline: People's Speech (CC-BY / CC-BY-SA per item) — yielded 0 accepted examples in this run. `AAAI2025/MathSpeech` (licence undeclared) is evaluation-only and never redistributed.
* **Known gaps:** TTS-only, one LaTeX convention, skewed domain sizes (calculus/algebra largest; chemistry/physics ≈360 each), repeated expressions with different voices (3,701 unique LaTeX in train).
