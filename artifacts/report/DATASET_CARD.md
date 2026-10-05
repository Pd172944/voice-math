# MathSpeech-synthetic — dataset card

Splits (HF `DatasetDict` at `data/processed/mathspeech`; manifests in `data/manifests/*.jsonl`; audio stored as relative paths): `mathspeech_train` 27,243 (35.9 h) · `validation` 442 · `test` 433 · `test_comp` 416 · `test_hard` 54 · `test_ood` 128.

Fields: `audio` (path to 16 kHz FLAC), `spoken_transcript`, `target_text` (canonical LaTeX), `domain`, `synthetic`, `speaker_id`, `source`, `license`, `difficulty` (1–5), `template`, `duration`, `confidence`, `speed`, `id`.

* **Sources:** (1) probabilistic grammar, 68 templates / 10 domains (`src/data/grammar.py`); (2) compositional random trees (`src/data/compose.py`) with unambiguous spoken grouping; (3) 64 hand-written OOD expressions (`src/data/ood.py`, test only).
* **Audio:** Kokoro-82M (Apache-2.0). Train voices: af_heart, af_bella, af_nicole, af_sarah, am_adam, am_michael, am_puck, af_kore (+ af_alloy, af_aoede, af_nova, am_echo, am_eric, am_liam for the compositional corpus); held-out voices af_sky, am_fenrir. Single letters and Greek/jargon words get explicit phoneme markup; compositional train clips get a random ±7 % pitch/tempo shift.
* **Leakage control:** splits by MD5 of whitespace-stripped LaTeX; `validate_dataset.py` checks that no validation/test canonical LaTeX (after normalization) appears in train; it caught and forced replacement of 3 near-duplicate OOD items. `test_hard` templates and `test_ood` are never in train.
* **Licence:** synthetic text (MIT, this repo) + TTS output; no third-party audio included. Optional real-audio pipeline source: People's Speech (CC-BY / CC-BY-SA) — 0 clips accepted in this run. `AAAI2025/MathSpeech` (licence undeclared) is evaluation-only and not redistributed.
* **Known gaps:** TTS-only, one LaTeX convention, calculus/algebra-heavy, repeated expressions across voices in train.
