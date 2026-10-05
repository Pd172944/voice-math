# MathSpeech — speech recognition that writes mathematical notation

> General-purpose speech recognition is optimized for natural language. Mathematical speech is different: the acoustic
> realization of a symbol, its semantic role, and its written representation are tightly coupled.
>
> MathSpeech fine-tunes an open-source speech model on mathematical and scientific speech so that spoken mathematical
> expressions are transcribed directly into canonical notation.

![architecture](docs/architecture.svg)

| Spoken | Generic Whisper (large-v3-turbo) | **MathSpeech** |
|---|---|---|
| "the integral from zero to infinity of x to the sixth d x equals one third" | `That is the integral from 0 to infinity of x to the 6th dx equals 1 third.` | `\int_0^\infty x^6\,dx = \frac{1}{3}` |
| "for all n in Q, n squared is greater than or equal to zero" | `For all n in q n squared is greater than or equal to zero.` | `\forall n \in \mathbb{Q}, \; n^2 \geq 0` |
| "A bold v sub i equals lambda sub i bold v sub i" | `A bold V sub i equals lambda sub i bold V sub i.` | `A\mathbf{v}_i = \lambda_i\mathbf{v}_i` |
| "x is distributed as normal mu sigma squared" | `x is distributed as normal mu sigma squared.` | `X \sim \mathcal{N}(\mu, \sigma^2)` |
| "the inner product of bold x and bold y" | `the inner product of Bold X and Bold Y.` | `\langle \mathbf{x}, \mathbf{y} \rangle` |

More (with audio players and rendered equations): `artifacts/evaluation/report.html`; 18 curated pairs in `artifacts/demo/examples.json`.

## What was built

* **Model:** `openai/whisper-large-v3-turbo` (809 M, MIT) + LoRA (r=32, 29.6 M trainable params, incl. the output head). Fallback: `openai/whisper-small.en`.
  *Why Whisper here:* best-in-class English ASR, native HF/PEFT support, byte-level BPE tokenizer that can spell LaTeX, permissive licence, fine-tunes in <1 h on one H100.
  Qwen2-Audio / NeMo were considered; they are heavier (7 B) or need a separate toolchain for a 5-day prototype.
* **Data:** 14,917 train / 184 val / 433 test / 54 hard-test synthetic clips (13.8 h train) — 10 domains (algebra, calculus, linear algebra, probability, set theory, analysis, ML, physics, chemistry, CS),
  68 grammar templates, 8 train voices + 2 held-out voices (Kokoro-82M). **LaTeX is the source of truth**; the spoken form is generated from the same draws. Splits are by LaTeX hash (verified: zero canonical-LaTeX overlap with train),
  and the *hard* test set uses 9 grammar templates never seen in training, spoken by unseen voices.
* **Normalization layer** (`src/normalization/`): lexicon + structured parser (sub/superscripts, fractions, integrals, sums, limits, derivatives, partials, norms, probability, vectors…) + context gating
  (`"the pie is on the table"` stays text, `"pi is approximately three point fourteen"` → `\pi \approx 3.14`) + validity filters + optional LLM fallback (OpenRouter; off by default). 95% round-trip on its own grammar's phrasing.
* **Real audio pipeline** (`scripts/build_real_corpus.py`): HF streaming → ASR → math detection → canonicalize → deterministic filters, with full provenance (source, licence, ASR text, canonical, confidence, method, transform). Rejected rows are kept with reasons.
* **Demo** (`app/app.py`, Gradio): upload/record audio → generic ASR, generic+rules, MathSpeech LaTeX and a rendered equation; example buttons from the held-out set.
* **Tracking:** W&B project `prithvidixit05-/voiceTrain` (runs `canonical-v3-headlora`, `spoken-ablation`, `canonical-v2`).

## Results (single H100, GPU 0)

Systems — **A** generic Whisper (raw) · **C** generic Whisper + rule normalizer · **B** MathSpeech (direct LaTeX) · **D** ablation: Whisper LoRA on *spoken* text + the same rule normalizer.

**Test** (433 unseen expressions, held-out voices included; same grammar families as training)

| metric | A generic | C generic+rules | **B MathSpeech** | D spoken-LoRA+rules |
|---|---|---|---|---|
| normalized exact match | 0.000 | 0.503 | **0.912** | 0.901 |
| LaTeX token edit distance ↓ | 2.729 | 0.420 | **0.012** | 0.068 |
| symbol acc | 0.001 | 0.941 | 0.989 | 0.992 |
| Greek acc | 0.000 | 0.943 | 0.977 | 0.989 |
| operator acc | 0.001 | 0.978 | 0.997 | 0.999 |
| sub/superscript acc | 0.000 | 0.901 | 0.987 | 0.986 |
| bracket acc | 0.000 | 0.750 | 1.000 | 0.986 |
| ordinary WER vs spoken text | 0.117 | – | – | 0.003 |

**Hard test** (54 clips; *unseen grammar templates*, unseen voices)

| metric | A | C | B MathSpeech | D |
|---|---|---|---|---|
| exact match | 0.000 | 0.426 | 0.111 | **0.444** |
| token edit ↓ | 2.674 | 0.318 | 0.368 | **0.192** |

**Real human speech, evaluation only** (AAAI2025/MathSpeech, 1,101 YouTube-sourced clips, labels cleaned; model never trained on any real audio)

| | A generic | C generic+rules | B MathSpeech |
|---|---|---|---|
| token edit ↓ | 2.010 | 0.654 | **0.357** |
| script acc | 0.000 | 0.581 | **0.750** |
| symbol acc | 0.011 | **0.713** | 0.686 |
| exact (loose labels) | 0.000 | 0.148 | **0.172** |

(Generic Whisper WER on those clips vs the spoken transcript: 0.314, including hallucinated repetition loops such as "by by by …".)

**Training**: LoRA r=32/α=64 on q,k,v,out,fc1,fc2 + `proj_out`; lr 1.5e-4 cosine, 100 warm-up, 4 epochs, batch 16×2 accumulation, bf16 autocast, seed 1234.
1,864 steps, **39 min** on one NVIDIA H100 80GB, best val loss 0.032. Spoken-target ablation: 2 epochs, 21 min.

### Honest reading of the results

1. **The task is learnable and the gain is large where the training grammar covers the structure.** Direct canonical fine-tuning roughly halves the remaining error versus generic ASR + hand-written rules (0.50 → 0.91 exact; token edit 0.42 → 0.012) and also transfers partly to real human speech it never saw.
2. **It does not generalize to new structures.** On held-out templates the direct model collapses (0.11 exact) while "fine-tune ASR on words, then normalize" (D) holds up (0.44) — the normalizer supplies compositional structure the LoRA model has only memorized from 68 templates. Answer to the "direct vs. ASR + normalizer" question for this prototype: direct is best in-distribution; the two-stage system is more robust. A real fix is far more diverse text (LLM-written spoken math, real lectures).
3. **Baseline C is not independent of the data generator.** I wrote the rule normalizer against the same family of spoken phrasings, so C (and D) are favored on synthetic data; the real-speech probe is the fairer comparison and shows a smaller gap on symbol accuracy.
4. **The first LoRA run capped at 79% exact** because Whisper's default decoding *suppresses symbol tokens* (`\ { } ^ _ …`) and the LoRA didn't touch the tied output head; fixing both (generation without suppression, `proj_out` LoRA) took it to 91% (val loss 0.437 → 0.032). Both are documented in the code.
5. Small per-domain counts (ML 4, physics 8 test examples) make the per-domain table anecdotal; chemistry/CS are not in the standard test split because of the LaTeX-hash split.

## Reproduce

```bash
git clone https://github.com/Pd172944/voice-math.git && cd voice-math
cp .env.example .env            # optional: HF_TOKEN, WANDB_API_KEY, OPENROUTER_API_KEY
bash scripts/run_all.sh         # everything; idempotent, resumes after failure (SKIP_REAL=1 to skip the real-audio stage)
python app/app.py               # demo at http://localhost:7860
python scripts/show_dataset.py -n 8   # inspect random examples (+ artifacts/datasets/sample.html with KaTeX)
python scripts/eval_external.py       # real-speech probe (downloads an eval-only dataset to data/raw/, not redistributed)
make test
```

Single-GPU by design: `CUDA_VISIBLE_DEVICES` defaults to 0. Paths come from `DATA_DIR`, `OUTPUT_DIR`, `MODEL_ID`, `HF_HOME` (see `.env.example`). Wall time on one H100 + a CPU-heavy host: TTS ≈ 6 min, training ≈ 39 min (+21 min ablation), eval ≈ 10 min.

## Layout
`src/data/grammar.py` expression grammar · `src/normalization/` lexicon, rules, parser, LLM fallback, canonicalize · `src/evaluation/metrics.py` · `src/inference/asr.py` ·
`scripts/` pipeline steps (discover → generate_math → generate_spoken → generate_tts → build_real_corpus → build_dataset → validate_dataset → train → evaluate → generate_demo → generate_report) · `app/app.py` · `configs/` · `tests/`.
Model card: `docs/MODEL_CARD.md` · Dataset card: `docs/DATASET_CARD.md`.

## Limitations
* Training audio is **synthetic TTS** (one engine, clean read speech, no noise/accents/disfluency). Real-world robustness is only probed, not trained.
* **Real-audio corpus is empty in this run.** The licensed source we could stream (People's Speech, CC-BY) is almost all civic/meeting speech: of 1,500 keyword-prefiltered clips, 0 passed the strict filters (1,022 low-confidence, 441 not math, 36 unknown words) — the filters worked, there was just no spoken math. Existing math-speech datasets found on the Hub (e.g. `AAAI2025/MathSpeech`) have **no declared licence**, so they are used only as an evaluation probe, never for training or redistribution.
* LaTeX style is one convention (our grammar's); ambiguous speech ("x sub i j", "R n") is resolved by convention. Letter confusions (b/d/p/t/v) in TTS audio are the main residual acoustic error.
* Rule normalizer + LLM fallback are untested on long natural lecture sentences (no span-level math detection yet).
* Weights (113 MB LoRA adapters) are not committed to git (size); `run_all.sh` regenerates them.

## Future work
Span-level math detection for lecture audio · LLM-paraphrased and LLM-composed spoken math for structural diversity · second TTS engine + noise/room augmentation · real lecture data with permissive licences (and human-verified test set) · constrained decoding (balanced braces, valid commands) · publish adapters to the HF Hub.
