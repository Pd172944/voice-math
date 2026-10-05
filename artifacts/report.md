# MathSpeech – experiment results


## Test — unseen expressions (template families seen in training)

| metric | A. Qwen3-ASR zero-shot | C. A + rule normaliser | B. MathSpeech (Qwen3-ASR+LoRA) | Whisper v3-turbo zero-shot | Whisper + rules | Whisper + LoRA (v1) |
|---|---|---|---|---|---|---|
| exact_match | 0.000 | 0.480 | 0.975 | 0.000 | 0.503 | 0.912 |
| latex_token_edit | 2.816 | 0.246 | 0.003 | 2.729 | 0.420 | 0.012 |
| symbol_acc | 0.000 | 0.958 | 0.997 | 0.001 | 0.941 | 0.989 |
| greek_acc | 0.000 | 0.920 | 0.989 | 0.000 | 0.943 | 0.977 |
| operator_acc | 0.000 | 0.992 | 1.000 | 0.001 | 0.978 | 0.997 |
| script_acc | 0.000 | 0.966 | 0.987 | 0.000 | 0.901 | 0.987 |
| bracket_acc | 0.000 | 0.953 | 1.000 | 0.000 | 0.750 | 1.000 |
| wer_vs_spoken | 0.057 | – | – | 0.117 | – | – |

## Test-comp — unseen random compositions of known constructs

| metric | A. Qwen3-ASR zero-shot | C. A + rule normaliser | B. MathSpeech (Qwen3-ASR+LoRA) |
|---|---|---|---|
| exact_match | 0.000 | 0.219 | 0.940 |
| latex_token_edit | 2.901 | 0.345 | 0.004 |
| symbol_acc | 0.001 | 0.830 | 0.999 |
| greek_acc | 0.000 | 0.837 | 0.997 |
| operator_acc | 0.002 | 0.897 | 1.000 |
| script_acc | 0.000 | 0.850 | 0.979 |
| bracket_acc | 0.000 | 0.941 | 1.000 |
| wer_vs_spoken | 0.073 | – | – |

## Test-hard — 9 grammar templates never seen in training, unseen voices

| metric | A. Qwen3-ASR zero-shot | C. A + rule normaliser | B. MathSpeech (Qwen3-ASR+LoRA) | Whisper v3-turbo zero-shot | Whisper + rules | Whisper + LoRA (v1) |
|---|---|---|---|---|---|---|
| exact_match | 0.000 | 0.556 | 0.481 | 0.000 | 0.426 | 0.111 |
| latex_token_edit | 2.652 | 0.191 | 0.202 | 2.674 | 0.318 | 0.368 |
| symbol_acc | 0.007 | 0.933 | 0.812 | 0.000 | 0.869 | 0.553 |
| greek_acc | 0.000 | 0.857 | 1.000 | 0.000 | 1.000 | 0.857 |
| operator_acc | 0.013 | 0.923 | 0.885 | 0.000 | 0.904 | 0.724 |
| script_acc | 0.000 | 0.583 | 1.000 | 0.000 | 0.347 | 0.778 |
| bracket_acc | 0.000 | 1.000 | 1.000 | 0.000 | 0.875 | 0.750 |
| wer_vs_spoken | 0.139 | – | – | 0.168 | – | – |

## Test-OOD — 64 hand-written expressions, natural phrasing, unseen voices

| metric | A. Qwen3-ASR zero-shot | C. A + rule normaliser | B. MathSpeech (Qwen3-ASR+LoRA) |
|---|---|---|---|
| exact_match | 0.000 | 0.406 | 0.797 |
| latex_token_edit | 2.968 | 0.420 | 0.044 |
| symbol_acc | 0.008 | 0.903 | 0.983 |
| greek_acc | 0.000 | 0.913 | 1.000 |
| operator_acc | 0.010 | 0.969 | 0.993 |
| script_acc | 0.000 | 0.867 | 0.962 |
| bracket_acc | 0.000 | 0.792 | 0.979 |
| wer_vs_spoken | 0.055 | – | – |

## Real human speech (evaluation only; AAAI2025/MathSpeech, n=1101)

| metric | zeroshot | rules | mathspeech |
|---|---|---|---|
| exact_match | 0.000 | 0.084 | 0.305 |
| latex_token_edit | 2.166 | 0.527 | 0.264 |
| symbol_acc | 0.000 | 0.744 | 0.854 |
| greek_acc | 0.000 | 0.880 | 0.892 |
| operator_acc | 0.000 | 0.800 | 0.957 |
| script_acc | 0.000 | 0.628 | 0.800 |
| bracket_acc | 0.000 | 0.347 | 0.432 |
| wer_vs_spoken | 0.182 | – | – |

## Training — Qwen3-ASR LoRA (final)

- model: `Qwen/Qwen3-ASR-1.7B-hf` + LoRA, target_mode=`canonical`
- steps: 2550, examples: 27204, best val loss: 0.0122
- GPU: NVIDIA H100 80GB HBM3, wall time: 25.5 min

See `artifacts/evaluation/report.html` for audio + typeset examples.
