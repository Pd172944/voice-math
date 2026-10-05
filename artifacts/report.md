# MathSpeech – experiment results


## Test (unseen expressions, partly unseen voices)

| metric | zeroshot | rules | canonical | spoken_ft_rules |
|---|---|---|---|---|
| exact_match | 0.000 | 0.503 | 0.912 | 0.901 |
| latex_token_edit | 2.729 | 0.420 | 0.012 | 0.068 |
| symbol_acc | 0.001 | 0.941 | 0.989 | 0.992 |
| greek_acc | 0.000 | 0.943 | 0.977 | 0.989 |
| operator_acc | 0.001 | 0.978 | 0.997 | 0.999 |
| script_acc | 0.000 | 0.901 | 0.987 | 0.986 |
| bracket_acc | 0.000 | 0.750 | 1.000 | 0.986 |
| wer_vs_spoken | 0.117 | – | – | 0.003 |

| domain | n | generic | generic+rules | MathSpeech |
|---|---|---|---|---|
| algebra | 171 | 0.00 | 0.41 | 0.90 |
| analysis | 8 | 0.00 | 0.62 | 1.00 |
| calculus | 182 | 0.00 | 0.68 | 0.91 |
| linear_algebra | 20 | 0.00 | 0.00 | 0.95 |
| ml | 4 | 0.00 | 0.00 | 0.50 |
| physics | 8 | 0.00 | 0.38 | 0.75 |
| probability | 10 | 0.00 | 0.40 | 1.00 |
| set_theory | 30 | 0.00 | 0.40 | 1.00 |

## Hard test (unseen grammar templates, unseen voices)

| metric | zeroshot | rules | canonical | spoken_ft_rules |
|---|---|---|---|---|
| exact_match | 0.000 | 0.426 | 0.111 | 0.444 |
| latex_token_edit | 2.674 | 0.318 | 0.368 | 0.192 |
| symbol_acc | 0.000 | 0.869 | 0.553 | 0.894 |
| greek_acc | 0.000 | 1.000 | 0.857 | 1.000 |
| operator_acc | 0.000 | 0.904 | 0.724 | 0.962 |
| script_acc | 0.000 | 0.347 | 0.778 | 0.583 |
| bracket_acc | 0.000 | 0.875 | 0.750 | 1.000 |
| wer_vs_spoken | 0.168 | – | – | 0.000 |

| domain | n | generic | generic+rules | MathSpeech |
|---|---|---|---|---|
| algebra | 6 | 0.00 | 0.00 | 0.00 |
| analysis | 6 | 0.00 | 1.00 | 0.00 |
| calculus | 6 | 0.00 | 0.33 | 1.00 |
| chemistry | 6 | 0.00 | 0.00 | 0.00 |
| linear_algebra | 6 | 0.00 | 1.00 | 0.00 |
| ml | 6 | 0.00 | 0.17 | 0.00 |
| physics | 6 | 0.00 | 0.00 | 0.00 |
| probability | 6 | 0.00 | 0.50 | 0.00 |
| set_theory | 6 | 0.00 | 0.83 | 0.00 |

## Training

- model: `openai/whisper-large-v3-turbo` + LoRA, target_mode=`canonical`
- steps: 1864, examples: 14917, best val loss: 0.0324
- GPU: NVIDIA H100 80GB HBM3, wall time: 39.0 min

See `artifacts/evaluation/report.html` for audio + rendered examples.

## Real human speech probe (evaluation only; AAAI2025/MathSpeech, n=1101)

| | generic | generic+rules | MathSpeech |
|---|---|---|---|
| token edit | 2.010 | 0.654 | 0.357 |
| script acc | 0.000 | 0.581 | 0.750 |
| symbol acc | 0.011 | 0.713 | 0.686 |

## Ablation log

| run | change | test exact | test_hard exact |
|---|---|---|---|
| v1 | LoRA on attn+MLP only, 3 ep, default Whisper token suppression lifted | 0.792 | 0.111 |
| v3 (final) | + LoRA on proj_out, 4 ep, lr 1.5e-4 | 0.912 | 0.111 |
| D | spoken-target LoRA (2 ep) + rule normalizer | 0.901 | 0.444 |
