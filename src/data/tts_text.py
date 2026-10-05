"""Make spoken-math text safe for Kokoro TTS: explicit phonemes for single letters / Greek / jargon."""
import random, re

LETTER_PH = dict(a="ˈA", b="bˈi", c="sˈi", d="dˈi", e="ˈi", f="ˈɛf", g="ʤˈi", h="ˈAʧ", i="ˈI", j="ʤˈA", k="kˈA", l="ˈɛl", m="ˈɛm", n="ˈɛn",
                 o="ˈO", p="pˈi", q="kjˈu", r="ˈɑɹ", s="ˈɛs", t="tˈi", u="jˈu", v="vˈi", w="dˈʌbᵊlju", x="ˈɛks", y="wˈI", z="zˈi")
WORD_PH = dict(xi="ksˈI", phi="fˈI", chi="kˈI", psi="sˈI", tau="tˈW", rho="ɹˈO", nu="nˈu", mu="mjˈu", eta="ˈAtə", zeta="zˈAtə",
               theta="θˈAtə", kappa="kˈæpə", omega="OmˈɛɡƏ", sin="sˈI", cos="kˈɑs", tanh="tˈæn ˈAʧ", relu="ɹˈɛlju", hbar="ˈAʧ bˈɑɹ")
REWRITE = {"sin": "sine", "cos": "cosine", "ph": "p H", "kl": "k l", "relu": "re lu", "tanh": "tan h"}


def to_tts(text, rng=None, pauses=True):
    rng = rng or random.Random(0)
    out = []
    for tok in text.split():
        low = tok.lower()
        if len(tok) == 1 and low in LETTER_PH:
            out.append(f"[{tok}](/{LETTER_PH[low]}/)")
        elif low in ("pH",) or tok == "pH":
            out.append("[p](/pˈi/) [H](/ˈAʧ/)")
        elif low in ("kl",):
            out.append("[k](/kˈA/) [l](/ˈɛl/)")
        elif low in ("relu",):
            out.append("[ReLU](/ɹˈɛlju/)")
        elif low in ("tanh",):
            out.append("[tanh](/tˈæn ˈAʧ/)")
        elif low in WORD_PH and low not in ("sin", "cos"):
            out.append(f"[{tok}](/{WORD_PH[low]}/)")
        else:
            out.append(REWRITE.get(low, tok) if low in ("sin", "cos") else tok)
    s = " ".join(out)
    if pauses:  # occasional commas -> short pauses at natural boundaries
        s = re.sub(r" (equals|is equal to) ", lambda m: (", " if rng.random() < 0.25 else " ") + m.group(1) + " ", s)
        s = re.sub(r" (plus|minus) ", lambda m: (", " if rng.random() < 0.1 else " ") + m.group(1) + " ", s)
    return s
