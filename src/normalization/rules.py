"""Pre-processing and context rules: tokenisation, math-context detection, unicode handling."""
import re
from .lexicon import MATH_TRIGGERS, NUMBERS, GREEK

UNICODE = {"²": " squared ", "³": " cubed ", "π": " pi ", "∞": " infinity ", "∫": " integral ", "∑": " sum ", "∂": " partial ",
           "≥": " greater than or equal to ", "≤": " less than or equal to ", "≈": " approximately ", "∈": " in ", "∇": " nabla ",
           "θ": " theta ", "λ": " lambda ", "σ": " sigma ", "μ": " mu ", "ε": " epsilon ", "δ": " delta ", "α": " alpha ", "β": " beta ",
           "γ": " gamma ", "ω": " omega ", "φ": " phi ", "ρ": " rho ", "τ": " tau ", "∀": " for all ", "∃": " there exists ",
           "×": " times ", "÷": " divided by ", "−": " minus ", "√": " square root of "}
CONTRACT = [(r"\bx\^2\b", "x squared"), (r"\b(\w)\^2\b", r"\1 squared"), (r"\b(\w)\^3\b", r"\1 cubed"), (r"\b(\w)\^(\w)\b", r"\1 to the \2"),
            (r"\b(\w)_(\w)\b", r"\1 sub \2"), (r"(?<=\d)\s*/\s*(?=\d)", " over "), (r"(?<=\w)\s*\+\s*(?=\w)", " plus "),
            (r"(?<=\w)\s*=\s*(?=\w)", " equals "), (r"\bd([a-zA-Z])\b", r"d \1"), (r"\bepsilon[- ]delta\b", "epsilon delta")]


def tokenize(text):
    t = text
    for k, v in UNICODE.items():
        t = t.replace(k, v)
    t = t.replace("$", " ").replace("\\(", " ").replace("\\)", " ")
    for pat, rep in CONTRACT:
        t = re.sub(pat, rep, t)
    t = re.sub(r"(?<=\d)\.(?=\d)", "<DOT>", t)          # protect decimals
    t = re.sub(r"[,;:?!\"“”()]", " ", t)
    t = re.sub(r"\.(\s|$)", r" \1", t)
    t = t.replace("<DOT>", ".")
    t = re.sub(r"(\w)'s\b", r"\1 s", t)
    toks = []
    for w in t.split():
        w = w.strip(".")
        if not w:
            continue
        if re.fullmatch(r"[a-zA-Z]+-[a-zA-Z]+", w) and w.split("-")[0].lower() in ("p", "two", "l", "one", "infinity"):
            toks += w.split("-")
        elif re.fullmatch(r"-\d+(\.\d+)?", w):
            toks += ["minus", w[1:]]
        else:
            toks.append(w)
    return toks


def is_number_token(w):
    return bool(re.fullmatch(r"\d+(\.\d+)?", w)) or w.lower() in NUMBERS


def math_score(tokens):
    """Heuristic math-context score. 'pie'/'pi' only count when other math evidence exists."""
    low = [w.lower() for w in tokens]
    trig = sum(1 for w in low if w in MATH_TRIGGERS and w != "pi")
    pi = sum(1 for w in low if w in ("pi", "pie"))
    nums = sum(1 for w in low if is_number_token(w))
    letters = sum(1 for w in tokens if len(w) == 1 and w.isalpha() and w.lower() not in ("a", "i"))
    score = trig + 0.5 * nums + 0.5 * letters
    txt = " ".join(low)
    if re.search(r"(^| )[rnzqc] [nmdk]( |$)", txt) and any(w in "RNZQC" and len(w) == 1 for w in tokens):
        score += 1
    if "for all" in txt or "there exists" in txt or "is in" in txt or "is not in" in txt:
        score += 1
    if pi and (trig or nums):
        score += 1 + pi
    elif pi and any(w == "pi" for w in low):
        score += 0.5
    return score


def is_math(tokens, thresh=1.5):
    return math_score(tokens) >= thresh
