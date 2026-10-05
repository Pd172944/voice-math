"""Hybrid canonicalizer: rules/parser first, optional LLM fallback, deterministic validity filters."""
import re
from .rules import tokenize, is_math, math_score
from .parser import Parser, tidy
from . import llm_normalizer

ALLOWED_CMDS = set(r"""frac int sum prod lim partial nabla sqrt pi alpha beta gamma delta epsilon varepsilon zeta eta theta iota kappa lambda mu nu xi rho
sigma tau upsilon phi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Phi Psi Omega infty in notin subset subseteq supset cup cap setminus
forall exists emptyset leq geq neq approx pm times cdot div to mapsto implies iff sim propto mid mathbb mathbf mathcal mathrm vec bar hat tilde
overline sin cos tan log ln exp arg min max sup inf det operatorname hbar ell langle rangle left right ldots text epsilon tanh Omega dagger""".split())


def balanced(s):
    for a, b in ("()", "[]", "{}"):
        d = 0
        for c in s:
            d += (c == a) - (c == b)
            if d < 0: return False
        if d: return False
    return True


def valid_latex(s):
    return bool(s) and balanced(s) and all(c in ALLOWED_CMDS for c in re.findall(r"\\([a-zA-Z]+)", s))


def canonicalize_rules(text):
    toks = tokenize(text)
    if not toks:
        return dict(latex="", confidence=0.0, method="rules", is_math=False)
    score = math_score(toks)
    if not is_math(toks):
        return dict(latex=text.strip(), confidence=0.0, method="passthrough", is_math=False, math_score=score)
    p = Parser(toks)
    try:
        out = tidy(p.parse_seq())
    except Exception as e:   # robustness: never crash the pipeline on odd ASR output
        return dict(latex=text.strip(), confidence=0.0, method="rules_error", is_math=True, math_score=score, error=repr(e))
    n_unknown = len(p.unknown)
    conf = max(0.0, 1.0 - n_unknown / max(1, len(toks)) * 3)
    if not valid_latex(out):
        conf *= 0.3
    return dict(latex=out, confidence=round(conf, 3), method="rules", is_math=True, math_score=score, unknown=p.unknown)


def canonicalize(text, use_llm=False, llm_threshold=0.7):
    """Return dict(latex, confidence, method, is_math, ...). LLM only used if enabled, key available, and rules are not confident."""
    r = canonicalize_rules(text)
    if use_llm and r["is_math"] and r["confidence"] < llm_threshold and llm_normalizer.available():
        l = llm_normalizer.llm_normalize(text)
        if l and valid_latex(l["latex"]):
            r.update(latex=l["latex"], confidence=min(l["confidence"], 0.9), method="llm", llm=l)
    return r


def canon(text):
    return canonicalize(text)["latex"]
