"""Diagnostic metrics: WER, normalized exact match, symbol/Greek/operator/bracket/script accuracy, LaTeX token edit distance."""
import re
from collections import Counter

TOKEN_RE = re.compile(r"\\[a-zA-Z]+|\\.|[A-Za-z]|\d+(?:\.\d+)?|[^\sA-Za-z\d]")
GREEK_RE = re.compile(r"\\(?:var)?(?:alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Phi|Psi|Omega)(?![a-zA-Z])")
OP_RE = re.compile(r"\\(?:int|sum|prod|lim|partial|nabla|leq|geq|neq|approx|pm|times|cdot|in|notin|subset|subseteq|cup|cap|to|sim|forall|exists|infty|frac|sqrt)(?![a-zA-Z])|[=+\-<>]")


def norm_latex(s):
    """Canonical comparison form: equivalent spellings map to the same string."""
    s = s or ""
    s = re.sub(r"\\(?:left|right|,|;|!|:|quad|qquad|displaystyle)(?![a-zA-Z])", "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\epsilon", "\\epsilon").replace("\\varepsilon", "\\epsilon")
    s = s.replace("\\mathrm", "\\mathrm").replace("\\text", "\\mathrm")
    s = re.sub(r"\\frac\s*(\d)\s*(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac\s*(\d)\s*\{", r"\\frac{\1}{", s)
    s = re.sub(r"(?<![\w}])(\d+)\s*/\s*(\d+)(?!\w)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\(sqrt|hat|bar|vec|tilde|overline|mathbb|mathbf|mathcal|mathrm)\s*([A-Za-z0-9])", r"\\\1{\2}", s)
    s = re.sub(r"([\^_])\s*(\\[a-zA-Z]+|[A-Za-z0-9])", r"\1{\2}", s)      # x^2 == x^{2}
    s = re.sub(r"\{\{([^{}]*)\}\}", r"{\1}", s)
    s = s.replace("\\operatorname{", "\\mathrm{").replace("\\mid", "|")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\\mathrm\{([^{}]*)\}", r"\\mathrm{\1}", s)
    s = s.strip("$")
    return s


def tokens(s):
    return TOKEN_RE.findall(norm_latex(s))


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def token_edit_distance(hyp, ref):
    h, r = tokens(hyp), tokens(ref)
    return edit_distance(h, r) / max(1, len(r))


def _recall(hyp_items, ref_items):
    """Multiset recall of reference items found in hypothesis. Returns (hit, total)."""
    c = Counter(hyp_items); hit = 0
    for k, v in Counter(ref_items).items():
        hit += min(v, c[k])
    return hit, len(ref_items)


def _acc(pattern_fn, hyp, ref):
    return _recall(pattern_fn(norm_latex(hyp)), pattern_fn(norm_latex(ref)))


def scripts(s):
    return re.findall(r"[\^_](?:\{[^{}]*\}|\\[a-zA-Z]+|[A-Za-z0-9])", s)


def brackets(s):
    return re.findall(r"[()\[\]]|\\\{|\\\}|\\\|", s)


def symbols(s):  # all non-alnum latex commands + literal operators
    return re.findall(r"\\[a-zA-Z]+|[=+\-<>|!]", s)


def wer(hyp, ref):
    from jiwer import wer as _w
    n = lambda x: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", (x or "").lower())).strip()
    return _w(n(ref) or "x", n(hyp))


def score_example(hyp, ref, spoken_ref=None):
    out = dict(exact=float(norm_latex(hyp) == norm_latex(ref)), tok_edit=token_edit_distance(hyp, ref))
    for name, fn in [("symbol", symbols), ("greek", lambda s: GREEK_RE.findall(s)), ("operator", lambda s: OP_RE.findall(s)),
                     ("script", scripts), ("bracket", brackets)]:
        h, t = _acc(fn, hyp, ref)
        out[name] = (h, t)
    return out


def aggregate(rows):
    """rows: list of score_example dicts -> dataset-level metrics."""
    agg = dict(n=len(rows), exact_match=sum(r["exact"] for r in rows) / max(1, len(rows)),
               latex_token_edit=sum(r["tok_edit"] for r in rows) / max(1, len(rows)))
    for k in ("symbol", "greek", "operator", "script", "bracket"):
        h = sum(r[k][0] for r in rows); t = sum(r[k][1] for r in rows)
        agg[f"{k}_acc"] = (h / t) if t else None
        agg[f"{k}_support"] = t
    return agg
