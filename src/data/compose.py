"""Compositional expression generator: random *trees* of math constructs (not fixed templates), each node emitting LaTeX and a
matching, unambiguous spoken form from the same draw. Gives unbounded structural variety so a model must learn composition.

Node = X(tex, spk, prec, sub_end, num): prec 0 atom, 1 factor (power/func/frac/...), 2 product, 3 sum.
Compound arguments are disambiguated in speech with "the quantity ... end quantity", "open paren ... close paren" or
"the fraction ... over ... end fraction", as a careful speaker would."""
import random
from dataclasses import dataclass

from .grammar import say_int, say_num, GREEK


@dataclass
class X:
    tex: str
    spk: str
    prec: int = 0
    sub_end: bool = False   # spoken form ends in "sub <idx>" (juxtaposing a letter after it is ambiguous)
    num: bool = False
    open_start: bool = False   # spoken form can swallow a preceding factor (e.g. 'x over y')
    open_end: bool = False     # spoken form can swallow a following factor (e.g. 'the square root of x ...')


VARS = list("xyztnkmuvwabcfghpqrs")
IDXS = list("ijknm")
CAPS = list("ABCDEFGHMNPQRSTUVWXYZ")
GREEKS = ["alpha", "beta", "gamma", "delta", "epsilon", "theta", "lambda", "mu", "nu", "rho", "sigma", "tau", "phi", "psi", "omega", "eta", "kappa", "xi", "zeta"]
OPEN_CLOSE = [("open paren", "close paren"), ("left paren", "right paren"), ("open parenthesis", "close parenthesis")]
EQW = ["equals", "is equal to", "equals", "equals"]


def pick(r, xs):
    return xs[r.randrange(len(xs))]


def num(r, lo=1, hi=9):
    k = r.random()
    if k < 0.1:
        d = pick(r, ["0.5", "0.1", "2.5", "3.14", "1.5", "0.01"])
        return X(d, say_num(d), 0, num=True)
    n = r.choice(list(range(lo, hi + 1)) + ([12, 16, 20, 100] if hi >= 9 else []))
    return X(str(n), say_int(n), 0, num=True)


def var(r, pool=None):
    v = pick(r, pool or VARS)
    return X(v, v)


def greek(r):
    g = pick(r, GREEKS)
    return X("\\" + g, g)


def sub(r, base=None):
    b = base or (var(r) if r.random() < 0.7 else greek(r))
    k = r.random()
    if k < 0.6:
        i = pick(r, IDXS)
        return X(f"{b.tex}_{i}", f"{b.spk} sub {i}", 0, sub_end=True)
    if k < 0.75:
        n = pick(r, [0, 1, 2])
        return X(f"{b.tex}_{n}", f"{b.spk} sub {say_int(n)}", 0, sub_end=True)
    if k < 0.9:
        i, j = r.sample(IDXS, 2)
        return X(f"{b.tex}_{{{i},{j}}}", f"{b.spk} sub {i} {j}", 0, sub_end=True)
    i = pick(r, IDXS)
    s = pick(r, [("+1", "plus one"), ("-1", "minus one")])
    return X(f"{b.tex}_{{{i}{s[0]}}}", f"{b.spk} sub {i} {s[1]}", 0, sub_end=True)


def atom(r):
    k = r.random()
    if k < 0.45:
        return var(r)
    if k < 0.65:
        return num(r)
    if k < 0.8:
        return greek(r)
    return sub(r)


def quantity(r, e):
    """Spoken grouping for a compound argument."""
    if r.random() < 0.6:
        return f"the quantity {e.spk} end quantity"
    o, c = pick(r, OPEN_CLOSE)
    return f"{o} {e.spk} {c}"


def arg_spk(r, e):
    return e.spk if e.prec <= 1 else quantity(r, e)


def paren(r, e):
    o, c = pick(r, OPEN_CLOSE)
    return X(f"({e.tex})", f"{o} {e.spk} {c}", 0)


def power(r, d):
    base = atom(r) if r.random() < 0.7 or d <= 1 else paren(r, sum_(r, d - 1, 2))
    k = r.random()
    if k < 0.3:
        return X(f"{base.tex}^2", f"{base.spk} " + pick(r, ["squared", "squared", "to the second"]), 1, sub_end=False)
    if k < 0.42:
        return X(f"{base.tex}^3", f"{base.spk} cubed", 1)
    if k < 0.62:
        e = var(r, list("nkmpq")) if r.random() < 0.8 else greek(r)
        return X(f"{base.tex}^{e.tex if len(e.tex) == 1 or e.tex.startswith(chr(92)) else '{' + e.tex + '}'}", f"{base.spk} " + pick(r, ["to the", "to the power of", "raised to the"]) + f" {e.spk}", 1)
    if k < 0.74:
        return X(f"{base.tex}^{{-1}}", f"{base.spk} " + pick(r, ["inverse", "to the minus one"]), 1)
    if k < 0.84:
        n = r.choice([4, 5, 6]); return X(f"{base.tex}^{n}", f"{base.spk} to the {['', '', '', '', 'fourth', 'fifth', 'sixth'][n]}", 1)
    if k < 0.92:
        v = var(r, list("nkm")); c = pick(r, [("+1", "plus one"), ("-1", "minus one"), ("+2", "plus two")])
        return X(f"{base.tex}^{{{v.tex}{c[0]}}}", f"{base.spk} to the quantity {v.spk} {c[1]} end quantity", 1)
    v = var(r, list("xnk"))
    return X(f"{base.tex}^{{-{v.tex}}}", f"{base.spk} to the minus {v.spk}", 1)


FUNCS = [("\\sin", "sine"), ("\\cos", "cosine"), ("\\tan", "tangent"), ("\\log", "log"), ("\\ln", "natural log"), ("\\exp", "exponential"), ("\\arctan", "arc tangent")]


def _func(r, d):
    k = r.random()
    if k < 0.45:
        t, s = pick(r, FUNCS)
        if t in ("\\sin", "\\cos", "\\tan") and r.random() < 0.5:
            s = {"\\sin": "sin", "\\cos": "cos", "\\tan": "tan"}[t]
        a = atom(r) if (r.random() < 0.6 or d <= 1) else factor(r, d - 1)
        if r.random() < 0.15 and t in ("\\sin", "\\cos", "\\tan") and a.prec == 0:      # sin^2 x
            return X(f"{t}^2 {a.tex}", f"{s} squared {a.spk}", 1)
        if a.prec == 0 and r.random() < 0.4:
            return X(f"{t} {a.tex}", f"{s} {a.spk}", 1)
        return X(f"{t}({a.tex})", f"{s} of {a.spk}", 1)
    if k < 0.75:
        f = pick(r, list("fghFG")); a = atom(r)
        if r.random() < 0.25:
            b = atom(r)
            return X(f"{f}({a.tex}, {b.tex})", f"{f} of {a.spk} and {b.spk}", 1)
        if r.random() < 0.2 and d > 1:
            g = pick(r, [c for c in "fgh" if c != f]); b = atom(r)
            return X(f"{f}({g}({b.tex}))", f"{f} of {g} of {b.spk}", 1)
        return X(f"{f}({a.tex})", f"{f} of {a.spk}", 1)
    if k < 0.9:
        a = var(r, list("xt")); e = pick(r, [("e", "e")])
        s = r.random()
        if s < 0.5: return X(f"e^{a.tex}", f"e to the {a.spk}", 1)
        if s < 0.8: return X(f"e^{{-{a.tex}}}", f"e to the minus {a.spk}", 1)
        return X(f"e^{{{a.tex}^2}}", f"e to the {a.spk} squared", 1)
    f = pick(r, list("fg")); a = var(r, list("xt"))
    return X(f"{f}'({a.tex})", f"{f} prime of {a.spk}", 1)


def _frac(r, d):
    if r.random() < 0.15:
        a, b = pick(r, [(1, 2), (1, 3), (2, 3), (3, 4), (1, 4)]);
        w = {(1, 2): "one half", (1, 3): "one third", (2, 3): "two thirds", (3, 4): "three fourths", (1, 4): "one fourth"}[(a, b)]
        return X(f"\\frac{{{a}}}{{{b}}}", w, 1)
    n = sum_(r, d - 1, r.choice([1, 1, 2])) if d > 1 else atom(r)
    m = sum_(r, d - 1, r.choice([1, 1, 2])) if d > 1 else atom(r)
    if n.prec <= 1 and m.prec <= 1:
        return X(f"\\frac{{{n.tex}}}{{{m.tex}}}", f"{n.spk} over {m.spk}", 1)
    return X(f"\\frac{{{n.tex}}}{{{m.tex}}}", f"the fraction {n.spk} over {m.spk} end fraction", 1)


def _sqrt(r, d):
    a = (atom(r) if r.random() < 0.5 or d <= 1 else power(r, d - 1)) if r.random() < 0.6 else sum_(r, max(1, d - 1), 2)
    if r.random() < 0.12:
        n = pick(r, [(3, "cube"), (4, "fourth")]); return X(f"\\sqrt[{n[0]}]{{{a.tex}}}", f"the {n[1]} root of {arg_spk(r, a)}", 1)
    return X(f"\\sqrt{{{a.tex}}}", f"the square root of {arg_spk(r, a)}", 1)


def _abs(r, d):
    a = atom(r) if r.random() < 0.6 or d <= 1 else sum_(r, d - 1, 2)
    return X(f"|{a.tex}|", f"the absolute value of {arg_spk(r, a)}", 1)


def accent(r, d):
    k = r.random()
    if k < 0.25:
        b = pick(r, VARS + ["\\theta", "\\mu", "\\sigma", "\\beta"]); s = b.lstrip("\\")
        w = pick(r, [("hat", "\\hat"), ("bar", "\\bar"), ("tilde", "\\tilde")])
        return X(f"{w[1]}{{{b}}}", f"{s} {w[0]}", 0)
    if k < 0.45:
        b = pick(r, VARS); return X(f"\\vec{{{b}}}", f"vector {b}", 0)
    if k < 0.7:
        b = pick(r, list("xyzvwuabAWXb")); return X(f"\\mathbf{{{b}}}", f"bold {b}", 0)
    if k < 0.8:
        b = pick(r, list("LHFNDSPT")); return X(f"\\mathcal{{{b}}}", f"script {b}", 0)
    if k < 0.9:
        a = pick(r, list("xyvw")); return X(f"\\mathbf{{{a}}}^T", f"bold {a} transpose", 1)
    A = pick(r, list("ABMWQ"))
    return X(f"{A}^T", f"{A} transpose", 1)


def _binom(r, d):
    n, k = pick(r, list("nmN")), pick(r, list("kjr"))
    return X(f"\\binom{{{n}}}{{{k}}}", f"{n} choose {k}", 1)


def func(r, d):
    e = _func(r, d)
    e.open_end = not e.tex.startswith("e^")
    return e


def frac(r, d):
    e = _frac(r, d)
    if " over " in e.spk and not e.spk.startswith("the fraction"):
        e.open_start = e.open_end = True
    return e


def sqrt_(r, d):
    e = _sqrt(r, d); e.open_end = True; return e


def abs_(r, d):
    e = _abs(r, d); e.open_end = True; return e


def binom(r, d):
    e = _binom(r, d); e.open_start = e.open_end = True; return e


def factor(r, d=2):
    if d <= 0:
        return atom(r)
    k = r.random()
    if k < 0.2: return atom(r)
    if k < 0.38: return power(r, d)
    if k < 0.52: return func(r, d)
    if k < 0.64: return frac(r, d)
    if k < 0.72: return sqrt_(r, d)
    if k < 0.78: return abs_(r, d)
    if k < 0.86: return accent(r, d)
    if k < 0.98: return paren(r, sum_(r, max(1, d - 1), 2))
    return binom(r, d)


def prod(r, d, n=None):
    n = n or r.choice([1, 1, 2, 2, 3])
    fs = [factor(r, d) for _ in range(n)]
    if n == 1:
        return fs[0]
    tex, spk = fs[0].tex, fs[0].spk
    prev = fs[0]
    for f in fs[1:]:
        explicit = prev.sub_end or prev.open_end or f.open_start or f.num or r.random() < 0.1
        if explicit:
            tex += f" \\cdot {f.tex}"; spk += f" times {f.spk}"
        else:
            tex += f" {f.tex}"; spk += f" {f.spk}"
        prev = f
    return X(tex, spk, 2, sub_end=prev.sub_end, open_end=prev.open_end, num=fs[0].num and False)


def sum_(r, d, n=None):
    n = n or r.choice([2, 2, 3])
    ts = [prod(r, d - 1 if d > 1 else 0, r.choice([1, 1, 2])) for _ in range(n)]
    tex, spk = ts[0].tex, ts[0].spk
    if r.random() < 0.08:
        tex, spk = "-" + tex, "minus " + spk
    for t in ts[1:]:
        if r.random() < 0.6: tex += f" + {t.tex}"; spk += f" plus {t.spk}"
        else: tex += f" - {t.tex}"; spk += f" minus {t.spk}"
    return X(tex, spk, 3 if n > 1 else ts[0].prec)


RELS = [("=", EQW), ("<", ["is less than", "less than"]), (">", ["is greater than", "greater than"]), ("\\leq", ["is less than or equal to", "less than or equal to"]),
        ("\\geq", ["is greater than or equal to", "greater than or equal to"]), ("\\neq", ["is not equal to", "does not equal"]), ("\\approx", ["is approximately", "approximately equal to"])]


def rel(r, a, b, kind=None):
    k = kind or (RELS[0] if r.random() < 0.6 else pick(r, RELS))
    return X(f"{a.tex} {k[0]} {b.tex}", f"{a.spk} {pick(r, k[1])} {b.spk}", 4)


# ----------------------------------------------------------------------------- big operators (always last in an expression)
BOUNDS_LO = [("0", "zero"), ("0", "zero"), ("1", "one"), ("-1", "minus one"), ("a", "a"), ("-\\infty", "minus infinity"), ("-\\pi", "minus pi"), ("-a", "minus a")]
BOUNDS_HI = [("1", "one"), ("\\pi", "pi"), ("\\infty", "infinity"), ("b", "b"), ("2", "two"), ("2 \\pi", "two pi"), ("T", "T"), ("L", "L"), ("x", "x")]


def sc(b):
    return b if (len(b) == 1 or (b.startswith("\\") and b[1:].isalpha())) else "{" + b + "}"


def body(r, d, allow_sum=False):
    b = sum_(r, d, 2) if allow_sum else prod(r, d, r.choice([1, 1, 2]))
    if not allow_sum and b.prec == 3:
        b = paren(r, b)
    return b


def integral(r, d):
    lo, hi = pick(r, BOUNDS_LO), pick(r, BOUNDS_HI)
    v = pick(r, list("xtsu")); b = body(r, max(1, d - 1), allow_sum=True)
    k = r.random()
    if k < 0.15:
        return X(f"\\int {b.tex}\\,d{v}", f"the integral of {b.spk} d {v}", 3)
    if k < 0.25:
        D = pick(r, list("DRSV")); w = pick(r, list("AV"))
        return X(f"\\iint_{D} {b.tex}\\,d{w}", f"the double integral over {D} of {b.spk} d {w}", 3)
    return X(f"\\int_{sc(lo[0])}^{sc(hi[0])} {b.tex}\\,d{v}", f"the integral from {lo[1]} to {hi[1]} of {b.spk} d {v}", 3)


def sumop(r, d):
    i = pick(r, list("ijk")); lo = pick(r, [("1", "one"), ("0", "zero")]); hi = pick(r, [("n", "n"), ("N", "N"), ("\\infty", "infinity"), ("m", "m")])
    b = body(r, max(1, d - 1))
    if r.random() < 0.2:
        return X(f"\\prod_{{{i}={lo[0]}}}^{sc(hi[0])} {b.tex}", f"the product from {i} equals {lo[1]} to {hi[1]} of {b.spk}", 3)
    if r.random() < 0.12:
        return X(f"\\sum_{i} {b.tex}", f"the sum over {i} of {b.spk}", 3)
    return X(f"\\sum_{{{i}={lo[0]}}}^{sc(hi[0])} {b.tex}", f"the sum from {i} equals {lo[1]} to {hi[1]} of {b.spk}", 3)


def limit(r, d):
    v = pick(r, list("xnth")); a = pick(r, [("0", "zero"), ("1", "one"), ("\\infty", "infinity"), ("\\pi", "pi"), ("-\\infty", "minus infinity"), ("a", "a")])
    b = body(r, max(1, d - 1))
    return X(f"\\lim_{{{v} \\to {a[0]}}} {b.tex}", f"the limit as {v} goes to {a[1]} of {b.spk}", 3)


def deriv(r, d):
    v = pick(r, list("xtsz")); k = r.random()
    b = factor(r, max(1, d - 1))
    if k < 0.5:
        return X(f"\\frac{{d}}{{d{v}}} {b.tex}", f"the derivative of {b.spk} with respect to {v}", 3)
    if k < 0.75:
        return X(f"\\frac{{\\partial}}{{\\partial {v}}} {b.tex}", f"the partial derivative of {b.spk} with respect to {v}", 3)
    if k < 0.9:
        return X(f"\\frac{{d^2}}{{d{v}^2}} {b.tex}", f"the second derivative of {b.spk} with respect to {v}", 3)
    return X(f"\\nabla {b.tex}" if b.prec == 0 else f"\\nabla ({b.tex})", f"nabla {b.spk}" if b.prec == 0 else f"nabla of {b.spk}", 3)


def bigop(r, d):
    return pick(r, [integral, integral, sumop, sumop, limit, deriv])(r, d)


# ----------------------------------------------------------------------------- domain roots
def root_algebra(r, d):
    return rel(r, sum_(r, d, 2), sum_(r, d, r.choice([1, 2])))


def root_calculus(r, d):
    big = bigop(r, d)
    if r.random() < 0.5:
        return big
    return rel(r, big, sum_(r, d - 1, 2) if r.random() < 0.7 else prod(r, d - 1))


def root_linalg(r, d):
    k = r.random(); A = pick(r, list("ABMWQ")); v = pick(r, list("xyvwub"))
    if k < 0.15: return rel(r, X(f"{A}\\mathbf{{{v}}}", f"{A} bold {v}"), X(f"\\lambda \\mathbf{{{v}}}", f"lambda bold {v}"))
    if k < 0.3:
        B = pick(r, [c for c in "ABCM" if c != A]); return rel(r, X(f"({A} + {B})^T", f"open paren {A} plus {B} close paren transpose"), X(f"{A}^T + {B}^T", f"{A} transpose plus {B} transpose"))
    if k < 0.45:
        B = pick(r, [c for c in "ABCM" if c != A])
        return rel(r, X(f"\\operatorname{{tr}}({A} {B})", f"the trace of {A} {B}"), X(f"\\operatorname{{tr}}({B} {A})", f"the trace of {B} {A}")) if r.random() < 0.5 else rel(r, X(f"\\det({A}^T)", f"the determinant of {A} transpose"), X(f"\\det({A})", f"the determinant of {A}"))
    if k < 0.6:
        a, b, c, e = r.sample(list("abcdwxyz"), 4)
        return rel(r, X(f"{A}", f"{A}"), X(f"\\begin{{pmatrix}} {a} & {b} \\\\ {c} & {e} \\end{{pmatrix}}", f"the two by two matrix with rows {a} {b} and {c} {e}"))
    if k < 0.7:
        a, b = r.sample(list("xyzabuv"), 2); return X(f"\\begin{{pmatrix}} {a} \\\\ {b} \\end{{pmatrix}}", f"the column vector {a} {b}", 1)
    if k < 0.85:
        p = pick(r, [("1", "one"), ("2", "two"), ("p", "p"), ("\\infty", "infinity")]); b = pick(r, list("xyvw"))
        return rel(r, X(f"\\|\\mathbf{{{b}}}\\|_{sc(p[0])}", f"the {p[1]} norm of bold {b}"), sum_(r, d - 1, 2), pick(r, [RELS[0], RELS[3], RELS[4]]))
    return rel(r, X(f"\\operatorname{{tr}}({A}{pick(r, 'BC')})", f"the trace of {A} {pick(r, 'BC')}") if False else X(f"\\operatorname{{tr}}({A})", f"the trace of {A}"), sum_(r, d - 1, 2))


def root_prob(r, d):
    k = r.random(); V = pick(r, list("XYZ")); x = V.lower()
    A, B = r.sample(list("ABCE"), 2)
    if k < 0.2:
        return X(f"P({A} \\cup {B}) \\leq P({A}) + P({B})", f"the probability of {A} union {B} is less than or equal to the probability of {A} plus the probability of {B}", 4)
    if k < 0.35:
        return rel(r, X(f"P({A} \\mid {B})", f"the probability of {A} given {B}"),
                   X(f"\\frac{{P({B} \\mid {A}) P({A})}}{{P({B})}}", f"the fraction the probability of {B} given {A} times the probability of {A} over the probability of {B} end fraction"))
    if k < 0.5:
        f = factor(r, max(1, d - 1))
        return rel(r, X(f"E[{f.tex}]", f"the expectation of {f.spk}"), X(f"\\sum_i p_i {f.tex}", f"the sum over i of p sub i times {f.spk}"))
    if k < 0.65:
        a = num(r, 2, 9)
        return rel(r, X(f"\\operatorname{{Var}}({a.tex} {V})", f"the variance of {a.spk} {V}"), X(f"{a.tex}^2 \\operatorname{{Var}}({V})", f"{a.spk} squared times the variance of {V}"))
    if k < 0.8:
        mu = pick(r, [("\\mu", "mu"), ("0", "zero"), ("\\mu_1", "mu sub one")]); sg = pick(r, [("\\sigma^2", "sigma squared"), ("1", "one"), ("\\sigma_1^2", "sigma sub one squared")])
        return X(f"{V} \\sim \\mathcal{{N}}({mu[0]}, {sg[0]})", f"{V} is normally distributed with mean {mu[1]} and variance {sg[1]}", 4)
    if k < 0.9:
        return X(f"P({V} = {x}) \\geq 0", f"the probability that {V} equals {x} is greater than or equal to zero", 4)
    n = pick(r, "nm"); c = pick(r, "kxs")
    return rel(r, X("\\hat{p}", "p hat"), X(f"\\frac{{{c}}}{{{n}}}", f"{c} over {n}", 1))


def root_sets(r, d):
    k = r.random(); A, B, C = r.sample(list("ABCSTUXY"), 3); x = pick(r, list("xyabn"))
    S = pick(r, [("\\mathbb{R}", "R"), ("\\mathbb{N}", "N"), ("\\mathbb{Z}", "Z"), ("\\mathbb{Q}", "Q"), ("\\mathbb{C}", "C")])
    if k < 0.2:
        cond = rel(r, factor(r, 1), num(r), pick(r, RELS[1:5]))
        return X(f"\\forall {x} \\in {S[0]}, {cond.tex}", f"for all {x} in {S[1]} {cond.spk}", 4)
    if k < 0.35:
        cond = rel(r, X(f"{x}^2", f"{x} squared"), num(r))
        return X(f"\\exists {x} \\in {S[0]}, {cond.tex}", f"there exists {x} in {S[1]} {cond.spk}", 4)
    if k < 0.5:
        return rel(r, X(f"{A} \\cup ({B} \\cap {C})", f"{A} union open paren {B} intersection {C} close paren"),
                   X(f"({A} \\cup {B}) \\cap ({A} \\cup {C})", f"open paren {A} union {B} close paren intersection open paren {A} union {C} close paren"))
    if k < 0.6:
        i = pick(r, "ij"); n = pick(r, "nN")
        return X(f"\\bigcup_{{{i}=1}}^{{{n}}} {A}_{i}", f"the union over {i} from one to {n} of {A} sub {i}", 3)
    if k < 0.75:
        c = rel(r, X(x, x), num(r, 0, 5), pick(r, RELS[1:5]))
        return X(f"\\{{{x} \\in {S[0]} \\mid {c.tex}\\}}", f"the set of all {x} in {S[1]} such that {c.spk}", 1)
    if k < 0.85:
        return X(f"{A} \\cap {B} \\subseteq {A}", f"{A} intersection {B} is a subset of or equal to {A}", 4)
    return X(f"{x} \\in {A} \\setminus {B}", f"{x} is in {A} set minus {B}", 4)


def root_analysis(r, d):
    k = r.random(); e = pick(r, [("\\epsilon", "epsilon"), ("\\delta", "delta")])
    if k < 0.25:
        a, b = r.sample(list("xyab"), 2)
        return X(f"|{a} - {b}| < {e[0]}", f"the absolute value of {a} minus {b} is less than {e[1]}", 4)
    if k < 0.4:
        return X("\\forall \\epsilon > 0, \\; |a_n - L| < \\epsilon", "for all epsilon greater than zero the absolute value of a sub n minus L is less than epsilon", 4)
    if k < 0.6:
        A = pick(r, "AKSE"); f = factor(r, 1)
        if r.random() < 0.5:
            return X(f"\\sup_{{x \\in {A}}} {f.tex}", f"the supremum over x in {A} of {f.spk}", 3)
        return X(f"\\inf_{{x \\in {A}}} {f.tex}", f"the infimum over x in {A} of {f.spk}", 3)
    if k < 0.75:
        v = pick(r, "xyz"); return X(f"{v}_n \\to {v}", f"{v} sub n converges to {v}", 4)
    if k < 0.9:
        A = pick(r, "ABKU"); return rel(r, X(f"\\overline{{\\overline{{{A}}}}}", f"the closure of the closure of {A}"), X(f"\\overline{{{A}}}", f"the closure of {A}")) if False else X(f"\\partial {A} \\subseteq \\overline{{{A}}}", f"the boundary of {A} is a subset of or equal to the closure of {A}", 4)
    return X("g : A \\to B", "g from A to B", 4)


def root_ml(r, d):
    k = r.random(); th = pick(r, [("\\theta", "theta"), ("w", "w"), ("\\phi", "phi")])
    if k < 0.2:
        return rel(r, X(f"{th[0]}_{{t+1}}", f"{th[1]} sub t plus one"), X(f"{th[0]}_t - \\eta \\nabla L({th[0]}_t)", f"{th[1]} sub t minus eta nabla L of {th[1]} sub t"))
    if k < 0.35:
        return X(f"\\nabla_{th[0]} L({th[0]})", f"nabla sub {th[1]} L of {th[1]}", 3)
    if k < 0.5:
        op = pick(r, [("\\arg\\min", "arg min"), ("\\arg\\max", "arg max")])
        return X(f"{op[0]}_{th[0]} L({th[0]})", f"{op[1]} over {th[1]} of L of {th[1]}", 3)
    if k < 0.65:
        return X("L = \\frac{1}{N} \\sum_{i=1}^N (y_i - \\hat{y}_i)^2", "L equals one over N times the sum from i equals one to N of open paren y sub i minus y hat sub i close paren squared", 4)
    if k < 0.78:
        f = pick(r, ["f", "g"])
        return X(f"\\mathbb{{E}}_{{x \\sim p(x)}}[{f}(x)]", f"the expectation over x distributed as p of x of {f} of x", 3)
    if k < 0.84:
        i = pick(r, "ij")
        return rel(r, X(f"\\hat{{y}}_{i}", f"y hat sub {i}"), X(f"f(x_{i})", f"f of x sub {i}"))
    if k < 0.92:
        return X("\\mathbb{E}[\\ell(f(x), y)]", "the expectation of ell of f of x and y", 3)
    return X("D_{KL}(p \\| q) \\geq 0", "KL divergence of p and q is greater than or equal to zero", 4)


def root_physics(r, d):
    k = r.random()
    if k < 0.2:
        return rel(r, X("\\vec{L}", "vector L"), X("I \\vec{\\omega}", "I vector omega"))
    if k < 0.4:
        m = pick(r, ["m", "M"]); g = pick(r, ["g", "a"])
        return rel(r, X("U", "U"), X(f"{m} {g} h", f"{m} {g} h")) if r.random() < 0.5 else rel(r, X("T", "T"), X(f"\\frac{{1}}{{2}} I \\omega^2", "one half I omega squared"))
    if k < 0.55:
        return rel(r, X("\\nabla \\cdot \\vec{E}", "nabla dot vector E"), X("\\frac{\\rho}{\\epsilon_0}", "rho over epsilon sub zero"))
    if k < 0.7:
        return rel(r, X("\\vec{v}", "vector v"), X("\\frac{d\\vec{r}}{dt}", "d vector r d t"))
    if k < 0.85:
        return rel(r, X("\\Delta p", "delta p"), X("F \\Delta t", "F delta t"))
    q1 = pick(r, [("m_1", "m sub one"), ("q_1", "q sub one")]); q2 = pick(r, [("m_2", "m sub two"), ("q_2", "q sub two")])
    return rel(r, X("F", "F"), X(f"k \\frac{{{q1[0]} {q2[0]}}}{{r^2}}", f"k times {q1[1]} {q2[1]} over r squared"))


def root_chem(r, d):
    k = r.random()
    f = pick(r, [("\\mathrm{H_2O}", "H two O"), ("\\mathrm{CO_2}", "C O two"), ("\\mathrm{NH_3}", "N H three"), ("\\mathrm{CH_4}", "C H four"), ("\\mathrm{O_2}", "O two"), ("\\mathrm{NaCl}", "N a C l")])
    g = pick(r, [("\\mathrm{H_2}", "H two"), ("\\mathrm{N_2}", "N two"), ("\\mathrm{Cl_2}", "C l two"), ("\\mathrm{O_2}", "O two")])
    if k < 0.4:
        a, b = r.choice([(1, 2), (2, 1), (2, 2), (3, 2)])
        return X(f"{a} {g[0]} + {b} {f[0]} \\to {a + b} {f[0]}", f"{say_int(a)} {g[1]} plus {say_int(b)} {f[1]} yields {say_int(a + b)} {f[1]}", 4)
    if k < 0.7:
        return rel(r, X("\\Delta G", "delta G"), X("\\Delta H - T \\Delta S", "delta H minus T delta S"))
    return rel(r, X("K", "K"), X("\\frac{[\\mathrm{C}]}{[\\mathrm{A}]}", "the concentration of C over the concentration of A"))


def root_cs(r, d):
    k = r.random()
    inner = factor(r, 1) if r.random() < 0.5 else prod(r, 1, 2)
    a = pick(r, [("O", "big O"), ("\\Theta", "theta"), ("\\Omega", "omega")])
    if k < 0.45:
        return X(f"{a[0]}({inner.tex})", f"{a[1]} of {inner.spk}", 3)
    if k < 0.7:
        return rel(r, X("T(n)", "T of n"), X(f"{a[0]}({inner.tex})", f"{a[1]} of {inner.spk}"))
    if k < 0.85:
        b = pick(r, [("2", "two"), ("3", "three")])
        return rel(r, X("T(n)", "T of n"), X(f"2 T(n - 1) + {b[0]}", f"two T of n minus one plus {b[1]}"))
    return X("\\log_2 n", "log base two of n", 3)


ROOTS = [("algebra", root_algebra, 3.0), ("calculus", root_calculus, 3.5), ("linear_algebra", root_linalg, 1.5), ("probability", root_prob, 2.0), ("set_theory", root_sets, 1.2),
         ("analysis", root_analysis, 1.0), ("ml", root_ml, 2.0), ("physics", root_physics, 1.5), ("chemistry", root_chem, 0.5), ("cs", root_cs, 0.7)]


def compose(n, seed=0, max_depth=3, max_repeat=3):
    """Generate n unique compositional expressions. Returns records like grammar.generate()."""
    r = random.Random(seed)
    names = [x[0] for x in ROOTS]; w = [x[2] for x in ROOTS]
    seen, cnt, out, tries = set(), {}, [], 0
    while len(out) < n and tries < n * 30:
        tries += 1
        dom = r.choices(range(len(ROOTS)), w)[0]
        d = r.choice([1, 2, 2, 3]) if max_depth >= 3 else r.choice([1, 2])
        try:
            e = ROOTS[dom][1](r, d)
        except Exception:
            continue
        if e is None or not e.spk.strip():
            continue
        key = e.tex.replace(" ", "")
        cnt[key] = cnt.get(key, 0) + 1
        if cnt[key] > max_repeat or len(e.tex) > 95 or len(e.spk.split()) > 42:
            continue
        seen.add(key)
        out.append(dict(latex=e.tex, spoken_text=" ".join(e.spk.split()), domain=names[dom], difficulty=min(5, 2 + d + (e.prec >= 3)), template="compose", heldout=False))
    return out


if __name__ == "__main__":
    import sys
    for x in compose(int(sys.argv[1]) if len(sys.argv) > 1 else 40, seed=int(sys.argv[2]) if len(sys.argv) > 2 else 1):
        print(f"{x['domain']:14s} {x['latex']:60s} | {x['spoken_text']}")
