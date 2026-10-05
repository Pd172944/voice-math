"""Grammar for synthetic mathematical/scientific expressions.

Every template returns (latex, spoken) built from the same random draws, so the LaTeX is the source of truth and the
spoken form is a randomly-phrased realisation of it. Templates are registered per domain with a difficulty (1-5) and a
`heldout` flag: heldout templates are never used for train/val, only for the "hard" test split.
"""
import random
from dataclasses import dataclass
from typing import Callable

TEMPLATES = []  # list of Template


@dataclass
class Template:
    name: str
    domain: str
    difficulty: int
    fn: Callable
    heldout: bool = False


def tpl(domain, difficulty, heldout=False):
    def deco(fn):
        TEMPLATES.append(Template(fn.__name__, domain, difficulty, fn, heldout))
        return fn
    return deco


# ----------------------------------------------------------------------------- spoken helpers
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def say_int(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n == 100:
        return "one hundred"
    return " ".join(ONES[int(c)] for c in str(n))


def say_num(s):
    s = str(s)
    if s.startswith("-"):
        return "minus " + say_num(s[1:])
    if "." in s:
        a, b = s.split(".")
        return say_int(int(a)) + " point " + " ".join(ONES[int(c)] for c in b)
    return say_int(int(s))


def pick(r, *opts):
    return r.choice(opts)


EQ = ("equals", "is equal to", "equals", "equals")
LETTERS = list("xyztnkmuvwabc")
IDX = list("ijknm")
CAPS = list("ABCMPQSTUVWXYZ")
GREEK = ["alpha", "beta", "gamma", "delta", "epsilon", "theta", "lambda", "mu", "nu", "rho", "sigma", "tau", "phi", "psi", "omega", "eta", "kappa", "xi", "zeta"]


def greek(r):
    g = r.choice(GREEK)
    return "\\" + g, g


def var(r, pool=None):
    v = r.choice(pool or LETTERS)
    return v, v


def num(r, lo=0, hi=12):
    if r.random() < 0.12:
        d = r.choice(["0.5", "0.1", "2.5", "3.14", "1.5", "0.01", "9.8"])
        return d, say_num(d)
    n = r.choice(list(range(lo, hi + 1)) + [16, 20, 100]) if hi >= 12 else r.randint(lo, hi)
    return str(n), say_int(n)


def atom(r):
    k = r.random()
    if k < 0.5:
        return var(r)
    if k < 0.8:
        return num(r, 1, 9)
    return greek(r)


def power(r, base=None):
    b = base or var(r)
    k = r.random()
    if k < 0.35:
        return f"{b[0]}^2", f"{b[1]} " + pick(r, "squared", "squared", "to the second")
    if k < 0.5:
        return f"{b[0]}^3", f"{b[1]} cubed"
    if k < 0.75:
        e = var(r, list("nkmp"))
        return f"{b[0]}^{e[0]}", f"{b[1]} " + pick(r, "to the", "to the power of", "raised to the") + f" {e[1]}"
    if k < 0.9:
        return f"{b[0]}^{{-1}}", f"{b[1]} " + pick(r, "inverse", "to the minus one")
    n = r.choice([4, 5, 6])
    return f"{b[0]}^{n}", f"{b[1]} to the {['', '', '', '', 'fourth', 'fifth', 'sixth'][n]}"


def sub(r, base=None):
    b = base or var(r)
    k = r.random()
    if k < 0.65:
        i = var(r, IDX + ["0", "1", "2"] if False else IDX)
        if r.random() < 0.2:
            n = r.choice([0, 1, 2])
            return f"{b[0]}_{n}", f"{b[1]} sub {say_int(n)}"
        return f"{b[0]}_{i[0]}", f"{b[1]} sub {i[1]}"
    if k < 0.9:
        i, j = r.sample(IDX, 2)
        return f"{b[0]}_{{{i},{j}}}", f"{b[1]} sub {i} {j}"
    return f"{b[0]}_{{t+1}}", f"{b[1]} sub t plus one"


def func_app(r, arg=None):
    a = arg or (var(r) if r.random() < 0.7 else greek(r))
    k = r.random()
    if k < 0.3:
        fn = r.choice(["f", "g", "h"])
        return f"{fn}({a[0]})", f"{fn} of {a[1]}"
    f, name = r.choice([("\\sin", "sine"), ("\\cos", "cosine"), ("\\log", "log"), ("\\ln", "natural log"), ("\\tan", "tangent"), ("\\exp", "exponential")])
    if f == "\\ln":
        name = "natural log"
    if f in ("\\sin", "\\cos", "\\tan") and r.random() < 0.5:
        name = {"\\sin": "sine", "\\cos": "cosine", "\\tan": "tangent"}[f]
    elif f in ("\\sin", "\\cos", "\\tan"):
        name = {"\\sin": "sin", "\\cos": "cos", "\\tan": "tan"}[f]
    if r.random() < 0.5:
        return f"{f}({a[0]})", f"{name} of {a[1]}"
    return f"{f} {a[0]}", f"{name} {a[1]}"


def exp_e(r):
    k = r.random()
    if k < 0.4:
        a = var(r)
        return f"e^{a[0]}", f"e to the {a[1]}"
    if k < 0.7:
        a = var(r)
        return f"e^{{-{a[0]}}}", f"e to the minus {a[1]}"
    if k < 0.85:
        a = var(r)
        return f"e^{{-{a[0]}^2}}", f"e to the minus {a[1]} squared"
    a = var(r)
    return f"e^{{{a[0]}^2}}", f"e to the {a[1]} squared"


def simple_term(r):
    k = r.random()
    if k < 0.2:
        return var(r)
    if k < 0.3:
        return num(r, 1, 9)
    if k < 0.5:
        return power(r)
    if k < 0.6:
        return sub(r)
    if k < 0.75:
        return func_app(r)
    if k < 0.85:
        return exp_e(r)
    if k < 0.93:
        c = num(r, 2, 9)
        v = var(r)
        return f"{c[0]}{v[0]}", f"{c[1]} {v[1]}"
    return greek(r)


def sc(b):
    """Wrap a sub/superscript in braces unless it is a single char or a single command like \\pi."""
    return b if (len(b) == 1 or (b.startswith("\\") and b[1:].isalpha())) else "{" + b + "}"


def paren(r, inner):
    o, c = r.choice([("open paren", "close paren"), ("left paren", "right paren"), ("open parenthesis", "close parenthesis"),
                     ("open paren", "close paren")])
    return f"({inner[0]})", f"{o} {inner[1]} {c}"


def frac_simple(r):
    n = simple_term(r)
    d = simple_term(r)
    return f"\\frac{{{n[0]}}}{{{d[0]}}}", f"{n[1]} over {d[1]}"


def sqrt_term(r):
    if r.random() < 0.5:
        a = var(r)
        return f"\\sqrt{{{a[0]}}}", f"the square root of {a[1]}"
    a, b = power(r, var(r, list("xyzab"))), power(r, var(r, list("xyzab")))
    inner = (f"{a[0]} + {b[0]}", f"{a[1]} plus {b[1]}")
    p = paren(r, inner)
    return f"\\sqrt{{{inner[0]}}}", f"the square root of {p[1]}"


def sum_expr(r, nterms=None):
    n = nterms or r.choice([2, 2, 3])
    parts = [simple_term(r) for _ in range(n)]
    tex, spk = parts[0]
    for t in parts[1:]:
        if r.random() < 0.65:
            tex += f" + {t[0]}"; spk += f" plus {t[1]}"
        else:
            tex += f" - {t[0]}"; spk += f" minus {t[1]}"
    return tex, spk


def rel(r, l, rr, kind="="):
    m = {"=": (" = ", EQ), "<": (" < ", ("is less than", "less than")), ">": (" > ", ("is greater than", "greater than")),
         "\\leq": (" \\leq ", ("is less than or equal to", "less than or equal to")),
         "\\geq": (" \\geq ", ("is greater than or equal to", "greater than or equal to")),
         "\\neq": (" \\neq ", ("is not equal to", "does not equal")),
         "\\approx": (" \\approx ", ("is approximately", "approximately equal to"))}
    t, w = m[kind]
    return f"{l[0]}{t}{rr[0]}", f"{l[1]} {r.choice(w)} {rr[1]}"


# ----------------------------------------------------------------------------- ALGEBRA
@tpl("algebra", 1)
def alg_eq_simple(r):
    return rel(r, simple_term(r), simple_term(r))


@tpl("algebra", 2)
def alg_sum_eq(r):
    return rel(r, sum_expr(r), simple_term(r), r.choice(["=", "=", "\\leq", "\\geq", "<", ">", "\\neq"]))


@tpl("algebra", 2)
def alg_frac_eq(r):
    return rel(r, simple_term(r), frac_simple(r))


@tpl("algebra", 2)
def alg_sqrt_eq(r):
    return rel(r, simple_term(r), sqrt_term(r))


@tpl("algebra", 3)
def alg_pythag(r):
    a, b, c = r.sample(list("abcxyzrs"), 3)
    sq = lambda v: (f"{v}^2", f"{v} squared")
    return f"{a}^2 + {b}^2 = {c}^2", f"{a} squared plus {b} squared {r.choice(EQ)} {c} squared"


@tpl("algebra", 3)
def alg_paren_prod(r):
    a, b = sum_expr(r, 2), sum_expr(r, 2)
    pa, pb = paren(r, a), paren(r, b)
    return f"{pa[0]}{pb[0]}", f"{pa[1]} {pb[1]}"


@tpl("algebra", 4, heldout=True)
def alg_quadratic(r):
    return "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}", "x equals negative b plus or minus the square root of b squared minus four a c all over two a"


@tpl("algebra", 2)
def alg_const(r):
    k = r.choice([("\\pi", "pi"), ("e", "e"), ("\\phi", "phi")])
    v = r.choice([("3.14", "three point one four"), ("2.72", "two point seven two"), ("1.62", "one point six two")])
    kk = {"\\pi": v if False else ("3.14", "three point one four"), "e": ("2.72", "two point seven two"), "\\phi": ("1.62", "one point six two")}[k[0]]
    return rel(r, k, (kk[0], kk[1]), "\\approx")


@tpl("algebra", 2)
def alg_circle(r):
    k = r.random()
    if k < 0.34:
        return "A = \\pi r^2", f"A {r.choice(EQ)} pi r squared"
    if k < 0.67:
        return "C = 2 \\pi r", f"C {r.choice(EQ)} two pi r"
    return "V = \\frac{4}{3} \\pi r^3", f"V {r.choice(EQ)} four thirds pi r cubed"


@tpl("algebra", 3)
def alg_fraction_numeric(r):
    a, b = r.choice([(1, 2), (1, 3), (2, 3), (3, 4), (1, 4), (5, 8)])
    nm = {(1, 2): "one half", (1, 3): "one third", (2, 3): "two thirds", (3, 4): "three fourths", (1, 4): "one fourth", (5, 8): "five eighths"}[(a, b)]
    v = var(r)
    return f"\\frac{{{a}}}{{{b}}} {v[0]}^2", f"{nm} {v[1]} squared"


# ----------------------------------------------------------------------------- FUNCTIONS / CALCULUS
@tpl("calculus", 2)
def fn_def(r):
    v = var(r, list("xt"))
    f = r.choice("fgh")
    rhs = r.choice([power(r, v), func_app(r, v), exp_e(r) if False else func_app(r, v), sum_expr(r, 2)])
    return f"{f}({v[0]}) = {rhs[0]}", f"{f} of {v[1]} {r.choice(EQ)} {rhs[1]}"


@tpl("calculus", 2)
def fn_prime(r):
    f = r.choice("fgh")
    v = var(r, list("xt"))
    k = r.random()
    if k < 0.5:
        return f"{f}'({v[0]})", f"{f} prime of {v[1]}"
    if k < 0.75:
        return f"{f}''({v[0]})", f"{f} double prime of {v[1]}"
    return f"{f}'({v[0]}) = 0", f"{f} prime of {v[1]} {r.choice(EQ)} zero"


@tpl("calculus", 3)
def calc_deriv(r):
    y = r.choice("fyuw")
    x = r.choice("xtsz")
    k = r.random()
    if k < 0.3:
        return f"\\frac{{d{y}}}{{d{x}}}", pick(r, f"d {y} d {x}", f"d {y} over d {x}", f"the derivative of {y} with respect to {x}")
    if k < 0.55:
        return f"\\frac{{d^2{y}}}{{d{x}^2}}", pick(r, f"d squared {y} d {x} squared", f"d squared {y} over d {x} squared", f"the second derivative of {y} with respect to {x}")
    t = func_app(r, (x, x)) if r.random() < 0.5 else power(r, (x, x))
    if r.random() < 0.5:
        rhs = rel(r, (f"\\frac{{d}}{{d{x}}} {t[0]}", f"the derivative of {t[1]} with respect to {x}"), simple_term(r))
        return rhs
    return f"\\frac{{d}}{{d{x}}} {t[0]}", f"the derivative of {t[1]} with respect to {x}"


@tpl("calculus", 3)
def calc_partial(r):
    f = r.choice("fuLV")
    x, y = r.sample(list("xyzt"), 2)
    k = r.random()
    if k < 0.45:
        return f"\\frac{{\\partial {f}}}{{\\partial {x}}}", pick(r, f"partial {f} partial {x}", f"partial {f} over partial {x}", f"the partial derivative of {f} with respect to {x}", f"partial derivative {f} over partial {x}")
    if k < 0.7:
        return f"\\frac{{\\partial^2 {f}}}{{\\partial {x}^2}}", pick(r, f"partial squared {f} over partial {x} squared", f"the second partial derivative of {f} with respect to {x}")
    if k < 0.85:
        return f"\\frac{{\\partial^2 {f}}}{{\\partial {x} \\partial {y}}}", f"partial squared {f} over partial {x} partial {y}"
    rhs = simple_term(r)
    return f"\\frac{{\\partial {f}}}{{\\partial {x}}} = {rhs[0]}", f"the partial derivative of {f} with respect to {x} {r.choice(EQ)} {rhs[1]}"


def int_bound(r, kind):
    if kind == "lo":
        return r.choice([("0", "zero"), ("0", "zero"), ("1", "one"), ("-1", "minus one"), ("a", "a"), ("-\\infty", "minus infinity"), ("-\\pi", "minus pi")])
    return r.choice([("1", "one"), ("\\pi", "pi"), ("\\infty", "infinity"), ("b", "b"), ("2", "two"), ("2\\pi", "two pi"), ("T", "T")])


def int_body(r):
    k = r.random()
    if k < 0.3:
        return power(r, var(r, ["x"]))
    if k < 0.5:
        return func_app(r, ("x", "x"))
    if k < 0.65:
        return exp_e(r) if False else ("e^{-x}", "e to the minus x")
    if k < 0.8:
        return ("\\frac{1}{x}", "one over x")
    if k < 0.9:
        return sum_expr(r, 2)
    return ("f(x)", "f of x")


@tpl("calculus", 3)
def calc_int_def(r):
    lo, hi = int_bound(r, "lo"), int_bound(r, "hi")
    b = int_body(r)
    return f"\\int_{sc(lo[0])}^{sc(hi[0])} {b[0]}\\,dx", f"the integral from {lo[1]} to {hi[1]} of {b[1]} d x"


@tpl("calculus", 2)
def calc_int_indef(r):
    b = int_body(r)
    return f"\\int {b[0]}\\,dx", f"the integral of {b[1]} d x"


@tpl("calculus", 4)
def calc_int_eq(r):
    lo, hi = ("0", "zero"), r.choice([("1", "one"), ("\\pi", "pi"), ("\\infty", "infinity")])
    b = int_body(r)
    v = r.choice([("1", "one"), ("2", "two"), ("0", "zero"), ("\\frac{1}{3}", "one third"), ("\\frac{\\pi}{2}", "pi over two")])
    h = sc(hi[0])
    return f"\\int_0^{h} {b[0]}\\,dx = {v[0]}", f"the integral from zero to {hi[1]} of {b[1]} d x {r.choice(EQ)} {v[1]}"


@tpl("calculus", 3)
def calc_sum(r):
    i = r.choice("ikj")
    lo = r.choice([("1", "one"), ("0", "zero")])
    hi = r.choice([("n", "n"), ("N", "N"), ("\\infty", "infinity"), ("m", "m")])
    body = r.choice([sub(r, (r.choice("xaq"),) * 2), None, ("1/k^2", None)])
    if body is None or body[1] is None:
        body = (f"{i}^2", f"{i} squared") if r.random() < 0.5 else (f"\\frac{{1}}{{{i}}}", f"one over {i}")
    if r.random() < 0.5:
        b = r.choice("xaq")
        body = (f"{b}_{i}", f"{b} sub {i}")
    h = hi[0]
    return f"\\sum_{{{i}={lo[0]}}}^{sc(h)} {body[0]}", f"the sum from {i} equals {lo[1]} to {hi[1]} of {body[1]}"


@tpl("calculus", 4)
def calc_series(r):
    return "\\sum_{k=0}^\\infty \\frac{x^k}{k!}", "the sum from k equals zero to infinity of x to the k over k factorial"


@tpl("calculus", 3)
def calc_prod(r):
    i = r.choice("ik")
    return f"\\prod_{{{i}=1}}^n a_{i}", f"the product from {i} equals one to n of a sub {i}"


@tpl("calculus", 3)
def calc_limit(r):
    k = r.random()
    if k < 0.4:
        a = r.choice([("0", "zero"), ("1", "one"), ("\\infty", "infinity"), ("\\pi", "pi")])
        b = func_app(r, ("x", "x")) if r.random() < 0.5 else simple_term(r)
        return f"\\lim_{{x \\to {a[0]}}} {b[0]}", f"the limit as x goes to {a[1]} of {b[1]}"
    if k < 0.75:
        s = sub(r, ("a", "a"))
        if "t+1" in s[0] or "," in s[0]:
            s = ("a_n", "a sub n")
        return f"\\lim_{{n \\to \\infty}} {s[0].replace('a_' + s[0][2:], 'a_n')}", f"the limit as n goes to infinity of a sub n"
    return "\\lim_{x \\to 0} \\frac{\\sin x}{x} = 1", "the limit as x goes to zero of sine x over x equals one"


@tpl("calculus", 4, heldout=True)
def calc_ftc(r):
    return "\\int_a^b f'(x)\\,dx = f(b) - f(a)", "the integral from a to b of f prime of x d x equals f of b minus f of a"


# ----------------------------------------------------------------------------- LINEAR ALGEBRA
@tpl("linear_algebra", 2)
def la_matrix_ops(r):
    A = r.choice("ABCMPQ")
    k = r.random()
    if k < 0.35:
        return f"{A}^{{-1}}", f"{A} " + pick(r, "inverse", "to the minus one")
    if k < 0.65:
        return f"{A}^T", f"{A} transpose"
    if k < 0.85:
        return f"\\det({A})", pick(r, f"the determinant of {A}", f"determinant of {A}", f"det of {A}")
    return f"\\operatorname{{tr}}({A})", pick(r, f"the trace of {A}", f"trace of {A}")


@tpl("linear_algebra", 3)
def la_eigen(r):
    A = r.choice("ABM")
    if r.random() < 0.6:
        return f"{A}\\mathbf{{x}} = \\lambda\\mathbf{{x}}", f"{A} bold x {r.choice(EQ)} lambda bold x"
    i = r.choice("ij")
    return f"{A}\\mathbf{{v}}_{i} = \\lambda_{i}\\mathbf{{v}}_{i}", f"{A} bold v sub {i} {r.choice(EQ)} lambda sub {i} bold v sub {i}"


@tpl("linear_algebra", 3)
def la_det_char(r):
    A = r.choice("AB")
    inner = (f"{A} - \\lambda I", f"{A} minus lambda I")
    p = paren(r, inner)
    return f"\\det({inner[0]}) = 0", f"the determinant of {p[1]} {r.choice(EQ)} zero"


@tpl("linear_algebra", 3)
def la_norm(r):
    v = r.choice("xyvw")
    p = r.choice([("2", "two"), ("1", "one"), ("p", "p"), ("\\infty", "infinity")])
    if r.random() < 0.3:
        return f"\\ell_{p[0]}", pick(r, f"the {p[1]} norm", f"the little l {p[1]}")
    pp = p[0] if len(p[0]) == 1 or p[0].startswith("\\") else "{" + p[0] + "}"
    return f"\\|\\mathbf{{{v}}}\\|_{pp}", f"the {p[1]} norm of bold {v}"


@tpl("linear_algebra", 3)
def la_inner(r):
    u, v = r.sample("xyuvw", 2)
    k = r.random()
    if k < 0.5:
        return f"\\mathbf{{{u}}}^T \\mathbf{{{v}}}", f"bold {u} transpose bold {v}"
    return f"\\langle \\mathbf{{{u}}}, \\mathbf{{{v}}} \\rangle", f"the inner product of bold {u} and bold {v}"


@tpl("linear_algebra", 3)
def la_space(r):
    n = r.choice(["n", "m", "d", "k"])
    k = r.random()
    if k < 0.4:
        return f"\\mathbb{{R}}^{n}", f"R {n}"
    if k < 0.7:
        return f"\\mathbf{{x}} \\in \\mathbb{{R}}^{n}", f"bold x is in R {n}"
    m = r.choice([x for x in "nmdk" if x != n])
    A = r.choice("AWM")
    return f"{A} \\in \\mathbb{{R}}^{{{n} \\times {m}}}", f"{A} is in R {n} by {m}"


@tpl("linear_algebra", 4, heldout=True)
def la_svd(r):
    return "A = U \\Sigma V^T", "A equals U capital sigma V transpose"


# ----------------------------------------------------------------------------- PROBABILITY / STATISTICS
@tpl("probability", 2)
def pr_prob(r):
    X = r.choice("XYZ")
    x = X.lower()
    k = r.random()
    if k < 0.4:
        return f"P({X} = {x})", pick(r, f"the probability that {X} equals {x}", f"P of {X} equals {x}", f"the probability of {X} equals {x}")
    if k < 0.7:
        return f"P(A \\mid B)", pick(r, "the probability of A given B", "P of A given B")
    if k < 0.85:
        return f"P(A \\cap B)", "the probability of A intersection B"
    return f"P({X} \\leq {x})", f"the probability that {X} is less than or equal to {x}"


@tpl("probability", 2)
def pr_moments(r):
    X = r.choice("XYZ")
    k = r.random()
    if k < 0.4:
        return f"E[{X}]", pick(r, f"the expectation of {X}", f"E of {X}", f"the expected value of {X}")
    if k < 0.75:
        return f"\\operatorname{{Var}}({X})", pick(r, f"the variance of {X}", f"variance of {X}", f"var of {X}")
    Y = r.choice([c for c in "XYZ" if c != X])
    return f"\\operatorname{{Cov}}({X}, {Y})", f"the covariance of {X} and {Y}"


@tpl("probability", 3)
def pr_normal(r):
    X = r.choice("XYZ")
    mu = r.choice([("\\mu", "mu"), ("0", "zero")])
    sg = r.choice([("\\sigma^2", "sigma squared"), ("1", "one")])
    return f"{X} \\sim \\mathcal{{N}}({mu[0]}, {sg[0]})", pick(r, f"{X} is normally distributed with mean {mu[1]} and variance {sg[1]}", f"{X} is distributed as normal {mu[1]} {sg[1]}")


@tpl("probability", 3)
def pr_cond_exp(r):
    return "\\mathbb{E}[X \\mid Y = y]", pick(r, "the conditional expectation of X given Y equals y", "the expectation of X given Y equals y")


@tpl("probability", 3)
def pr_estimators(r):
    k = r.random()
    if k < 0.3:
        v = r.choice([("\\theta", "theta"), ("\\mu", "mu"), ("\\sigma", "sigma"), ("\\beta", "beta")])
        return f"\\hat{{{v[0]}}}", f"{v[1]} hat"
    if k < 0.6:
        v = r.choice("xyz")
        return f"\\bar{{{v}}}", f"{v} bar"
    if k < 0.8:
        v = r.choice("xy")
        return f"\\bar{{{v}}} = \\frac{{1}}{{n}} \\sum_{{i=1}}^n {v}_i", f"{v} bar {r.choice(EQ)} one over n times the sum from i equals one to n of {v} sub i"
    v = r.choice("xy")
    return f"\\tilde{{{v}}}", f"{v} tilde"


@tpl("probability", 4)
def pr_bayes(r):
    return "P(A \\mid B) = \\frac{P(B \\mid A) P(A)}{P(B)}", "the probability of A given B equals the probability of B given A times the probability of A over the probability of B"


@tpl("probability", 4, heldout=True)
def pr_rho(r):
    return "\\rho = \\frac{\\operatorname{Cov}(X, Y)}{\\sigma_X \\sigma_Y}", "rho equals the covariance of X and Y over sigma sub X sigma sub Y"


# ----------------------------------------------------------------------------- SET THEORY / LOGIC
@tpl("set_theory", 1)
def set_member(r):
    x = r.choice("xyabn")
    S = r.choice("ABSXC")
    k = r.random()
    if k < 0.6:
        return f"{x} \\in {S}", pick(r, f"{x} is in {S}", f"{x} belongs to {S}", f"{x} is an element of {S}")
    return f"{x} \\notin {S}", f"{x} is not in {S}"


@tpl("set_theory", 2)
def set_ops(r):
    A, B = r.sample("ABCSTUX", 2)
    k = r.random()
    if k < 0.3:
        return f"{A} \\subseteq {B}", pick(r, f"{A} is a subset of or equal to {B}", f"{A} subset or equal {B}")
    if k < 0.5:
        return f"{A} \\subset {B}", f"{A} is a subset of {B}"
    if k < 0.7:
        return f"{A} \\cup {B}", f"{A} union {B}"
    if k < 0.85:
        return f"{A} \\cap {B}", f"{A} intersection {B}"
    return f"{A} \\setminus {B}", f"{A} set minus {B}"


@tpl("set_theory", 3)
def set_quant(r):
    x = r.choice("xyn")
    S = r.choice([("\\mathbb{R}", "R"), ("\\mathbb{N}", "N"), ("\\mathbb{Z}", "Z"), ("\\mathbb{Q}", "Q"), ("\\mathbb{C}", "C")])
    k = r.random()
    if k < 0.45:
        return f"\\forall {x} \\in {S[0]}", f"for all {x} in {S[1]}"
    if k < 0.75:
        return f"\\exists {x} \\in {S[0]}", f"there exists {x} in {S[1]}"
    return f"\\forall {x} \\in {S[0]}, \\; {x}^2 \\geq 0", f"for all {x} in {S[1]} {x} squared is greater than or equal to zero"


@tpl("set_theory", 3)
def set_logic(r):
    k = r.random()
    if k < 0.5:
        return "p \\implies q", "p implies q"
    if k < 0.8:
        return "p \\iff q", "p if and only if q"
    return "A = \\emptyset", "A equals the empty set"


@tpl("set_theory", 3, heldout=True)
def set_chain(r):
    return "\\mathbb{N} \\subset \\mathbb{Z} \\subset \\mathbb{Q} \\subset \\mathbb{R}", "N is a subset of Z is a subset of Q is a subset of R"


# ----------------------------------------------------------------------------- ANALYSIS / TOPOLOGY
@tpl("analysis", 2)
def an_basic(r):
    k = r.random()
    if k < 0.3:
        return "\\epsilon - \\delta", "epsilon delta"
    if k < 0.55:
        A = r.choice("ABUKS")
        return f"\\overline{{{A}}}", pick(r, f"the closure of {A}", f"{A} closure")
    if k < 0.8:
        A = r.choice("ABUKS")
        return f"\\partial {A}", pick(r, f"the boundary of {A}", f"partial {A}")
    return "\\epsilon > 0", "epsilon is greater than zero"


@tpl("analysis", 3)
def an_abs(r):
    x, y = r.sample("xyab", 2)
    d = r.choice([("\\delta", "delta"), ("\\epsilon", "epsilon")])
    return f"|{x} - {y}| < {d[0]}", pick(r, f"the absolute value of {x} minus {y} is less than {d[1]}", f"absolute value of {x} minus {y} less than {d[1]}")


@tpl("analysis", 3)
def an_map(r):
    X, Y = r.sample("XYZ", 2)
    return f"f : {X} \\to {Y}", f"f from {X} to {Y}"


@tpl("analysis", 4)
def an_sup(r):
    A = r.choice("ASK")
    k = r.random()
    if k < 0.5:
        return f"\\sup_{{x \\in {A}}} f(x)", f"the supremum over x in {A} of f of x"
    return f"\\inf_{{x \\in {A}}} f(x)", f"the infimum over x in {A} of f of x"


@tpl("analysis", 4)
def an_converge(r):
    return "x_n \\to x", pick(r, "x sub n converges to x", "x sub n goes to x")


@tpl("analysis", 5, heldout=True)
def an_eps_delta_full(r):
    return "\\forall \\epsilon > 0 \\; \\exists \\delta > 0", "for all epsilon greater than zero there exists delta greater than zero"


# ----------------------------------------------------------------------------- MACHINE LEARNING
@tpl("ml", 3)
def ml_grad(r):
    k = r.random()
    if k < 0.5:
        return "\\nabla_\\theta L(\\theta)", pick(r, "nabla sub theta L of theta", "the gradient of L of theta with respect to theta")
    if k < 0.75:
        return "\\nabla f(x)", "nabla f of x"
    return "\\nabla_w J", "nabla sub w J"


@tpl("ml", 3)
def ml_argmin(r):
    p = r.choice([("\\theta", "theta"), ("w", "w"), ("\\phi", "phi")])
    op, opn = r.choice([("\\arg\\min", "arg min"), ("\\arg\\max", "arg max")])
    return f"{op}_{p[0]} L({p[0]})", f"{opn} over {p[1]} of L of {p[1]}"


@tpl("ml", 4)
def ml_expect(r):
    return "\\mathbb{E}_{x \\sim p(x)}[f(x)]", pick(r, "the expectation over x distributed as p of x of f of x", "the expectation of f of x where x is drawn from p of x")


@tpl("ml", 4)
def ml_softmax(r):
    k = r.random()
    if k < 0.5:
        return "\\operatorname{softmax}(x)", "softmax of x"
    return "\\operatorname{softmax}(x)_i = \\frac{e^{x_i}}{\\sum_j e^{x_j}}", "softmax of x sub i equals e to the x sub i over the sum over j of e to the x sub j"


@tpl("ml", 3)
def ml_sgd(r):
    return "\\theta_{t+1} = \\theta_t - \\eta \\nabla L(\\theta_t)", "theta sub t plus one equals theta sub t minus eta nabla L of theta sub t"


@tpl("ml", 3)
def ml_acts(r):
    k = r.random()
    if k < 0.4:
        return "\\sigma(x) = \\frac{1}{1 + e^{-x}}", "sigma of x equals one over one plus e to the minus x"
    if k < 0.7:
        return "\\operatorname{ReLU}(x) = \\max(0, x)", "ReLU of x equals max of zero and x"
    return "\\tanh(x)", "tanh of x"


@tpl("ml", 4)
def ml_kl(r):
    k = r.random()
    if k < 0.5:
        return "D_{KL}(p \\| q)", "KL divergence of p and q"
    return "L = \\frac{1}{N} \\sum_{i=1}^N \\ell(y_i, \\hat{y}_i)", "L equals one over N times the sum from i equals one to N of ell of y sub i and y hat sub i"


@tpl("ml", 5, heldout=True)
def ml_ce(r):
    return "- \\sum_i y_i \\log \\hat{y}_i", "minus the sum over i of y sub i log y hat sub i"


# ----------------------------------------------------------------------------- PHYSICS
@tpl("physics", 1)
def ph_classic(r):
    k = r.random()
    if k < 0.3:
        return "F = ma", pick(r, "F equals m a", "F is equal to m a")
    if k < 0.6:
        return "E = mc^2", pick(r, "E equals m c squared", "E is equal to m c squared")
    if k < 0.8:
        return "p = mv", "p equals m v"
    return "V = IR", "V equals I R"


@tpl("physics", 2)
def ph_vectors(r):
    k = r.random()
    if k < 0.4:
        return "\\vec{F} = m\\vec{a}", "vector F equals m vector a"
    if k < 0.7:
        return "\\vec{F} = q\\vec{E}", "vector F equals q vector E"
    return "\\vec{F}", "vector F"


@tpl("physics", 3)
def ph_kinematics(r):
    k = r.random()
    if k < 0.4:
        return "\\frac{d^2 x}{dt^2}", "d squared x d t squared"
    if k < 0.7:
        return "x = \\frac{1}{2} a t^2", "x equals one half a t squared"
    return "K = \\frac{1}{2} m v^2", "K equals one half m v squared"


@tpl("physics", 3)
def ph_quantum(r):
    k = r.random()
    if k < 0.35:
        return "E = \\hbar \\omega", "E equals h bar omega"
    if k < 0.7:
        return "\\Delta x \\Delta p \\geq \\frac{\\hbar}{2}", "delta x delta p is greater than or equal to h bar over two"
    return "\\hat{H} \\psi = E \\psi", "H hat psi equals E psi"


@tpl("physics", 4)
def ph_maxwell(r):
    k = r.random()
    if k < 0.5:
        return "\\nabla \\cdot \\vec{E} = \\frac{\\rho}{\\epsilon_0}", "nabla dot vector E equals rho over epsilon sub zero"
    return "\\nabla \\times \\vec{B} = \\mu_0 \\vec{J}", "nabla cross vector B equals mu sub zero vector J"


@tpl("physics", 5, heldout=True)
def ph_schrodinger(r):
    return "i \\hbar \\frac{\\partial \\psi}{\\partial t} = \\hat{H} \\psi", "i h bar partial psi over partial t equals H hat psi"


# ----------------------------------------------------------------------------- CHEMISTRY
@tpl("chemistry", 2)
def ch_formula(r):
    f, s = r.choice([("\\mathrm{H_2O}", "H two O"), ("\\mathrm{CO_2}", "C O two"), ("\\mathrm{NaCl}", "N a C l"), ("\\mathrm{CH_4}", "C H four"),
                     ("\\mathrm{H_2SO_4}", "H two S O four"), ("\\mathrm{NH_3}", "N H three"), ("\\mathrm{O_2}", "O two")])
    return f, s


@tpl("chemistry", 3)
def ch_thermo(r):
    k = r.random()
    if k < 0.5:
        return "\\Delta G = \\Delta H - T \\Delta S", "delta G equals delta H minus T delta S"
    if k < 0.8:
        return "PV = nRT", "P V equals n R T"
    return "\\mathrm{pH} = -\\log[\\mathrm{H}^+]", "pH equals minus log of the concentration of H plus"


@tpl("chemistry", 4, heldout=True)
def ch_reaction(r):
    return "2\\mathrm{H_2} + \\mathrm{O_2} \\to 2\\mathrm{H_2O}", "two H two plus O two yields two H two O"


# ----------------------------------------------------------------------------- COMPUTER SCIENCE
@tpl("cs", 2)
def cs_big_o(r):
    k = r.random()
    inner, sp = r.choice([("n", "n"), ("n^2", "n squared"), ("n \\log n", "n log n"), ("\\log n", "log n"), ("2^n", "two to the n"), ("n^3", "n cubed")])
    if k < 0.6:
        return f"O({inner})", f"big O of {sp}"
    if k < 0.8:
        return f"\\Theta({inner})", f"theta of {sp}"
    return f"\\Omega({inner})", f"omega of {sp}"


@tpl("cs", 3)
def cs_recur(r):
    return "T(n) = 2T\\left(\\frac{n}{2}\\right) + n", "T of n equals two T of n over two plus n"


@tpl("cs", 3)
def cs_log_base(r):
    b = r.choice([("2", "two"), ("10", "ten"), ("e", "e")])
    return f"\\log_{b[0] if len(b[0]) == 1 else '{' + b[0] + '}'} n", f"log base {b[1]} of n"


@tpl("cs", 2)
def cs_fact(r):
    return "n!", "n factorial"


DOMAINS = sorted({t.domain for t in TEMPLATES})


DOMAIN_WEIGHTS = dict(algebra=1.3, calculus=2.2, linear_algebra=1.6, probability=1.6, set_theory=1.1, analysis=1.0, ml=1.6, physics=1.3,
                      chemistry=0.6, cs=0.7)


def generate(n, seed=0, include_heldout=False, only_heldout=False, domains=None, max_repeat=40):
    """Return up to n records {latex, spoken_text, domain, difficulty, template, heldout}.
    Domains are sampled by weight (balanced mix); a given LaTeX may recur with different spoken phrasing up to max_repeat times."""
    r = random.Random(seed)
    pool = [t for t in TEMPLATES if (t.heldout if only_heldout else (include_heldout or not t.heldout))]
    if domains:
        pool = [t for t in pool if t.domain in domains]
    by_dom = {}
    for t in pool:
        by_dom.setdefault(t.domain, []).append(t)
    doms = list(by_dom); w = [DOMAIN_WEIGHTS.get(d, 1.0) for d in doms]
    seen, cnt, out, tries = set(), {}, [], 0
    while len(out) < n and tries < n * 80:
        tries += 1
        t = r.choice(by_dom[r.choices(doms, w)[0]])
        try:
            tex, spk = t.fn(r)
        except Exception:
            continue
        spk = " ".join(spk.split()); key = tex.replace(" ", "")
        if not spk or cnt.get(key, 0) >= max_repeat:
            continue
        seen.add((key, spk)); cnt[key] = cnt.get(key, 0) + 1
        out.append(dict(latex=tex, spoken_text=spk, domain=t.domain, difficulty=t.difficulty, template=t.name, heldout=t.heldout))
    return out


if __name__ == "__main__":
    import sys
    for x in generate(int(sys.argv[1]) if len(sys.argv) > 1 else 40, seed=1, include_heldout=True):
        print(f"{x['domain']:14s} {x['latex']:55s} | {x['spoken_text']}")
