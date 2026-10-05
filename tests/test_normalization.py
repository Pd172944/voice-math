import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.normalization.canonicalize import canonicalize
from src.evaluation.metrics import norm_latex

CASES = {
 "F sub x of x equals zero": "F_x(x) = 0",
 "F sub x y": "F_{x,y}",
 "sigma squared": r"\sigma^2",
 "lambda sub i": r"\lambda_i",
 "R n": r"\mathbb{R}^n",
 "partial derivative": r"\partial",
 "greater than or equal": r"\geq",
 "for all x in R": r"\forall x \in \mathbb{R}",
 "x bar": r"\bar{x}", "x hat": r"\hat{x}",
 "the limit as n goes to infinity of a sub n": r"\lim_{n \to \infty} a_n",
 "the integral from zero to one of x squared d x": r"\int_0^1 x^2\,dx",
 "partial f partial x": r"\frac{\partial f}{\partial x}",
 "pi is approximately three point fourteen": r"\pi \approx 3.14",
 "the integral from zero to pie of x squared d x": r"\int_0^\pi x^2\,dx",
}

def test_cases():
    bad = {k: canonicalize(k)["latex"] for k, v in CASES.items() if norm_latex(canonicalize(k)["latex"]) != norm_latex(v)}
    assert not bad, bad

def test_context():
    assert canonicalize("the pie is on the table")["latex"] == "the pie is on the table"
    assert not canonicalize("the pie is on the table")["is_math"]
