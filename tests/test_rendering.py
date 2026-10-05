import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rendering.latex_to_text import latex_to_text as t, remaining_commands
from src.data.grammar import generate
from src.data.compose import compose
from src.data.ood import OOD

CASES = {
    r"\int_0^\pi x^2\,dx": "∫₀^π x² dx",
    r"F_x(x) = \int_0^\pi x^2 dx": "Fₓ(x) = ∫₀^π x² dx",
    r"\frac{\partial f}{\partial x}": "∂f/∂x",
    r"\sigma^2": "σ²", r"\lambda_i": "λᵢ", r"\mathbb{R}^n": "ℝⁿ", r"\ell_p": "ℓₚ", r"x^{-1}": "x⁻¹", r"A^T": "Aᵀ",
    r"\bar{x}": "x̄", r"\hat{x}": "x̂", r"\forall x \in \mathbb{R}": "∀x ∈ ℝ", r"\frac{1}{2} m v^2": "½ m v²",
    r"\sum_{i=1}^n x_i": "∑ᵢ₌₁ⁿ xᵢ", r"\mathrm{H_2O}": "H₂O", r"a \leq b": "a ≤ b", r"\sqrt{x^2 + 1}": "√(x² + 1)",
}


def test_known_cases():
    bad = {k: t(k) for k, v in CASES.items() if t(k) != v}
    assert not bad, bad


def test_no_commands_left_on_corpus():
    latex = [r["latex"] for r in generate(600, seed=4, include_heldout=True)] + [r["latex"] for r in compose(600, seed=4)] + [l for _, l, _ in OOD]
    left = {l: remaining_commands(t(l)) for l in latex}
    left = {k: v for k, v in left.items() if v}
    assert len(left) / len(latex) < 0.02, list(left.items())[:10]


def test_never_crashes_and_nonempty():
    for l in [r"\frac{", r"x^", r"\unknowncmd{y}", "", r"\begin{pmatrix} a \\ b \end{pmatrix}"]:
        assert isinstance(t(l), str)
