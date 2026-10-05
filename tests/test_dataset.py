import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.grammar import generate, TEMPLATES
from src.normalization.canonicalize import valid_latex, canonicalize
from src.evaluation.metrics import norm_latex


def test_grammar_valid_and_diverse():
    recs = generate(800, seed=3)
    assert len({r["domain"] for r in recs}) >= 8
    assert all(valid_latex(r["latex"]) for r in recs)
    assert all(r["spoken_text"] for r in recs)


def test_heldout_templates_not_in_train_pool():
    ho = {t.name for t in TEMPLATES if t.heldout}
    assert ho and not ({r["template"] for r in generate(2000, seed=1)} & ho)


def test_rule_normalizer_roundtrip_rate():
    recs = generate(400, seed=9)
    ok = sum(norm_latex(canonicalize(r["spoken_text"])["latex"]) == norm_latex(r["latex"]) for r in recs) / len(recs)
    assert ok > 0.78   # in-distribution phrasing; chemistry/long-tail templates are intentionally hard for the rules


def test_metric_equivalences():
    assert norm_latex(r"\frac12") == norm_latex("1/2") == norm_latex(r"\frac{1}{2}")
    assert norm_latex("x^2") == norm_latex("x^{2}")
