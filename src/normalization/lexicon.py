"""Math terminology lexicon: spoken phrase -> LaTeX. Longest-match wins."""

GREEK = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu",
         "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"]
GREEK_UPPER = ["gamma", "delta", "theta", "lambda", "xi", "pi", "sigma", "phi", "psi", "omega"]
GREEK_VARIANTS = {"varepsilon": r"\varepsilon", "varphi": r"\varphi", "vartheta": r"\vartheta", "varrho": r"\varrho"}

SYMBOLS = {  # spoken (lowercase) -> latex
    "infinity": r"\infty", "partial": r"\partial", "nabla": r"\nabla", "del": r"\nabla", "for all": r"\forall",
    "forall": r"\forall", "there exists": r"\exists", "exists": r"\exists", "element of": r"\in", "belongs to": r"\in",
    "is in": r"\in", "in": r"\in", "not in": r"\notin", "subset or equal": r"\subseteq", "subset of or equal to": r"\subseteq",
    "subset of": r"\subset", "subset": r"\subset", "superset of": r"\supset", "union": r"\cup", "intersection": r"\cap",
    "empty set": r"\emptyset", "approximately": r"\approx", "approximately equal to": r"\approx", "is approximately": r"\approx",
    "greater than or equal to": r"\geq", "greater than or equal": r"\geq", "less than or equal to": r"\leq",
    "less than or equal": r"\leq", "not equal to": r"\neq", "does not equal": r"\neq", "greater than": ">", "less than": "<",
    "plus or minus": r"\pm", "times": r"\cdot", "cross": r"\times", "dot": r"\cdot", "implies": r"\implies",
    "if and only if": r"\iff", "maps to": r"\mapsto", "goes to": r"\to", "tends to": r"\to", "approaches": r"\to",
    "proportional to": r"\propto", "such that": r"\mid", "equals": "=", "is equal to": "=", "equal to": "=", "plus": "+",
    "minus": "-", "divided by": r"\div", "factorial": "!", "ell": r"\ell", "h bar": r"\hbar", "hbar": r"\hbar",
    "planck constant": r"\hbar", "set minus": r"\setminus", 
    "natural numbers": r"\mathbb{N}", "integers": r"\mathbb{Z}", "rationals": r"\mathbb{Q}", "reals": r"\mathbb{R}",
    "real numbers": r"\mathbb{R}", "complex numbers": r"\mathbb{C}", "tilde": r"\sim", "distributed as": r"\sim",
    "dagger": r"^\dagger", "is not in": r"\notin", "yields": r"\to", "converges to": r"\to", "is an element of": r"\in",
    "is a subset of or equal to": r"\subseteq", "is a subset of": r"\subset", "is not equal to": r"\neq", "is in": r"\in", "prime": "'", "ellipsis": r"\ldots", "dot dot dot": r"\ldots",
}

FUNCTIONS = {"sine": r"\sin", "sin": r"\sin", "cosine": r"\cos", "cos": r"\cos", "tangent": r"\tan", "tan": r"\tan",
             "log": r"\log", "natural log": r"\ln", "ln": r"\ln", "exponential": r"\exp", "exp": r"\exp",
             "arg min": r"\arg\min", "argmin": r"\arg\min", "arg max": r"\arg\max", "argmax": r"\arg\max",
             "determinant": r"\det", "det": r"\det", "trace": r"\operatorname{tr}", "softmax": r"\operatorname{softmax}",
             "sup": r"\sup", "supremum": r"\sup", "inf": r"\inf", "infimum": r"\inf", "max": r"\max", "min": r"\min",
             "variance": r"\operatorname{Var}", "covariance": r"\operatorname{Cov}", "sigmoid": r"\sigma"}

BLACKBOARD = {"r": "R", "n": "N", "z": "Z", "q": "Q", "c": "C", "e": "E", "p": "P"}

# words that signal a math context (used to decide whether ambiguous words such as "pie"/"in"/"cross" are math)
MATH_TRIGGERS = set("""equals equal squared cubed integral derivative partial sum limit infinity sigma lambda theta alpha beta gamma
delta epsilon omega mu nu rho tau phi psi chi eta zeta kappa sub over norm vector matrix transpose inverse determinant
approximately plus minus times divided probability expectation variance covariance gradient nabla subset union intersection
exists forall det trace implies iff element subset bold closure absolute inner normally distributed divergence yields converges softmax norm gradient derivative supremum infimum greater less factorial sine cosine tangent log exponential prime hat bar tilde square root fraction""".split())
MATH_TRIGGERS.add("pi")


def number_words():
    ones = "zero one two three four five six seven eight nine".split()
    teens = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
    tens = "twenty thirty forty fifty sixty seventy eighty ninety".split()
    m = {w: str(i) for i, w in enumerate(ones)}
    m.update({w: str(10 + i) for i, w in enumerate(teens)})
    m.update({w: str(20 + 10 * i) for i, w in enumerate(tens)})
    m["hundred"] = "100"
    return m


NUMBERS = number_words()
ORDINAL_POW = {"squared": "2", "cubed": "3", "fourth": "4", "fifth": "5", "sixth": "6"}
