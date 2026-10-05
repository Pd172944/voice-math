"""On-the-fly LaTeX -> readable Unicode text (no TeX engine needed).

    latex_to_text(r"\\int_0^\\pi x^2\\,dx")      -> "∫₀^π x² dx"
    latex_to_text(r"\\frac{\\partial f}{\\partial x}") -> "∂f/∂x"
    latex_to_text(r"\\mathbb{R}^n \\ni \\lambda_i")   -> "ℝⁿ ∋ λᵢ"

Unicode super/subscripts are used whenever every character has one; otherwise it falls back to ^(…) / _(…).
Unknown commands are kept verbatim (so nothing is silently dropped)."""
import re
import unicodedata

GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ϵ", "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "ϑ", "iota": "ι",
         "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "omicron": "ο", "pi": "π", "varpi": "ϖ", "rho": "ρ", "varrho": "ϱ", "sigma": "σ", "varsigma": "ς", "tau": "τ",
         "upsilon": "υ", "phi": "ϕ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π",
         "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω"}
REL = {"leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠", "approx": "≈", "equiv": "≡", "sim": "∼", "simeq": "≃", "cong": "≅", "propto": "∝", "in": "∈", "notin": "∉",
       "ni": "∋", "subset": "⊂", "subseteq": "⊆", "supset": "⊃", "supseteq": "⊇", "to": "→", "rightarrow": "→", "leftarrow": "←", "leftrightarrow": "↔", "mapsto": "↦",
       "Rightarrow": "⇒", "Leftarrow": "⇐", "Leftrightarrow": "⇔", "implies": "⟹", "iff": "⟺", "gets": "←", "ll": "≪", "gg": "≫", "perp": "⊥", "parallel": "∥", "mid": "∣",
       "colon": ":", "models": "⊨", "vdash": "⊢", "land": "∧", "lor": "∨"}
BIN = {"pm": "±", "mp": "∓", "times": "×", "cdot": "·", "div": "÷", "cup": "∪", "cap": "∩", "setminus": "∖", "circ": "∘", "ast": "∗", "star": "⋆", "oplus": "⊕", "otimes": "⊗",
       "wedge": "∧", "vee": "∨", "bullet": "•"}
SYM = {"infty": "∞", "partial": "∂", "nabla": "∇", "forall": "∀", "exists": "∃", "emptyset": "∅", "varnothing": "∅", "hbar": "ℏ", "ell": "ℓ", "ldots": "…", "dots": "…", "cdots": "⋯",
       "vdots": "⋮", "ddots": "⋱", "dagger": "†", "prime": "′", "neg": "¬", "lnot": "¬", "top": "⊤", "bot": "⊥", "angle": "∠", "degree": "°", "aleph": "ℵ", "Re": "ℜ", "Im": "ℑ",
       "langle": "⟨", "rangle": "⟩", "lfloor": "⌊", "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉", "|": "‖", "Vert": "‖", "vert": "|", "{": "{", "}": "}", "lbrace": "{", "rbrace": "}",
       "%": "%", "&": "&", "#": "#", "$": "$", "_": "_", "backslash": "\\", "quad": "  ", "qquad": "    ", ",": " ", ";": " ", ":": " ", "!": "", " ": " ", "\\": "\n",
       "bigcup": "⋃", "bigcap": "⋂", "int": "∫", "iint": "∬", "iiint": "∭", "oint": "∮", "sum": "∑", "prod": "∏", "coprod": "∐"}
BIGOPS = {"int", "iint", "iiint", "oint", "sum", "prod", "bigcup", "bigcap", "coprod"}
FUNCS = {"sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "coth", "log", "ln", "lg", "exp", "det", "dim", "ker", "deg", "gcd", "lcm",
         "max", "min", "sup", "inf", "lim", "liminf", "limsup", "Pr", "mod", "arg", "rank", "tr", "Var", "Cov", "softmax", "ReLU", "span", "sgn", "diag", "argmin", "argmax"}
LIMIT_FUNCS = {"lim", "liminf", "limsup", "max", "min", "sup", "inf", "arg", "argmin", "argmax"}
SUP = dict(zip("0123456789+-=()abcdefghijklmnoprstuvwxyzABDEGHIJKLMNOPRTUVW", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻᴬᴮᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᴿᵀᵁⱽᵂ"))
SUP.update({"−": "⁻", "*": "*", "′": "′", "†": "†", "∘": "∘", "θ": "ᶿ", "β": "ᵝ", "γ": "ᵞ", "δ": "ᵟ", "φ": "ᵠ", "χ": "ᵡ"})
SUB = dict(zip("0123456789+-=()aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ"))
SUB.update({"−": "₋", "y": "ᵧ", "β": "ᵦ", "γ": "ᵧ", "ρ": "ᵨ", "φ": "ᵩ", "χ": "ᵪ"})
SUB["γ"] = "ᵧ"
ACCENT = {"hat": "̂", "widehat": "̂", "bar": "̄", "overline": "̅", "tilde": "̃", "widetilde": "̃", "vec": "⃗", "dot": "̇", "ddot": "̈",
          "check": "̌", "acute": "́", "grave": "̀", "underline": "̲", "overrightarrow": "⃗"}
VULGAR = {("1", "2"): "½", ("1", "3"): "⅓", ("2", "3"): "⅔", ("1", "4"): "¼", ("3", "4"): "¾", ("1", "5"): "⅕", ("2", "5"): "⅖", ("3", "5"): "⅗", ("4", "5"): "⅘",
          ("1", "6"): "⅙", ("5", "6"): "⅚", ("1", "8"): "⅛", ("3", "8"): "⅜", ("5", "8"): "⅝", ("7", "8"): "⅞"}
BB = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "𝔸𝔹ℂ𝔻𝔼𝔽𝔾ℍ𝕀𝕁𝕂𝕃𝕄ℕ𝕆ℙℚℝ𝕊𝕋𝕌𝕍𝕎𝕏𝕐ℤ")); BB["1"] = "𝟙"
CAL = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "𝒜ℬ𝒞𝒟ℰℱ𝒢ℋℐ𝒥𝒦ℒℳ𝒩𝒪𝒫𝒬ℛ𝒮𝒯𝒰𝒱𝒲𝒳𝒴𝒵"))
RELS_SET = set(REL.values()) | {"=", "<", ">"}
BIN_SET = set(BIN.values()) - {"·", "∘"} | {"+", "−"}

TOKEN = re.compile(r"\\[a-zA-Z]+|\\.|\s+|[{}^_&]|[^\\{}^_&\s]")


def _bold(c):
    if "a" <= c <= "z": return chr(0x1D41A + ord(c) - 97)
    if "A" <= c <= "Z": return chr(0x1D400 + ord(c) - 65)
    if "0" <= c <= "9": return chr(0x1D7CE + ord(c) - 48)
    return c


def _map_script(s, table):
    out = []
    for c in s:
        if c in table: out.append(table[c])
        elif c in " ,": out.append(c)
        else: return None
    return "".join(out)


class Renderer:
    def __init__(self, src, style="unicode"):
        self.t = TOKEN.findall(src); self.i = 0; self.style = style

    # ---- token helpers
    def peek(self, k=0):
        j = self.i + k
        return self.t[j] if j < len(self.t) else None

    def skip_ws(self):
        while self.peek() is not None and self.peek().isspace(): self.i += 1

    def read_arg(self):
        """Next argument: {group} or a single token/command."""
        self.skip_ws(); tk = self.peek()
        if tk is None: return ""
        if tk == "{":
            self.i += 1; s = self.seq(until_brace=True); return s
        if tk.startswith("\\"):
            return self.atom()
        self.i += 1
        return tk

    def raw_group(self):
        """Read {...} verbatim (for \\text, \\begin names)."""
        self.skip_ws()
        if self.peek() != "{": return self.read_arg()
        self.i += 1; depth = 1; buf = []
        while self.peek() is not None:
            tk = self.peek(); self.i += 1
            if tk == "{": depth += 1
            elif tk == "}":
                depth -= 1
                if depth == 0: break
            buf.append(tk)
        return "".join(buf)

    # ---- scripts
    def script(self, s, table, sym):
        s = re.sub(r"\s+", "", s)
        m = _map_script(s, table) if self.style == "unicode" else None
        if m is not None: return m
        if len(s) == 1: return sym + s
        return f"{sym}({s})"

    def scripts(self, base, bigop=False):
        sub = sup = None
        while True:
            save = self.i; self.skip_ws()
            tk = self.peek()
            if tk == "_" and sub is None: self.i += 1; sub = self.read_arg()
            elif tk == "^" and sup is None: self.i += 1; sup = self.read_arg()
            elif tk == "'":
                self.i += 1; base += "′"
            else:
                self.i = save; break
        if bigop:   # operators keep their limits in-line: ∑_(i=1)^n, ∫₀^π, lim_(n→∞)
            out = base
            if sub is not None: out += self.script(sub, SUB, "_")
            if sup is not None: out += self.script(sup, SUP, "^")
            return out
        if sub is not None: base += self.script(sub, SUB, "_")
        if sup is not None: base += self.script(sup, SUP, "^")
        return base

    # ---- atoms
    def atom(self):
        tk = self.peek(); self.i += 1
        if tk == "{":
            return self.scripts(self.seq(until_brace=True))
        if not tk.startswith("\\"):
            c = {"-": "−", "*": "∗" if False else "*"}.get(tk, tk)
            if c in "=<>+−": return self.scripts(c)
            return self.scripts(c)
        name = tk[1:]
        if name in GREEK: return self.scripts(GREEK[name])
        if name in SYM:
            return self.scripts(SYM[name], bigop=name in BIGOPS)
        if name in REL: return self.scripts(REL[name])
        if name in BIN: return self.scripts(BIN[name])
        if name in FUNCS:
            return self.scripts(name, bigop=name in LIMIT_FUNCS)
        if name in ("left", "right", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "displaystyle", "textstyle", "limits", "nolimits"):
            if name in ("left", "right", "bigl", "bigr", "big", "Big"):
                d = self.peek()
                if d is not None and d.strip():
                    self.i += 1
                    if d == ".": return ""
                    return {"\\{": "{", "\\}": "}", "\\|": "‖", "\\langle": "⟨", "\\rangle": "⟩"}.get(d, d)
            return ""
        if name == "frac" or name in ("dfrac", "tfrac"):
            a, b = self.read_arg(), self.read_arg(); return self.scripts(self.fraction(a, b))
        if name == "binom":
            a, b = self.read_arg(), self.read_arg(); return self.scripts(f"C({a}, {b})")
        if name == "sqrt":
            self.skip_ws(); idx = None
            if self.peek() == "[":
                self.i += 1; buf = []
                while self.peek() not in ("]", None): buf.append(self.peek()); self.i += 1
                self.i += 1; idx = "".join(buf)
            a = self.read_arg(); sym = {"3": "∛", "4": "∜", None: "√"}.get(idx)
            if sym is None: sym = (_map_script(idx, SUP) or f"^({idx})") + "√"
            return self.scripts(sym + (a if self.simple(a) else f"({a})"))
        if name in ACCENT:
            a = self.read_arg(); cm = ACCENT[name]
            if name in ("overline", "underline", "overrightarrow", "widehat", "widetilde") and len(a) > 1:
                return self.scripts("".join(ch + cm for ch in a))
            if len(a) > 1 and name in ("hat", "bar", "tilde", "vec", "dot"):
                return self.scripts(a[:-1] + a[-1] + cm) if len(a) == 1 else self.scripts(f"{a}{cm}")
            return self.scripts(a + cm)
        if name in ("mathbb", "Bbb"):
            a = self.read_arg(); return self.scripts("".join(BB.get(c, c) for c in a))
        if name in ("mathbf", "boldsymbol", "bm", "textbf", "mathbfit"):
            a = self.read_arg(); return self.scripts("".join(_bold(c) for c in a))
        if name in ("mathcal", "mathscr"):
            a = self.read_arg(); return self.scripts("".join(CAL.get(c, c) for c in a))
        if name in ("mathrm", "text", "textrm", "operatorname", "mathsf", "mathit", "mathtt", "textit", "mbox"):
            a = self.raw_group(); return self.scripts(self.sub_render(a) if name in ("mathrm", "operatorname", "mathsf") else a, bigop=a in LIMIT_FUNCS)
        if name == "begin":
            return self.environment(self.raw_group())
        if name in ("end",):
            self.raw_group(); return ""
        if name in ("not",):
            nxt = self.atom(); return nxt + "̸"
        if name in ("overset", "underset", "stackrel"):
            a, b = self.read_arg(), self.read_arg(); return b
        if name in ("phantom", "hphantom", "vphantom", "label", "tag"):
            self.read_arg(); return ""
        if name == "pmod":
            return f" (mod {self.read_arg()})"
        return tk   # unknown: keep verbatim

    def simple(self, s):
        return len(s) <= 1 or bool(re.fullmatch(r"[\w∂∇ℝℕℤℚℂ°′.̂̄̃⃗̇ᵀ⁻¹²³ⁿᵢⱼₓ]+", s)) and len(s) <= 6 and not re.search(r"[+−=<>]", s)

    def fraction(self, a, b):
        if (a, b) in VULGAR: return VULGAR[(a, b)]
        def tight(x):   # operand that reads unambiguously without parentheses
            y = re.sub(r"\s+", "", x)
            if re.search(r"[+−=<>±,/()·×÷]|[A-Za-z]{3,}", y) or not re.fullmatch(r"[\w∂∇π°′\u0300-\u036f\u20d7ᵀ⁻⁰¹²³⁴⁵⁶⁷⁸⁹ⁿᵢⱼₓ₀₁₂₃ℝℕℤℚℂℏ]+", y): return None
            if re.search(r"\d[A-Za-zα-ωΑ-Ω]|[A-Za-zα-ωΑ-Ω]\d", y) and len(y) > 2 and not re.fullmatch(r"[a-zA-Z]\d+", y): return None
            return y
        ta, tb = tight(a), tight(b)
        if re.fullmatch(r"\d+", re.sub(r"\s+", "", a)) and tb and ta is None: ta = None
        a2 = ta if ta is not None else f"({a})"
        b2 = tb if (tb is not None and not (len(tb) > 1 and re.search(r"\d", tb) and re.search(r"[A-Za-zα-ω]", tb) and not re.search(r"[²³¹⁰⁴⁵⁶⁷⁸⁹ⁿᵢⱼₓ₀₁₂₃]", tb))) else f"({b})"
        return f"{a2}/{b2}"

    def environment(self, name):
        body = []; depth = 1
        while self.peek() is not None:
            tk = self.peek()
            if tk == "\\begin": depth += 1
            if tk == "\\end":
                self.i += 1; self.raw_group(); depth -= 1
                if depth == 0: break
                continue
            body.append(tk); self.i += 1
        src = "".join(body)
        rows = [[Renderer(c, self.style).seq().strip() for c in r.split("&")] for r in re.split(r"\\\\", src) if r.strip()]
        if name.startswith("cases"):
            return "{ " + "; ".join("  ".join(r) for r in rows) + " }"
        l, r = {"pmatrix": ("(", ")"), "bmatrix": ("[", "]"), "vmatrix": ("|", "|"), "Vmatrix": ("‖", "‖"), "matrix": ("", "")}.get(name, ("(", ")"))
        return l + "; ".join(" ".join(r_) for r_ in rows) + r

    def sub_render(self, s):
        return Renderer(s, self.style).seq().strip() if re.search(r"[\\^_{]", s) else s

    # ---- sequence
    def seq(self, until_brace=False):
        out = []; pend_space = False; prev_big = False
        while self.peek() is not None:
            tk = self.peek()
            if tk == "}":
                if until_brace: self.i += 1; break
                self.i += 1; continue
            if tk.isspace():
                self.i += 1
                # keep optional spaces (e.g. between factors) but drop the syntactic one after a command like \alpha / \lambda
                prev = self.t[self.i - 2] if self.i >= 2 else ""
                if not (prev.startswith("\\") and prev[1:].isalpha() and prev[1:] in GREEK | SYM | REL | BIN) or prev[1:] in REL | BIN:
                    pend_space = True
                continue
            if tk == "&": self.i += 1; out.append("  "); continue
            if tk in ("^", "_"):                       # script with empty base
                out.append(self.scripts("")); continue
            was_big = tk.startswith("\\") and tk[1:] in BIGOPS | LIMIT_FUNCS
            a = self.atom()
            if a == "": continue
            if (was_big and out and out[-1][-1:].isalnum()) or (prev_big and (a[:1].isalnum() or a[:1] in "([")):
                if out and not out[-1].endswith(" "): out.append(" ")
            prev_big = was_big
            if pend_space and out and not out[-1].endswith(" ") and not a.startswith(" "):
                out.append(" ")
            pend_space = False
            core = a.strip()
            if core in RELS_SET or (core in BIN_SET and out and out[-1].strip() not in ("", "(", "[", "{") and not (core in ("−", "+") and self.unary_context(out))):
                a = f" {core} "
            elif core in ("−", "+") and self.unary_context(out):
                a = core
            elif core in ("·", "×", "÷", "∘") and True:
                a = f" {core} "
            out.append(a)
        s = "".join(out)
        s = re.sub(r"[ \t]{2,}", " ", s).replace("( ", "(").replace(" )", ")").replace(" ,", ",")
        return s

    @staticmethod
    def unary_context(out):
        prev = "".join(out).rstrip()
        return prev == "" or prev[-1] in "(=<>≤≥[{,;:" or prev[-1] in "+−±×·∈→" or prev[-1] in RELS_SET


def latex_to_text(latex, style="unicode"):
    """Convert a LaTeX math string to readable Unicode text. style='unicode' (default) or 'plain' (^ and _ instead of super/subscript glyphs)."""
    if not latex:
        return ""
    s = latex.strip().strip("$")
    try:
        s = Renderer(s, style).seq().strip()
    except Exception:
        return latex
    s = s.replace("-", "−") if "−" not in s and re.search(r"\w-\w", s) else s
    s = re.sub(r"\s+([,;.])", r"\1", s)
    return unicodedata.normalize("NFC", s)


def remaining_commands(text):
    """Backslash commands left unconverted (for diagnostics/tests)."""
    return re.findall(r"\\[a-zA-Z]+", text)
