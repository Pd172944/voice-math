"""Structured spoken-math -> LaTeX parser (recursive descent over a token list)."""
import re
from .lexicon import GREEK, GREEK_UPPER, SYMBOLS, FUNCTIONS, NUMBERS, BLACKBOARD
from .rules import is_number_token

RELS = {"=", "<", ">", r"\leq", r"\geq", r"\neq", r"\approx", r"\in", r"\notin", r"\subset", r"\subseteq", r"\supset", r"\sim", r"\to",
        r"\mapsto", r"\implies", r"\iff", r"\propto", r"\mid"}
OPS = {"+", "-", r"\pm", r"\cup", r"\cap", r"\setminus", r"\div", r"\times", r"\cdot"}
PHRASES = {tuple(k.split()): v for k, v in SYMBOLS.items()}
FUNC_PHRASES = {tuple(k.split()): v for k, v in FUNCTIONS.items()}
REL_STOP = [("equals",), ("is", "equal"), ("equal", "to"), ("is", "less"), ("is", "greater"), ("less", "than"), ("greater", "than"),
            ("is", "approximately"), ("approximately",), ("is", "not"), ("does", "not"), ("is", "in"), ("is", "an"), ("is", "a")]
FRAC_WORDS = {"half": 2, "halves": 2, "third": 3, "thirds": 3, "fourth": 4, "fourths": 4, "quarter": 4, "quarters": 4, "fifth": 5, "fifths": 5,
              "eighth": 8, "eighths": 8}
EXP_WORDS = {"second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6"}
FILLER = {"the", "an", "is", "that", "equal", "than", "with", "then", "so", "we", "have", "this", "it", "all"}
OPEN = {("open", "paren"), ("left", "paren"), ("open", "parenthesis"), ("left", "parenthesis"), ("open", "bracket")}
CLOSE = {("close", "paren"), ("right", "paren"), ("close", "parenthesis"), ("right", "parenthesis"), ("close", "bracket")}


def num_value(toks, i):
    """Parse a number starting at toks[i]; returns (string, next_i) or None."""
    if i >= len(toks):
        return None
    w = toks[i].lower()
    if re.fullmatch(r"\d+(\.\d+)?", w):
        return w, i + 1
    if w not in NUMBERS:
        return None
    val = int(NUMBERS[w]); j = i + 1
    if val in (20, 30, 40, 50, 60, 70, 80, 90) and j < len(toks) and toks[j].lower() in NUMBERS and 1 <= int(NUMBERS[toks[j].lower()]) <= 9:
        val += int(NUMBERS[toks[j].lower()]); j += 1
    elif val == 100 and False:
        pass
    s = str(val)
    if j < len(toks) and toks[j].lower() == "point":
        k = j + 1; digs = ""
        while k < len(toks) and toks[k].lower() in NUMBERS:
            digs += NUMBERS[toks[k].lower()]; k += 1
        while k < len(toks) and re.fullmatch(r"\d", toks[k]):
            digs += toks[k]; k += 1
        if digs:
            return s + "." + digs, k
    if w == "one" and j < len(toks) and toks[j].lower() == "hundred":
        return "100", j + 1
    return s, j


SP = "\u2423"  # sentinel for "keep this space" (around relations / binary operators)


def tidy(s):
    """Compact juxtaposition (x y -> xy) but keep spaces after bare commands and around relations/operators."""
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(?<=[^\\A-Za-z])(\\[a-zA-Z]+) (?=[A-Za-z0-9])", lambda m: m.group(1) + SP, s)   # cmd + letter
    s = re.sub(r"^(\\[a-zA-Z]+) (?=[A-Za-z0-9])", lambda m: m.group(1) + SP, s)
    s = re.sub(r"(\\[a-zA-Z]+) (?=[A-Za-z0-9])", lambda m: m.group(1) + SP, s)
    s = s.replace(" ", "")
    return s.replace(SP, " ")


class Parser:
    def __init__(self, toks):
        self.t = toks
        self.i = 0
        self.unknown = []
        self.recognised = 0
        self.after_from = False
        self.last_rel = None

    # -- token helpers
    def low(self, k=0):
        j = self.i + k
        return self.t[j].lower() if j < len(self.t) else None

    def raw(self, k=0):
        j = self.i + k
        return self.t[j] if j < len(self.t) else None

    def end(self):
        return self.i >= len(self.t)

    def starts(self, phrase, k=0):
        return all(self.low(k + n) == w for n, w in enumerate(phrase))

    def at_any(self, phrases):
        return any(self.starts(p) for p in phrases)

    def match_table(self, table):
        best = None
        for p in table:
            if self.starts(p) and (best is None or len(p) > len(best)):
                best = p
        return best

    def is_single(self, k=0):
        w = self.raw(k)
        return w is not None and len(w) == 1 and w.isalpha()

    # -- main
    def parse_seq(self, stops=(), stop_rel=False):
        items = []  # (kind, tex)
        while not self.end():
            if self.at_any(stops) or (stop_rel and self.at_any(REL_STOP)):
                break
            if self.at_any(CLOSE):
                break
            n0 = self.i
            lw = self.low()
            if self.starts(("normally", "distributed")) or self.starts(("distributed", "as", "normal")):
                items.extend(self.normal_dist()); continue
            if lw == "from" and items and items[-1][0] == "var" and self.until([("to",)]) > 0:
                self.i += 1; items.append(("rel", ":")); self.after_from = True; continue
            if lw == "to" and self.after_from:
                self.i += 1; items.append(("rel", r"\to")); self.after_from = False; continue
            if lw == "over" and items:
                self.i += 1
                den = self.parse_term(stop_rel=True)
                if den is None:
                    break
                dtex = den[1]
                while not self.end() and self.at_start_of_term():
                    nx = self.parse_term()
                    if nx is None: break
                    dtex += " " + nx[1]
                nums = []
                while items and items[-1][0] in ("term", "var", "num", "paren"):
                    nums.insert(0, items.pop()[1])
                num = " ".join(nums)
                items.append(("term", rf"\frac{{{self.strip_paren(num) if len(nums) == 1 else num}}}{{{self.strip_paren(dtex) if ' ' not in dtex else dtex}}}"))
                self.recognised += 1
                continue
            if lw == "over":
                self.i += 1; continue
            # relation / operator phrases
            p = self.match_table(PHRASES) if lw != "partial" else None
            if p is None and lw in FILLER | {"a", "of", "and", "comma"}:
                if lw == "a" and self.low(1) not in ("subset", "superset", "member", "set"):
                    pass
                else:
                    self.i += 1; continue
            if p and self.phrase_is_operator(p, items):
                tex = PHRASES[p]
                self.i += len(p); self.recognised += len(p)
                if tex in RELS:
                    items.append(("rel", tex))
                elif tex in OPS:
                    if tex == r"\cdot" and p == ("times",):
                        items.append(("times", ""))
                    else:
                        items.append(("op", tex))
                else:
                    k2, t2 = self.postfix(("var", tex))
                    items.append((k2, t2))
                if tex in RELS: self.last_rel = tex
                continue
            if lw in ("minus", "negative") and not items:
                self.i += 1; items.append(("op", "-")); continue
            term = self.parse_term(stop_rel=stop_rel)
            if term is None:
                self.unknown.append(self.raw()); self.i += 1
                items.append(("term", r"\text{" + self.t[self.i - 1] + "}"))
                continue
            items.append(term)
            if self.i == n0:
                self.i += 1
        return self.join(items)

    def phrase_is_operator(self, p, items):
        tex = PHRASES[p]
        if p in (("dot",), ("cross",), ("del",), ("in",), ("union",), ("prime",), ("tilde",), ("dagger",)):
            return True
        return True

    def strip_paren(self, s):
        if s.startswith("(") and s.endswith(")"):
            d = 0
            for n, c in enumerate(s):
                d += (c == "(") - (c == ")")
                if d == 0 and n < len(s) - 1:
                    return s
            return s[1:-1]
        return s

    def join(self, items):
        out = ""
        prev = None
        for kind, tex in items:
            if kind in ("rel", "op"):
                if kind == "op" and tex == "-" and (prev is None or prev in ("rel", "op")):
                    out += "-"      # unary minus
                else:
                    out += f"{SP}{tex}{SP}"
            elif kind == "times":
                if prev == "num":
                    out += r" \times "
            else:
                if prev in ("term", "var", "num", "paren") and out and out[-1].isalpha() and tex[:1].isalnum() and False:
                    out += " "
                out += (" " if out and prev in ("term", "var", "num", "paren", "times") else "") + tex
            prev = kind if kind != "times" else prev
        return tidy(out).replace("  ", " ").strip()

    # -- terms
    def parse_term(self, stop_rel=False):
        lw = self.low()
        if lw is None:
            return None
        while lw in ("the", "an") and self.i + 1 < len(self.t):
            self.i += 1; lw = self.low()
        t = self.prefix_construct()
        if t is None:
            t = self.atom()
        if t is None:
            return None
        return self.postfix(t)

    def atom(self):
        w = self.raw(); lw = self.low()
        if self.at_any(OPEN):
            self.i += 2
            inner = self.parse_seq()
            if self.at_any(CLOSE):
                self.i += 2
            return ("paren", f"({inner})")
        nv = num_value(self.t, self.i)
        if nv:
            s, j = nv
            # fractions: one half, two thirds
            if j < len(self.t) and self.t[j].lower() in FRAC_WORDS and re.fullmatch(r"\d+", s) and not (s == "1" and self.t[j].lower() in ("fourth",) and False):
                den = FRAC_WORDS[self.t[j].lower()]
                self.i = j + 1; self.recognised += 2
                return ("term", rf"\frac{{{s}}}{{{den}}}")
            self.i = j; self.recognised += 1
            return ("num", s)
        if lw in ("capital", "uppercase") and self.low(1) in GREEK_UPPER:
            self.i += 2; self.recognised += 2
            g = self.low(-1)
            return ("var", "\\" + g.capitalize())
        if lw in ("bold", "boldface") and self.is_single(1):
            self.i += 2; self.recognised += 2
            return ("var", rf"\mathbf{{{self.raw(-1)}}}")
        if lw in ("vector",) and self.i + 1 < len(self.t) and (self.is_single(1) or self.low(1) in GREEK):
            self.i += 1
            a = self.atom(); self.recognised += 1
            return ("var", rf"\vec{{{a[1]}}}")
        if lw == "pie" or lw == "pi":
            self.i += 1; self.recognised += 1
            return ("var", r"\pi")
        if lw in GREEK and not (lw == "mu" and False):
            self.i += 1; self.recognised += 1
            return ("var", "\\" + lw)
        p = self.match_table(FUNC_PHRASES)
        if p:
            self.i += len(p); self.recognised += len(p)
            return ("func", FUNC_PHRASES[p])
        if w in BLACKBOARD.values() or (len(w) == 1 and w in "RNZQC"):
            nxt = self.low(1)
            if len(w) == 1 and w in "RNZQC":
                if nxt in ("n", "m", "d", "k") or (nxt in NUMBERS and False):
                    self.i += 2; self.recognised += 2
                    exp = nxt
                    if self.low() == "by" and self.is_single(1):
                        exp += r" \times " + self.raw(1); self.i += 2
                    exp = exp if len(exp) == 1 else "{" + exp + "}"
                    return ("var", rf"\mathbb{{{w}}}^{exp}")
                if self.last_rel in (r"\in", r"\subset", r"\subseteq", r"\notin", r"\supset") or (nxt == "is" and self.low(2) in ("a", "an", "in")):
                    self.i += 1; self.recognised += 1
                    return ("var", rf"\mathbb{{{w}}}")
                if nxt == "squared" or nxt == "cubed":
                    self.i += 2
                    return ("var", rf"\mathbb{{{w}}}^{2 if nxt == 'squared' else 3}")
        if len(w) == 1 and w.isalpha():
            self.i += 1; self.recognised += 1
            return ("var", w)
        if re.fullmatch(r"[A-Za-z]+\d*", w) and w.upper() == w and len(w) <= 4 and len(w) > 1:
            self.i += 1; return ("var", w)
        return None

    def sub_spec(self):
        """After 'sub': read the subscript."""
        toks = []
        nv = num_value(self.t, self.i)
        if nv:
            toks.append(nv[0]); self.i = nv[1]
        elif self.low() in GREEK:
            toks.append("\\" + self.low()); self.i += 1
        elif self.is_single():
            toks.append(self.raw()); self.i += 1
        else:
            return None
        if self.low() in ("plus", "minus") and self.low(1) in NUMBERS and self.low(1) == "one" and \
                (self.i + 2 >= len(self.t) or self.low(2) in ("equals", "of", "minus", "is", "plus", "comma", "over", "close", "right", "and")):
            sign = "+" if self.low() == "plus" else "-"
            toks = [toks[0] + sign + "1"]; self.i += 2
        elif self.is_single() and len(toks) == 1 and self.low(1) not in ("sub", "squared", "cubed", "bar", "hat", "tilde", "to", "prime", "inverse", "transpose", "over", "of", "factorial") \
                and self.raw() not in ("a",) and toks[0].isalpha():
            toks.append(self.raw()); self.i += 1
        if len(toks) == 1:
            return toks[0] if len(toks[0]) == 1 or toks[0].startswith("\\") else "{" + toks[0] + "}"
        return "{" + ",".join(toks) + "}" if "+" not in toks[0] and "-" not in toks[0] else "{" + toks[0] + "}"

    def exponent(self):
        neg = ""
        if self.low() == "minus" or self.low() == "negative":
            neg = "-"; self.i += 1
        nv = num_value(self.t, self.i)
        if nv:
            self.i = nv[1]; e = nv[0]
        elif self.low() in EXP_WORDS:
            e = EXP_WORDS[self.low()]; self.i += 1
        elif self.low() in GREEK:
            e = "\\" + self.low(); self.i += 1
        elif self.is_single():
            e = self.raw(); self.i += 1
        else:
            return None
        if self.low() == "sub" and len(e) == 1:
            self.i += 1
            sp = self.sub_spec()
            if sp: e += "_" + sp
        if self.low() == "squared":
            e += "^2"; self.i += 1
        if neg:
            return "{-" + e + "}"
        return e if len(e) == 1 or e.startswith("\\") and e[1:].isalpha() else "{" + e + "}"

    def postfix(self, t):
        kind, tex = t
        while not self.end():
            lw = self.low()
            if kind == "func":
                # function application: "sine of x" / "sine x"
                if lw == "of":
                    self.i += 1
                    arg = self.parse_term()
                    if arg is None: break
                    args = [arg[1]]
                    while self.low() == "and":
                        self.i += 1
                        a2 = self.parse_term()
                        if a2 is None: break
                        args.append(a2[1])
                    tex = f"{tex}({','.join(self.strip_paren(a) for a in args)})"
                elif self.at_start_of_term():
                    arg = self.parse_term()
                    if arg is None: break
                    tex = f"{tex} {arg[1]}"
                kind = "term"
                continue
            if lw == "squared":
                tex += "^2"; self.i += 1; kind = "term"; self.recognised += 1
            elif lw == "cubed":
                tex += "^3"; self.i += 1; kind = "term"; self.recognised += 1
            elif lw in ("to", "raised") and (self.low(1) == "the" or self.low(1) in ("a",)) or (lw == "raised" and self.low(1) == "to"):
                j = self.i
                self.i += 2 if lw == "to" else 3
                if self.low() == "power" and self.low(1) == "of":
                    self.i += 2
                elif self.low() == "the":
                    self.i += 1
                e = self.exponent()
                if e is None:
                    self.i = j; break
                tex += "^" + e; kind = "term"; self.recognised += 2
            elif lw == "to" and self.low(1) == "minus":
                break
            elif lw == "inverse":
                tex += "^{-1}"; self.i += 1; kind = "term"
            elif lw == "transpose":
                tex += "^T"; self.i += 1; kind = "term"
            elif lw == "prime":
                tex += "'"; self.i += 1
            elif lw == "double" and self.low(1) == "prime":
                tex += "''"; self.i += 2
            elif lw in ("bar", "hat", "tilde", "closure", "overline", "dot") and not (lw == "dot" and self.is_single(1) is False and False) and kind in ("var", "term", "num"):
                if lw == "dot":
                    break
                cmd = {"bar": r"\bar", "hat": r"\hat", "tilde": r"\tilde", "closure": r"\overline", "overline": r"\overline"}[lw]
                base = tex if kind == "var" and "_" not in tex and "^" not in tex else tex
                # hat/bar applies to the base symbol, subscript stays outside
                m = re.fullmatch(r"(\\[a-zA-Z]+|[A-Za-z])(_.*)?", tex)
                if m and m.group(2):
                    tex = f"{cmd}{{{m.group(1)}}}{m.group(2)}"
                else:
                    tex = f"{cmd}{{{tex}}}"
                self.i += 1; kind = "var"
            elif lw == "factorial":
                tex += "!"; self.i += 1; kind = "term"
            elif lw == "sub" and kind in ("var", "term", "func"):
                self.i += 1
                s = self.sub_spec()
                if s is None:
                    break
                if "^" in tex and False:
                    pass
                tex += "_" + s; kind = "var"; self.recognised += 1
            elif lw == "of" and kind == "var" and (self.is_func_like(tex)):
                self.i += 1
                arg = self.parse_term()
                if arg is None:
                    break
                args = [arg]
                if self.low() == "over":
                    self.i += 1
                    d = self.parse_term()
                    if d:
                        args = [("term", rf"\frac{{{self.strip_paren(arg[1])}}}{{{self.strip_paren(d[1])}}}")]
                a_tex = [a[1] for a in args]
                while self.low() == "and":
                    self.i += 1
                    a2 = self.parse_term()
                    if a2 is None: break
                    a_tex.append(a2[1])
                tex = f"{tex}({','.join(self.strip_paren(a) if len(a_tex) == 1 else a for a in a_tex)})".replace("),(", ",")
                kind = "term"; self.recognised += 1
            else:
                break
        return (kind, tex)

    def is_func_like(self, tex):
        base = re.sub(r"_.*$", "", tex)
        return bool(re.fullmatch(r"[A-Za-z]|\\(ell|sigma|psi|phi|rho|mu|theta|lambda|omega|alpha|beta|eta|tau)|[A-Z]{2,}", base)) and not tex.endswith("'")  or tex.endswith("'")

    def at_start_of_term(self):
        lw = self.low()
        if lw is None or lw in ("equals", "over", "plus", "minus", "times", "squared", "of", "and"):
            return False
        if self.at_any(REL_STOP) or self.at_any(CLOSE):
            return False
        return self.is_single() or lw in GREEK or is_number_token(lw) or lw in ("pi", "pie")

    # -- prefix constructs
    def until(self, words):
        """Find index of first occurrence of any word sequence (from self.i), or -1."""
        for j in range(self.i, len(self.t)):
            for w in words:
                if all(j + n < len(self.t) and self.t[j + n].lower() == x for n, x in enumerate(w)):
                    return j
        return -1

    def sub_parse(self, toks):
        p = Parser(toks)
        s = p.parse_seq()
        self.recognised += p.recognised; self.unknown += p.unknown
        return s

    def bound(self, stop_words):
        j = self.until(stop_words)
        if j < 0:
            return None
        sub = self.t[self.i:j]
        self.i = j
        return self.sub_parse(sub)

    def prefix_construct(self):
        lw = self.low()
        # integral
        if lw in ("closure", "boundary") and self.low(1) == "of":
            self.i += 2
            a = self.parse_term()
            return ("term", rf"\overline{{{a[1]}}}" if lw == "closure" else rf"\partial {a[1]}")
        if lw in ("integral", "integrals"):
            self.i += 1; self.recognised += 1
            lo = hi = None
            if self.low() == "from":
                self.i += 1
                lo = self.bound([("to",)])
                self.i += 1
                j = self.until([("of",), ("d",)])
                hi = self.sub_parse(self.t[self.i:j]) if j >= 0 else None
                if j >= 0: self.i = j
            if self.low() == "of":
                self.i += 1
            # body until "d <letter>"
            j = None
            for k in range(self.i, len(self.t)):
                w = self.t[k].lower()
                if w == "d" and k + 1 < len(self.t) and len(self.t[k + 1]) == 1:
                    j = k; break
                if re.fullmatch(r"d[a-z]", w):
                    j = k; break
            if j is None:
                body = self.sub_parse(self.t[self.i:]); self.i = len(self.t); var = "x"
            else:
                body = self.sub_parse(self.t[self.i:j]); self.i = j
                if self.t[j].lower() == "d":
                    var = self.t[j + 1]; self.i = j + 2
                else:
                    var = self.t[j][1]; self.i = j + 1
            sc = lambda b: b if len(b) == 1 or (b.startswith("\\") and b[1:].isalpha()) else "{" + b + "}"
            s = r"\int"
            if lo is not None: s += "_" + sc(lo)
            if hi is not None: s += "^" + sc(hi)
            return ("term", f"{s} {body}\\,d{var}")
        if lw in ("sum", "summation", "product") and (self.low(1) in ("from", "over", "of") or True):
            cmd = r"\prod" if lw == "product" else r"\sum"
            self.i += 1; self.recognised += 1
            if self.low() == "from":
                self.i += 1
                lo = self.bound([("to",)]); self.i += 1
                j = self.until([("of",)])
                hi = self.sub_parse(self.t[self.i:j]) if j >= 0 else ""
                if j >= 0: self.i = j + 1
                lo_s = re.sub(r"\s*=\s*", "=", lo)
                body = self.parse_seq(stop_rel=True)
                sc = lambda b: b if len(b) == 1 or (b.startswith("\\") and b[1:].isalpha()) else "{" + b + "}"
                return ("term", f"{cmd}_{{{lo_s}}}^{sc(hi)} {body}")
            if self.low() == "over":
                self.i += 1
                v = self.sub_parse([self.raw()]); self.i += 1
                if self.low() == "of": self.i += 1
                body = self.parse_seq(stop_rel=True)
                return ("term", f"{cmd}_{v} {body}")
            return ("term", cmd)
        if lw in ("limit", "lim"):
            self.i += 1; self.recognised += 1
            if self.low() == "as":
                self.i += 1
                j = self.until([("of",)])
                if j >= 0:
                    inner = self.t[self.i:j]
                    low = [w.lower() for w in inner]
                    for g in (["goes", "to"], ["tends", "to"], ["approaches"]):
                        for n in range(len(low)):
                            if low[n:n + len(g)] == g:
                                v = self.sub_parse(inner[:n]); a = self.sub_parse(inner[n + len(g):])
                                self.i = j + 1
                                body = self.parse_seq(stop_rel=True)
                                return ("term", rf"\lim_{{{v} \to {a}}} {body}")
            return ("term", r"\lim")
        if lw in ("derivative",) or (lw in ("second",) and self.low(1) in ("derivative", "partial")) or \
                (lw == "partial" and self.low(1) == "derivative") or (lw == "second" and self.low(1) == "partial"):
            return self.derivative_construct()
        if lw == "partial":
            return self.partial_construct()
        if lw == "d" and self.deriv_d_pattern():
            return self.d_pattern()
        if lw == "square" and self.low(1) == "root":
            self.i += 2
            if self.low() == "of": self.i += 1
            a = self.parse_term(stop_rel=True)
            if a is None: return ("term", r"\sqrt{}")
            self.recognised += 2
            return ("term", rf"\sqrt{{{self.strip_paren(a[1])}}}")
        if lw == "absolute" and self.low(1) == "value":
            self.i += 2
            if self.low() == "of": self.i += 1
            inner = self.parse_seq(stop_rel=True)
            self.recognised += 2
            return ("term", f"|{inner}|")
        if lw in ("norm",) or (self.low(1) == "norm" and (self.low() in NUMBERS or self.is_single() or self.low() in ("infinity", "two", "one"))):
            if self.low(1) == "norm":
                p = self.low(); self.i += 2
            else:
                p = None; self.i += 1
            ptex = None
            if p is not None:
                nv = num_value([p], 0)
                ptex = nv[0] if nv else (r"\infty" if p == "infinity" else p)
            if self.low() == "of":
                self.i += 1
                a = self.parse_term()
                inner = a[1] if a else ""
                sub = f"_{ptex}" if ptex else ""
                if ptex and len(ptex) > 1 and not ptex.startswith("\\"):
                    sub = "_{" + ptex + "}"
                return ("term", rf"\|{inner}\|{sub}")
            return ("var", rf"\ell_{ptex}" if ptex else r"\|")
        if lw in ("determinant", "det", "trace") and self.low(1) == "of":
            f = r"\det" if lw != "trace" else r"\operatorname{tr}"
            self.i += 2
            a = self.parse_seq(stop_rel=True)
            return ("term", f"{f}({self.strip_paren(a)})")
        if lw == "inner" and self.low(1) == "product":
            self.i += 3 if self.low(2) == "of" else 2
            a = self.parse_term(); 
            if self.low() == "and": self.i += 1
            b = self.parse_term()
            return ("term", rf"\langle {a[1]}, {b[1] if b else ''} \rangle")
        if (lw == "probability" and self.low(1) in ("that", "of")) or (self.raw() == "P" and self.low(1) == "of"):
            self.i += 2
            return ("term", f"P({self.cond_body()})")
        if lw in ("expectation", "expected") and self.low(1) in ("of", "value", "over"):
            return self.expectation()
        if self.raw() == "E" and self.low(1) == "of":
            self.i += 2
            return ("term", f"E[{self.sub_parse(self.t[self.i:self.consume_all()])}]")
        if lw == "conditional" and self.low(1) in ("expectation",):
            self.i += 3 if self.low(2) == "of" else 2
            return ("term", rf"\mathbb{{E}}[{self.cond_body()}]")
        if lw in ("variance", "var") and self.low(1) == "of":
            self.i += 2
            a = self.parse_term()
            return ("term", rf"\operatorname{{Var}}({a[1] if a else ''})")
        if lw == "covariance" and self.low(1) == "of":
            self.i += 2
            a = self.parse_term()
            if self.low() == "and": self.i += 1
            b = self.parse_term()
            return ("term", rf"\operatorname{{Cov}}({a[1]}, {b[1] if b else ''})")
        if lw == "gradient" and self.low(1) == "of":
            self.i += 2
            j = self.until([("with", "respect")])
            if j >= 0:
                body = self.sub_parse(self.t[self.i:j]); self.i = j + 3
                if self.low() == "to": self.i += 1
                v = self.parse_term()
                return ("term", rf"\nabla_{v[1]} {body}")
        if lw == "kl" and self.low(1) == "divergence":
            self.i += 3 if self.low(2) == "of" else 2
            a = self.parse_term()
            if self.low() == "and": self.i += 1
            b = self.parse_term()
            return ("term", rf"D_{{KL}}({a[1]} \| {b[1] if b else ''})")
        if lw == "log" and self.low(1) == "base":
            self.i += 2
            b = self.atom()
            if self.low() == "of": self.i += 1
            a = self.parse_term()
            bs = b[1] if len(b[1]) == 1 else "{" + b[1] + "}"
            return ("term", rf"\log_{bs} {a[1] if a else ''}")
        if lw == "big" and self.low(1) in ("o", "oh") and self.low(2) == "of":
            self.i += 3
            body = self.parse_seq(stop_rel=True)
            return ("term", f"O({body})")
        if lw in ("supremum", "infimum", "sup", "inf", "arg", "argmin", "argmax") and (self.low(1) == "over" or self.low(1) in ("min", "max")):
            f = {"supremum": r"\sup", "sup": r"\sup", "infimum": r"\inf", "inf": r"\inf"}.get(lw)
            self.i += 1
            if f is None:
                f = r"\arg\min" if self.low() == "min" or lw == "argmin" else r"\arg\max"
                if self.low() in ("min", "max"): self.i += 1
            if self.low() == "over": self.i += 1
            j = self.until([("of",)])
            sub = self.sub_parse(self.t[self.i:j]) if j >= 0 else ""
            if j >= 0: self.i = j + 1
            body = self.parse_seq(stop_rel=True)
            return ("term", f"{f}_{{{sub}}} {body}" if " " in sub or "\\in" in sub else f"{f}_{sub} {body}")
        if lw == "kl":
            pass
        return None

    def normal_dist(self):
        if self.low() == "normally":
            self.i += 2
            if self.low() == "with": self.i += 1
            if self.low() == "mean": self.i += 1
            j = self.until([("and", "variance"), ("variance",)])
            if j < 0:
                m = self.parse_term(); v = None
            else:
                m = ("t", self.sub_parse(self.t[self.i:j])); self.i = j
                self.i += 2 if self.low() == "and" else 1
                v = self.parse_term()
        else:
            self.i += 3
            m = self.parse_term(); v = self.parse_term()
        return [("rel", r"\sim"), ("term", rf"\mathcal{{N}}({m[1]}, {v[1] if v else ''})")]

    def consume_all(self):
        """Consume tokens up to a relation (or end) and return the end index."""
        j = self.i
        while j < len(self.t):
            self.i = j
            if self.at_any(REL_STOP): break
            j += 1
        e = j
        self.i = e
        return e

    def cond_body(self):
        j = self.i
        # whole remainder, split at 'given'
        rest = self.t[self.i:]
        low = [w.lower() for w in rest]
        self.i = len(self.t)
        if "given" in low:
            n = low.index("given")
            return f"{self.sub_parse(rest[:n])} \\mid {self.sub_parse(rest[n + 1:])}"
        return self.sub_parse(rest)

    def expectation(self):
        lw = self.low()
        self.i += 1
        if self.low() == "value": self.i += 1
        if self.low() == "over":
            # expectation over x distributed as p of x of f of x
            self.i += 1
            j = self.until([("distributed", "as")])
            v = self.sub_parse(self.t[self.i:j]); self.i = j + 2
            k = self.until([("of",)])
            # distribution may be 'p of x' -> find 2nd 'of'
            k2 = self.until([("of", "f"), ("of", "l")]) if False else None
            low = [w.lower() for w in self.t]
            ofs = [n for n in range(self.i, len(self.t)) if low[n] == "of"]
            if len(ofs) >= 2:
                dist = self.sub_parse(self.t[self.i:ofs[1]]); self.i = ofs[1] + 1
                body = self.sub_parse(self.t[self.i:]); self.i = len(self.t)
                return ("term", rf"\mathbb{{E}}_{{{v} \sim {dist}}}[{body}]")
        if self.low() == "of":
            self.i += 1
        low = [w.lower() for w in self.t]
        # "of BODY where V is drawn from DIST"
        if "where" in low[self.i:]:
            w = low.index("where", self.i)
            body = self.sub_parse(self.t[self.i:w])
            rest = self.t[w + 1:]
            rl = [x.lower() for x in rest]
            if "drawn" in rl and "from" in rl:
                f = rl.index("from")
                v = self.sub_parse(rest[:rl.index("is") if "is" in rl else 1])
                dist = self.sub_parse(rest[f + 1:])
                self.i = len(self.t)
                return ("term", rf"\mathbb{{E}}_{{{v} \sim {dist}}}[{body}]")
        if "given" in low[self.i:]:
            return ("term", rf"\mathbb{{E}}[{self.cond_body()}]")
        a = self.parse_term()
        return ("term", f"E[{a[1] if a else ''}]")

    def derivative_construct(self):
        second = False
        if self.low() == "second":
            second = True; self.i += 1
        part = False
        if self.low() == "partial":
            part = True; self.i += 1
        self.i += 1  # 'derivative'
        if self.low() == "of": self.i += 1
        j = self.until([("with", "respect")])
        if j < 0:
            return ("term", r"\partial" if part else "d")
        body = self.sub_parse(self.t[self.i:j]); self.i = j + 2
        if self.low() == "to": self.i += 1
        v = self.raw(); self.i += 1
        d = r"\partial" if part else "d"
        if second:
            if len(body) == 1:
                return ("term", rf"\frac{{{d}^2 {body}}}{{{d} {v}^2}}")
            return ("term", rf"\frac{{{d}^2}}{{{d} {v}^2}} {body}")
        if len(body) == 1 or re.fullmatch(r"\\[a-zA-Z]+", body):
            return ("term", rf"\frac{{{d} {body}}}{{{d} {v}}}")
        return ("term", rf"\frac{{{d}}}{{{d} {v}}} {body}")

    def deriv_d_pattern(self):
        l = [self.low(k) for k in range(8)]
        if l[1] == "squared" and len(self.t) > self.i + 4:
            return self.is_single(2) and (l[3] == "d" or (l[3] == "over" and l[4] == "d"))
        return self.is_single(1) and (l[2] == "d" or (l[2] == "over" and l[3] == "d")) and self.is_single(3 if l[2] == "d" else 4)

    def d_pattern(self):
        if self.low(1) == "squared":
            y = self.raw(2); self.i += 3
            if self.low() == "over": self.i += 1
            self.i += 1; x = self.raw(); self.i += 1
            if self.low() == "squared": self.i += 1
            return ("term", rf"\frac{{d^2 {y}}}{{d {x}^2}}")
        y = self.raw(1); self.i += 2
        if self.low() == "over": self.i += 1
        self.i += 1; x = self.raw(); self.i += 1
        return ("term", rf"\frac{{d {y}}}{{d {x}}}")

    def partial_construct(self):
        # partial [squared] f [over] partial x [squared | partial y]  |  bare partial A
        j = self.i + 1
        sq = False
        if self.low(1) == "squared":
            sq = True; j += 1
        if j < len(self.t) and (self.is_single(j - self.i) or self.low(j - self.i) in GREEK):
            f = self.t[j]; k = j + 1
            if k < len(self.t) and self.t[k].lower() == "over": k += 1
            if k < len(self.t) and self.t[k].lower() == "partial" and k + 1 < len(self.t):
                x = self.t[k + 1]; k += 2
                xs = rf"\partial {x}"
                if k < len(self.t) and self.t[k].lower() == "squared":
                    k += 1; xs = rf"\partial {x}^2"
                elif k + 1 < len(self.t) and self.t[k].lower() == "partial" and len(self.t[k + 1]) == 1:
                    xs += rf" \partial {self.t[k + 1]}"; k += 2
                self.i = k; self.recognised += 3
                if f.lower() in GREEK: f = "\\" + f.lower()
                num = rf"\partial^2 {f}" if sq else rf"\partial {f}"
                return ("term", rf"\frac{{{num}}}{{{xs}}}")
        self.i += 1
        return ("var", r"\partial")
