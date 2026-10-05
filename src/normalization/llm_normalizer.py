"""LLM fallback normalizer via OpenRouter (optional; only used if OPENROUTER_API_KEY is set).
The prompt forbids inventing content; outputs are validated by deterministic filters in canonicalize.py."""
import json, os, urllib.request

PROMPT_VERSION = "v1"
SYSTEM = ("You convert a spoken-mathematics transcript into canonical LaTeX. Rules: (1) only transcribe what was said; never add, "
          "solve, or complete expressions; (2) if the text contains no mathematical expression, answer exactly NONE; (3) output only "
          "JSON: {\"latex\": str, \"confidence\": float 0-1}. Examples: 'sigma squared' -> \\sigma^2 ; 'partial f partial x' -> "
          "\\frac{\\partial f}{\\partial x} ; 'the integral from zero to one of x squared d x' -> \\int_0^1 x^2\\,dx")


def available():
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def llm_normalize(text, model=None, timeout=60):
    if not available():
        return None
    model = model or os.environ.get("LLM_NORMALIZER_MODEL", "anthropic/claude-haiku-4.5")
    body = json.dumps({"model": model, "temperature": 0, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "Content-Type": "application/json"})
    try:
        out = json.load(urllib.request.urlopen(req, timeout=timeout))["choices"][0]["message"]["content"].strip()
        if out.upper().startswith("NONE"):
            return None
        out = out.strip("`").replace("json\n", "", 1)
        d = json.loads(out)
        return dict(latex=d["latex"], confidence=float(d.get("confidence", 0.5)), model=model, prompt_version=PROMPT_VERSION, system_prompt=SYSTEM)
    except Exception:
        return None
