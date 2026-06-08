"""LLM provider abstraction for GATOR.

Extracted from the original chemist_agent.py. Supports Ollama (local),
OpenAI (cloud), and Transformers (local HF). All use stdlib HTTP where
possible — no heavy SDK dependencies for Ollama/OpenAI.

Environment variables:
  Ollama:       OLLAMA_BASE_URL, OLLAMA_MODEL
  OpenAI:       OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL
  Transformers: HF_MODEL
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from functools import lru_cache
from typing import Any

__all__ = [
    "call_llm",
    "call_ollama",
    "call_openai",
    "call_openrouter",
    "call_transformers",
]


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------


def call_llm(
    provider: str,
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    timeout_s: float = 300.0,
    retries: int = 2,
    retry_backoff_s: float = 2.0,
    trust_remote_code: bool = False,
) -> str:
    """Call an LLM via the specified provider with retry logic.

    Parameters
    ----------
    provider : str
        One of 'ollama', 'openai', 'transformers'.
    prompt : str
        The full prompt text.
    model : str, optional
        Model name (provider-specific). Falls back to env var or default.
    temperature : float
        Sampling temperature.
    timeout_s : float
        Request timeout in seconds (Ollama/OpenAI).
    retries : int
        Number of retries for transient failures.
    retry_backoff_s : float
        Base backoff in seconds between retries.
    trust_remote_code : bool
        For transformers provider: allow custom code.

    Returns
    -------
    str
        The LLM's text response.
    """
    provider = provider.strip().lower()

    for attempt in range(max(0, retries) + 1):
        try:
            if provider == "ollama":
                return call_ollama(
                    prompt,
                    model=model,
                    temperature=temperature,
                    timeout_s=timeout_s,
                )
            if provider == "openai":
                return call_openai(
                    prompt,
                    model=model,
                    timeout_s=min(timeout_s, 300.0),
                )
            if provider == "openrouter":
                return call_openrouter(
                    prompt,
                    model=model,
                    temperature=temperature,
                    timeout_s=min(timeout_s, 300.0),
                )
            if provider == "transformers":
                return call_transformers(
                    prompt,
                    model=model,
                    temperature=temperature,
                    trust_remote_code=trust_remote_code,
                )
            raise ValueError(
                f"Unknown LLM provider: {provider!r}. "
                "Expected: 'ollama', 'openai', 'openrouter', or 'transformers'"
            )
        except Exception as e:
            transient = isinstance(e, TimeoutError) or "timed out" in str(e).lower()
            if transient and attempt < retries:
                time.sleep(retry_backoff_s * (2**attempt))
                continue
            raise
    raise RuntimeError("call_llm exhausted all retries without returning")


# ---------------------------------------------------------------------------
# Ollama (local, open-weights via HTTP)
# ---------------------------------------------------------------------------


def call_ollama(
    prompt: str,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.2,
    timeout_s: float = 180.0,
) -> str:
    """Call a local Ollama server via HTTP (stdlib)."""
    model = model or os.getenv("OLLAMA_MODEL") or "llama3.2:3b"
    base_url = base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"

    url = base_url.rstrip("/") + "/api/generate"
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": float(temperature)},
    }

    req = urllib.request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            raw = r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        hint = ""
        try:
            j = json.loads(detail)
            if (
                isinstance(j, dict)
                and isinstance(j.get("error"), str)
                and "not found" in j["error"].lower()
            ):
                hint = (
                    "\nHint: the model isn't installed in Ollama yet.\n"
                    f"- Pull it: `ollama pull {model}`\n"
                    "- Or list installed: `ollama list`\n"
                )
        except Exception:
            pass
        raise RuntimeError(
            f"Ollama API error: HTTP {getattr(e, 'code', '???')}\n{detail}{hint}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            "Could not reach Ollama server. Is it running?\n"
            "- Install: https://ollama.com/\n"
            "- Start server: `ollama serve`\n"
            f"- Base URL tried: {base_url}\n"
            f"Original error: {e}"
        ) from e

    resp = json.loads(raw)
    text = resp.get("response") if isinstance(resp, dict) else None
    if isinstance(text, str) and text.strip():
        return text.strip()
    return json.dumps(resp, indent=2, sort_keys=False)


# ---------------------------------------------------------------------------
# OpenAI (Responses API via HTTPS, stdlib)
# ---------------------------------------------------------------------------


def _openai_extract_text(resp: dict[str, Any]) -> str:
    output_text = resp.get("output_text")
    if isinstance(resp, dict) and isinstance(output_text, str):
        return output_text

    chunks: list[str] = []
    out = resp.get("output") if isinstance(resp, dict) else None
    if isinstance(out, list):
        for item in out:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") in {"output_text", "text"} and isinstance(c.get("text"), str):
                    chunks.append(c["text"])
    return "\n".join(chunks).strip()


def call_openai(
    prompt: str,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout_s: float = 90.0,
) -> str:
    """Call OpenAI Responses API via HTTPS (stdlib)."""
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    model = model or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
    base_url = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    url = base_url.rstrip("/") + "/responses"
    body = {"model": model, "input": prompt}

    req = urllib.request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            raw = r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        raise RuntimeError(f"OpenAI API error: HTTP {getattr(e, 'code', '???')}\n{detail}") from e

    resp = json.loads(raw)
    text = _openai_extract_text(resp)
    if not text:
        return json.dumps(resp, indent=2, sort_keys=False)
    return text


# ---------------------------------------------------------------------------
# OpenRouter (cloud, Chat Completions API via HTTPS)
# ---------------------------------------------------------------------------


def call_openrouter(
    prompt: str,
    api_key: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    timeout_s: float = 300.0,
) -> str:
    """Call OpenRouter Chat Completions API via HTTPS (stdlib)."""
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    model = model or "anthropic/claude-opus-4.6"
    url = "https://openrouter.ai/api/v1/chat/completions"

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
    }

    req = urllib.request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            raw = r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        raise RuntimeError(
            f"OpenRouter API error: HTTP {getattr(e, 'code', '???')}\n{detail}"
        ) from e

    resp = json.loads(raw)

    # Extract text from Chat Completions response
    choices = resp.get("choices") if isinstance(resp, dict) else None
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        text = message.get("content", "")
        if isinstance(text, str) and text.strip():
            return text.strip()

    return json.dumps(resp, indent=2, sort_keys=False)


# ---------------------------------------------------------------------------
# Transformers (local HF models, optional dependency)
# ---------------------------------------------------------------------------


def call_transformers(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.2,
    max_new_tokens: int = 900,
    trust_remote_code: bool = False,
) -> str:
    """Run a local open-weight model via Hugging Face Transformers."""
    model = model or os.getenv("HF_MODEL") or "Qwen/Qwen2.5-3B-Instruct"

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "Transformers backend requires `transformers` + `torch`.\n"
            f"Install: `pip install gator[transformers]`\n"
            f"Original error: {e}"
        ) from e

    device = (
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    )
    dtype_name = "float16" if device == "mps" else "float32"

    @lru_cache(maxsize=2)
    def _load(model_id: str, dev: str, dt: str, trust: bool) -> tuple[Any, Any]:
        dtype = torch.float16 if (dev == "mps" and dt == "float16") else None
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=trust)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=trust
        )
        mdl.eval()
        if dev != "cpu":
            mdl.to(dev)
        return tok, mdl

    tok, mdl = _load(model, device, dtype_name, bool(trust_remote_code))

    inputs = tok(prompt, return_tensors="pt")
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out_ids = mdl.generate(
            **inputs,
            do_sample=float(temperature) > 0,
            temperature=float(temperature),
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tok.eos_token_id,
        )

    gen_ids = out_ids[0]
    prompt_len = inputs["input_ids"].shape[-1]
    new_ids = gen_ids[prompt_len:] if gen_ids.shape[-1] > prompt_len else gen_ids
    return str(tok.decode(new_ids, skip_special_tokens=True).strip())
