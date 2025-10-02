# app/anthropic_ai.py
import os
import logging
from typing import List

try:
    import anthropic
except Exception:
    anthropic = None

log = logging.getLogger(__name__)

ANTHROPIC_MODEL_DEFAULT = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
ANTHROPIC_TEMPERATURE_DEFAULT = float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7"))
ANTHROPIC_MAX_TOKENS_DEFAULT = int(os.getenv("ANTHROPIC_MAX_TOKENS", "1200"))

def get_anthropic_report(prompt: str) -> str:
    if isinstance(prompt, str) and prompt.startswith("__RAW__"):
        return prompt.split("\n", 1)[1].strip()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "__RAW__\n**Reporte alternativo (Anthropic) no generado**\n- Motivo: falta ANTHROPIC_API_KEY"

    if anthropic is None:
        return "__RAW__\n**Reporte alternativo (Anthropic) no generado**\n- Motivo: paquete 'anthropic' no instalado (agregalo en requirements.txt)"

    client = anthropic.Anthropic(api_key=api_key)

    model = os.getenv("ANTHROPIC_MODEL", ANTHROPIC_MODEL_DEFAULT)
    temperature = float(os.getenv("ANTHROPIC_TEMPERATURE", ANTHROPIC_TEMPERATURE_DEFAULT))
    max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", ANTHROPIC_MAX_TOKENS_DEFAULT))

    log.info("Consultando Anthropic... (model=%s, max_tokens=%s, temperature=%s)", model, max_tokens, temperature)

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        log.exception("Error llamando a Anthropic: %r", e)
        return "__RAW__\n**Reporte alternativo (Anthropic) no generado**\n- Motivo: error al llamar a Anthropic: %r" % (e,)

    parts: List[str] = []
    for block in getattr(resp, "content", []) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
        elif isinstance(block, str):
            parts.append(block)

    out = "\n".join(p for p in parts if p).strip()
    if not out:
        out = "(Anthropic devolvió respuesta vacía)"
    log.info("Anthropic completó la respuesta.")
    return out
