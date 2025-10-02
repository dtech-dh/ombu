# app/openai_ai.py
import os
import logging
import httpx
from openai import OpenAI
from ..config import settings

log = logging.getLogger(__name__)

def _make_openai_client() -> OpenAI:
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    http_client = httpx.Client(proxies=proxy) if proxy else httpx.Client()
    return OpenAI(api_key=settings.OPENAI_API_KEY, http_client=http_client)

def get_openai_report(prompt: str) -> str:
    if isinstance(prompt, str) and prompt.startswith("__RAW__"):
        return prompt.split("\n", 1)[1].strip()

    log.info("Consultando OpenAI...")
    client = _make_openai_client()
    resp = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        temperature=settings.OPENAI_TEMPERATURE,
        max_tokens=settings.OPENAI_MAX_TOKENS,
        messages=[
            {"role": "system", "content": "Sos un asistente de análisis para negocio. Responde claro, conciso y accionable; no muestres tu proceso."},
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content.strip()
    log.info("OpenAI completó la respuesta.")
    return content
