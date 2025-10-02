# app/servicios/openai_assistant_runner.py
import os
import json
import logging
import argparse
import httpx
from typing import Optional

try:
    # mismo patrón que usás en otros módulos
    from ..config import settings
except Exception:
    # fallback si se ejecuta standalone
    class _S: pass
    settings = _S()
    settings.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    settings.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    settings.OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    settings.OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "1200"))
    settings.OPENAI_ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")

from openai import OpenAI

log = logging.getLogger(__name__)

# ---------- util cliente ----------
def _make_openai_client() -> OpenAI:
    """
    Crea el cliente de OpenAI usando httpx (respeta HTTP[S]_PROXY),
    igual que tu estructura actual.
    """
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    http_client = httpx.Client(proxies=proxy, timeout=httpx.Timeout(600.0, connect=5.0, read=600.0))
    return OpenAI(api_key=settings.OPENAI_API_KEY, http_client=http_client)

# ---------- Assistants API ----------
def run_with_assistant(prompt: str,
                       assistant_id: Optional[str] = None,
                       extra_instructions: Optional[str] = None) -> str:
    """
    Envía `prompt` a un Assistant concreto (OPENAI_ASSISTANT_ID).
    Crea un thread nuevo por corrida y devuelve el texto de la última respuesta.
    """
    assistant_id = assistant_id or getattr(settings, "OPENAI_ASSISTANT_ID", None)
    if not assistant_id:
        raise ValueError("Falta OPENAI_ASSISTANT_ID (config o env).")

    client = _make_openai_client()

    # Podés inspeccionar la "request" para comparar con tu flujo actual
    req_dbg = {
        "method": "POST",
        "url": "/v1/assistants/{assistant_id}/runs (via beta.threads.*)",
        "model": "(definido al crear el Assistant en la consola)",
        "max_tokens": settings.OPENAI_MAX_TOKENS,
        "temperature": settings.OPENAI_TEMPERATURE,
        "assistant_id": assistant_id,
    }
    log.debug("Request (assistants): %s", json.dumps(req_dbg, ensure_ascii=False))

    # 1) thread nuevo
    thread = client.beta.threads.create()

    # 2) cargamos el mensaje del usuario
    content = prompt if not extra_instructions else f"{extra_instructions}\n\n{prompt}"
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=content
    )

    # 3) corremos el Assistant y aguardamos (poll)
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )

    if run.status != "completed":
        # Devolvé algo útil para debug si el run no terminó ok
        raise RuntimeError(f"Run finalizó con estado={run.status}. last_error={getattr(run, 'last_error', None)}")

    # 4) leemos la última respuesta del hilo (desc)
    msgs = client.beta.threads.messages.list(thread_id=thread.id, order="desc", limit=1)
    if not msgs.data:
        return ""

    parts = []
    for block in msgs.data[0].content:
        if getattr(block, "type", "") == "text":
            parts.append(block.text.value)
    return "\n".join(parts).strip()

# ---------- Chat Completions (fallback para comparar) ----------
def run_with_chat_completions(prompt: str) -> str:
    """
    Mismo estilo que tu app/openai_ai.py para tener una base de comparación.
    """
    client = _make_openai_client()
    log.debug("Request (chat.completions): %s", json.dumps({
        "method": "POST",
        "url": "/v1/chat/completions",
        "model": settings.OPENAI_MODEL,
        "max_tokens": settings.OPENAI_MAX_TOKENS,
        "temperature": settings.OPENAI_TEMPERATURE,
    }))

    resp = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        temperature=settings.OPENAI_TEMPERATURE,
        max_tokens=settings.OPENAI_MAX_TOKENS,
        messages=[
            {"role": "system", "content": "Sos un asistente de análisis para negocio. Responde claro, conciso y accionable; no muestres tu proceso."},
            {"role": "user", "content": prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()

# ---------- CLI ----------
def _read_prompt_from_args(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    # stdin
    return "".join(iter(input, ""))

def main():
    parser = argparse.ArgumentParser(description="Runner para Assistant específico o Chat Completions.")
    parser.add_argument("--mode", choices=["assistant", "chat"], default="assistant",
                        help="assistant=usa OPENAI_ASSISTANT_ID | chat=usa Chat Completions")
    parser.add_argument("--prompt", help="Texto del prompt (prioridad sobre --prompt-file)")
    parser.add_argument("--prompt-file", help="Ruta a archivo con el prompt")
    parser.add_argument("--assistant-id", help="Sobrescribe OPENAI_ASSISTANT_ID")
    parser.add_argument("--extra", help="Instrucciones extra que se anteponen al prompt (solo assistants)", default=None)
    parser.add_argument("--loglevel", default=os.getenv("LOGLEVEL", "INFO"))
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    prompt = _read_prompt_from_args(args).strip()
    if not prompt:
        raise SystemExit("Falta prompt. Usa --prompt o --prompt-file.")

    if args.mode == "assistant":
        out = run_with_assistant(prompt, assistant_id=args.assistant_id, extra_instructions=args.extra)
    else:
        out = run_with_chat_completions(prompt)

    print(out)

if __name__ == "__main__":
    main()
