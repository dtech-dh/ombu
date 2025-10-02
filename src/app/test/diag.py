# app/diag.py
import os, json, sys, hashlib
from datetime import datetime, timezone
import logging

# Reutilizamos settings si está disponible
try:
    from ..config import settings  # noqa: F401
except Exception:
    class _S: pass
    settings = _S()  # fallback

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("diag")

def kv(k, v):
    print(f"{k:28s}: {v}")

def mask(s: str, keep=6):
    if not s:
        return ""
    return "*" * (len(s) - keep) + s[-keep:] if len(s) > keep else "*" * len(s)

def file_info(path):
    try:
        st = os.stat(path)
        ts = datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
        return f"exists (size={st.st_size} bytes, mtime={ts})"
    except FileNotFoundError:
        return "NOT FOUND"
    except Exception as e:
        return f"error: {e!r}"

def try_refresh(sa_json_path, scopes, subject):
    """Pide un access token (Domain-Wide Delegation) y devuelve metadata + creds."""
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request
    cred = service_account.Credentials.from_service_account_file(
        sa_json_path, scopes=scopes, subject=subject
    )
    cred.refresh(Request())  # token endpoint
    exp = cred.expiry.astimezone(timezone.utc).isoformat(timespec="seconds") if cred.expiry else "unknown"
    token_tail = mask(cred.token, keep=10)
    return exp, token_tail, cred

def diag_gspread_open(creds, sheet_id, range_expr):
    import gspread
    client = gspread.authorize(creds)
    sh = client.open_by_key(sheet_id)
    ws_name, rng = range_expr.split("!", 1)
    ws = sh.worksheet(ws_name)
    # lee 1 celda (la primera del rango) para confirmar acceso
    sample = ws.get(rng.split(":")[0])
    return {
        "title": sh.title,
        "worksheet": ws.title,
        "sample_cell": sample[0][0] if sample and sample[0] else "(vacío)"
    }

def diag_gmail(creds):
    """Intenta endpoints informativos; con sólo gmail.send pueden dar 403 (normal)."""
    out = {}
    try:
        from googleapiclient.discovery import build
        svc = build("gmail", "v1", credentials=creds, cache_discovery=False)
    except Exception as e:
        out["error"] = f"No se pudo iniciar Gmail API: {e}"
        return out

    # Perfil (requiere gmail.readonly/modify/metadata) — con gmail.send suele dar 403
    try:
        prof = svc.users().getProfile(userId="me").execute()
        out["profile"] = {k: prof.get(k) for k in ("emailAddress", "messagesTotal", "threadsTotal", "historyId")}
    except Exception as e:
        out["profile_error"] = str(e)

    # sendAs (requiere gmail.settings.basic) — con gmail.send puede dar 403
    try:
        sendas = svc.users().settings().sendAs().list(userId="me").execute()
        addrs = [item.get("sendAsEmail") for item in (sendas.get("sendAs") or [])]
        out["sendAs"] = addrs
    except Exception as e:
        out["sendAs_error"] = str(e)

    return out

def diag_openai():
    """Chequea OPENAI_API_KEY y prueba una completion mínima con httpx + OpenAI."""
    result = {}
    key = os.getenv("OPENAI_API_KEY") or ""
    result["OPENAI_API_KEY_tail"] = mask(key, keep=8)
    model = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    result["OPENAI_MODEL"] = model
    if not key:
        result["error"] = "OPENAI_API_KEY no seteada"
        return result
    try:
        from openai import OpenAI
        import httpx
        proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
        http_client = httpx.Client(proxies=proxy) if proxy else httpx.Client()
        client = OpenAI(api_key=key, http_client=http_client)
        _ = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0
        )
        result["chat_ok"] = True
    except Exception as e:
        result["error"] = f"OpenAI error: {e}"
    return result

def main():
    print("\n=== DIAGNÓSTICO CONTENEDOR GSHEETS/IA/MAIL ===\n")

    # 1) ENV visibles
    kv("PYTHON_VERSION", sys.version.split()[0])
    envs = {
        "GOOGLE_AUTH_MODE": os.getenv("GOOGLE_AUTH_MODE"),
        "IMPERSONATE_USER": os.getenv("IMPERSONATE_USER"),
        "GOOGLE_CREDENTIALS_PATH": os.getenv("GOOGLE_CREDENTIALS_PATH", "/app/credentials.json"),
        "GOOGLE_SHEET_ID": os.getenv("GOOGLE_SHEET_ID"),
        "GOOGLE_SHEET_RANGE": os.getenv("GOOGLE_SHEET_RANGE"),
        "EMAIL_MODE": os.getenv("EMAIL_MODE"),
        "SMTP_HOST": os.getenv("SMTP_HOST"),
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL"),
    }
    for k, v in envs.items():
        kv(k, v)

    sa_path = envs["GOOGLE_CREDENTIALS_PATH"]

    # 2) Credencial SA
    kv("credentials.json", file_info(sa_path))
    try:
        with open(sa_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        kv("SA client_email", data.get("client_email"))
        kv("SA client_id", str(data.get("client_id")))
        kv("SA project_id", data.get("project_id"))
        kv("private_key_id", mask(data.get("private_key_id"), keep=6))
        sha = hashlib.sha256(json.dumps(
            {k: data.get(k) for k in ("client_email","client_id","project_id","private_key_id")},
            sort_keys=True).encode()
        ).hexdigest()
        kv("cred_meta_sha256", sha)
    except Exception as e:
        kv("credentials.json parse", f"ERROR: {e!r}")

    # 3) Token + acceso Sheets
    from google.oauth2 import service_account  # noqa: F401
    scopes_sheets = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds_sheets = None
    try:
        exp, token_tail, creds_sheets = try_refresh(sa_path, scopes_sheets, envs["IMPERSONATE_USER"])
        kv("Sheets token expiry (UTC)", exp)
        kv("Sheets token tail", token_tail)
    except Exception as e:
        kv("Sheets token refresh", f"ERROR: {e}")

    if creds_sheets and envs["GOOGLE_SHEET_ID"] and envs["GOOGLE_SHEET_RANGE"]:
        try:
            meta = diag_gspread_open(creds_sheets, envs["GOOGLE_SHEET_ID"], envs["GOOGLE_SHEET_RANGE"])
            kv("Sheet title", meta["title"])
            kv("Worksheet", meta["worksheet"])
            kv("Sample cell", meta["sample_cell"])
        except Exception as e:
            kv("gspread open_by_key", f"ERROR: {e}")

    # 4) Gmail (si EMAIL_MODE=GMAIL_API)
    email_mode = (envs["EMAIL_MODE"] or "").upper()
    if email_mode == "GMAIL_API" and envs["IMPERSONATE_USER"]:
        try:
            exp2, token_tail2, gmail_creds = try_refresh(
                sa_path, ["https://www.googleapis.com/auth/gmail.send"], envs["IMPERSONATE_USER"]
            )
            kv("Gmail token expiry (UTC)", exp2)
            kv("Gmail token tail", token_tail2)

            if GMAIL_READ:
                gdiag = diag_gmail(gmail_creds)  # solo si pedís lectura
                if "error" in gdiag: kv("Gmail API init", f"ERROR: {gdiag['error']}")
                if "profile" in gdiag: kv("Gmail profile email", gdiag["profile"].get("emailAddress"))
                if "profile_error" in gdiag: kv("Gmail profile error", gdiag["profile_error"])
                if "sendAs" in gdiag: kv("Gmail sendAs list", ",".join(gdiag["sendAs"]))
                if "sendAs_error" in gdiag: kv("Gmail sendAs error", gdiag["sendAs_error"])
            else:
                kv("Gmail extra checks", "omitidos (permiso mínimo gmail.send)")
        except Exception as e:
            kv("Gmail token/diag", f"ERROR: {e}")

    # 5) OpenAI
    oai = diag_openai()
    for k, v in oai.items():
        kv(f"OpenAI {k}", v)

    print("\n=== FIN DIAGNÓSTICO ===\n")

if __name__ == "__main__":
    main()
