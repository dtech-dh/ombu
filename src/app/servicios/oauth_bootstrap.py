# app/oauth_bootstrap.py
import os
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
CLIENT_PATH = os.getenv("OAUTH_CLIENT_PATH", "/data/oauth_client.json")
TOKEN_PATH = os.getenv("OAUTH_TOKEN_PATH", "/data/token.json")
OAUTH_PORT = int(os.getenv("OAUTH_PORT", "8080"))

def main():
    if not os.path.exists(CLIENT_PATH):
        raise FileNotFoundError(f"No se encontró el OAuth client en {CLIENT_PATH}")

    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # refresh silencioso si ya había token
            creds.refresh(Request())
        else:
            # Flow de escritorio con loopback server
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_PATH, SCOPES)
            # No intentamos abrir el navegador desde el contenedor
            creds = flow.run_local_server(
                port=OAUTH_PORT,
                open_browser=False,
                prompt="consent",
                authorization_prompt_message=(
                    "\n>>> Abrí esta URL en tu navegador y autorizá el acceso:\n{url}\n\n"
                    f"Tras autorizar, vas a ser redirigido a http://localhost:{OAUTH_PORT}/ (este contenedor).\n"
                    "Volvé a la consola cuando el navegador diga que ya podés cerrar la ventana.\n"
                )
            )

        # Persistimos el token para corridas futuras
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
        print(f"\nToken guardado en {TOKEN_PATH}")

    print("OAuth listo.")

if __name__ == "__main__":
    main()
