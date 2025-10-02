import logging, base64
from email.message import EmailMessage
from email.utils import formataddr
from email_validator import validate_email, EmailNotValidError
from googleapiclient.discovery import build
from google.oauth2 import service_account
import smtplib
from ..config import settings

logger = logging.getLogger(__name__)

def _parse_recipients(to_field: str) -> list[str]:
    recips = [x.strip() for x in to_field.split(",") if x.strip()]
    valid = []
    for r in recips:
        try:
            validate_email(r, check_deliverability=False)
            valid.append(r)
        except EmailNotValidError as e:
            logger.warning("Email inválido omitido: %s (%s)", r, e)
    return valid

def _send_via_gmail_api(subject: str, html_body: str, text_body: str | None = None):
    if not settings.IMPERSONATE_USER:
        raise ValueError("IMPERSONATE_USER requerido para Gmail API (SA_DWD).")

    recipients = _parse_recipients(settings.EMAIL_TO)
    if not recipients:
        raise ValueError("No hay destinatarios válidos en EMAIL_TO")

    msg = EmailMessage()
    msg["Subject"] = subject
    from_addr = settings.EMAIL_FROM or formataddr(("IT Reports", settings.IMPERSONATE_USER))
    msg["From"] = from_addr
    msg["To"] = ", ".join(recipients)

    if text_body:
        msg.set_content(text_body)
        msg.add_alternative(html_body, subtype="html")
    else:
        msg.set_content(html_body, subtype="html")

    creds = service_account.Credentials.from_service_account_file(
        settings.GOOGLE_CREDENTIALS_PATH,
        scopes=["https://www.googleapis.com/auth/gmail.send"],
        subject=settings.IMPERSONATE_USER,
    )
    service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    service.users().messages().send(userId="me", body={"raw": raw}).execute()
    logger.info("Correo enviado por Gmail API como %s a %s", settings.IMPERSONATE_USER, recipients)

def _send_via_smtp(subject: str, html_body: str, text_body: str | None = None):
    recipients = _parse_recipients(settings.EMAIL_TO)
    if not recipients:
        raise ValueError("No hay destinatarios válidos en EMAIL_TO")
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.EMAIL_FROM or formataddr(("IT Reports", settings.SMTP_USER))
    msg["To"] = ", ".join(recipients)
    if text_body:
        msg.set_content(text_body); msg.add_alternative(html_body, subtype="html")
    else:
        msg.set_content(html_body, subtype="html")
    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
        server.ehlo(); server.starttls()
        server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
        server.send_message(msg)
    logger.info("Correo enviado por SMTP a %s", recipients)

def send_email(subject: str, html_body: str, text_body: str | None = None):
    mode = (settings.EMAIL_MODE or "GMAIL_API").upper()
    if mode == "GMAIL_API":
        return _send_via_gmail_api(subject, html_body, text_body)
    return _send_via_smtp(subject, html_body, text_body)
