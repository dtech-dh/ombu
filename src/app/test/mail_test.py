# app/mail_test.py
from datetime import datetime
from ..servicios.emailer import send_email

if __name__ == "__main__":
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subj = f"[TEST] Gmail API DWD OK - {now}"
    html = f"<html><body><h3>Prueba OK</h3><p>Enviado a las {now}.</p></body></html>"
    send_email(subject=subj, html_body=html)
    print("Mail enviado.")
