import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "800"))
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

    GOOGLE_SHEET_ID: str = os.getenv("GOOGLE_SHEET_ID", "")
    GOOGLE_SHEET_RANGE: str = os.getenv("GOOGLE_SHEET_RANGE", "Hoja1!A1:Z999")
    GOOGLE_CREDENTIALS_PATH: str = os.getenv("GOOGLE_CREDENTIALS_PATH", "/app/credentials.json")

    SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    EMAIL_FROM: str = os.getenv("EMAIL_FROM", "")
    EMAIL_TO: str = os.getenv("EMAIL_TO", "")  # comma-separated
    GMAIL_READ = (os.getenv("GMAIL_READ", "false").lower() == "true")


    REPORT_SUBJECT_PREFIX: str = os.getenv("REPORT_SUBJECT_PREFIX", "[Reporte IA]")
    APP_TIMEZONE: str = os.getenv("APP_TIMEZONE", "America/Argentina/Cordoba")

    GOOGLE_CREDENTIALS_PATH: str = os.getenv("GOOGLE_CREDENTIALS_PATH", "/app/credentials.json")
    GOOGLE_AUTH_MODE: str = os.getenv("GOOGLE_AUTH_MODE", "SA")  # SA | SA_DWD | OAUTH
    IMPERSONATE_USER: str = os.getenv("IMPERSONATE_USER", "").strip()

    EMAIL_MODE: str = os.getenv("EMAIL_MODE", "GMAIL_API")  # GMAIL_API | SMTP
    EMAIL_FROM: str = os.getenv("EMAIL_FROM", "")
    EMAIL_TO: str = os.getenv("EMAIL_TO", "")
    # SMTP_* se mantienen por compat si querés usar SMTP
    SALES_DATE_COL: str = os.getenv("SALES_DATE_COL", "Date")
    SALES_AMOUNT_COL: str = os.getenv("SALES_AMOUNT_COL", "Revenue")
    SALES_QTY_COL: str = os.getenv("SALES_QTY_COL", "Quantity")
    GROUP_BY_1: str = os.getenv("GROUP_BY_1", "Region")
    GROUP_BY_2: str = os.getenv("GROUP_BY_2", "SKU")
    CURRENCY: str = os.getenv("CURRENCY", "USD")
    SALES_LOOKBACK_DAYS: int = int(os.getenv("SALES_LOOKBACK_DAYS", "90"))

settings = Settings()
