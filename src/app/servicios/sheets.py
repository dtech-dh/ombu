from __future__ import annotations

import os
import logging
import re
import pandas as pd
import gspread
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.service_account import Credentials as SACredentials
from google.oauth2.credentials import Credentials as UserCredentials

from app.config import settings

logger = logging.getLogger(__name__)

# Sólo lectura de Sheets
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

# ==============================
# Helpers numéricos y de fechas
# ==============================
_num_keep_re = re.compile(r"[^\d,.\-()]+")  # conserva dígitos/coma/punto/signo/paréntesis

def _safe_parse_number(val):
    """Convierte strings a número sin romper decimales."""
    import numpy as np

    if val is None:
        return np.nan
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)

    s = str(val).strip()
    if s == "":
        return np.nan

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    s = _num_keep_re.sub("", s).replace(" ", "")

    if s.count(".") == 1 and s.count(",") == 0:
        pass
    elif s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    elif (s.count(",") + s.count(".")) >= 2:
        last_comma = s.rfind(",")
        last_dot = s.rfind(".")
        last = max(last_comma, last_dot)
        int_part = re.sub(r"[^\d]", "", s[:last])
        frac_part = re.sub(r"[^\d]", "", s[last + 1 :])
        s = f"{int_part}.{frac_part}"

    try:
        out = float(s)
        return -out if neg else out
    except Exception:
        return np.nan


def _coerce_numeric_series(series: pd.Series, threshold: float = 0.85) -> pd.Series:
    """Intenta convertir una serie de texto a número SI la mayoría luce numérica."""
    import numpy as np

    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    parsed = series.apply(_safe_parse_number)
    ratio = parsed.notna().mean() if len(parsed) else 0.0
    return parsed if ratio >= threshold else series


def _parse_date_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Normalización de fechas robusta desde Google Sheets."""
    if col not in df.columns:
        logger.warning("Columna de fecha '%s' no está en el DataFrame", col)
        return df

    s_in = df[col]
    intentos = []

    if pd.api.types.is_numeric_dtype(s_in):
        intentos.append("excel_serial(numeric)")
        dt = pd.to_datetime(s_in, unit="D", origin="1899-12-30", errors="coerce")
        df[col] = dt
        return df

    s_raw = s_in.astype(str)
    s = (
        s_raw.str.replace("\u00A0", " ")
        .str.replace("\u200B", "")
        .str.replace("\u2060", "")
        .str.replace("[\u202F\u2009]", " ", regex=True)
        .str.replace("[\\/|.]", "/", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    dt = pd.to_datetime(pd.Series([None] * len(s)), errors="coerce")

    date_fmt = os.getenv("SALES_DATE_FORMAT", "").strip()
    if date_fmt:
        intentos.append(f"format={date_fmt}")
        dt_fmt = pd.to_datetime(s, format=date_fmt, errors="coerce")
        dt = dt.combine_first(dt_fmt)

    intentos.append("excel_serial")
    s_num = s.str.replace(r"[,\s]", "", regex=True)
    num = pd.to_numeric(s_num, errors="coerce")
    dt_serial = pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce")
    dt = dt.combine_first(dt_serial)

    intentos.append("iso_like")
    iso_mask = s.str.match(r"^\d{4}-\d{2}-\d{2}")
    dt_iso = pd.to_datetime(s.where(iso_mask), errors="coerce", utc=False)
    dt = dt.combine_first(dt_iso)

    intentos.append("dayfirst")
    dt_dfirst = pd.to_datetime(s, errors="coerce", dayfirst=True)
    dt = dt.combine_first(dt_dfirst)

    intentos.append("monthfirst")
    dt_mfirst = pd.to_datetime(s, errors="coerce", dayfirst=False)
    dt = dt.combine_first(dt_mfirst)

    if dt.isna().any() and s.str.contains(r"(?i)\b(a\.?m\.?|p\.?m\.?|am|pm)\b").any():
        intentos.append("am/pm retry")
        s2 = s.str.replace(r"(?i)\s*(a\.?m\.?|p\.?m\.?|am|pm)\b", "", regex=True).str.strip()
        dt_ampm = pd.to_datetime(s2, errors="coerce", dayfirst=True)
        dt = dt.combine_first(dt_ampm)

    df[col] = dt
    return df

# ==============================
# Auth
# ==============================
def _get_creds():
    mode = (settings.GOOGLE_AUTH_MODE or "SA").upper()

    if mode == "SA_DWD":
        subject = settings.IMPERSONATE_USER
        if not subject:
            raise ValueError("Falta IMPERSONATE_USER para SA_DWD.")
        return service_account.Credentials.from_service_account_file(
            settings.GOOGLE_CREDENTIALS_PATH, scopes=SCOPES, subject=subject
        )

    if mode == "OAUTH":
        token_path = os.getenv("OAUTH_TOKEN_PATH", "/data/token.json")
        if not os.path.exists(token_path):
            raise FileNotFoundError("No existe token OAuth.")
        creds = UserCredentials.from_authorized_user_file(token_path, SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        return creds

    return SACredentials.from_service_account_file(settings.GOOGLE_CREDENTIALS_PATH, scopes=SCOPES)

# ==============================
# Lectura principal
# ==============================
def read_sheet_to_df() -> pd.DataFrame:
    creds = _get_creds()
    client = gspread.authorize(creds)

    logger.info("Sheet ID: %s | Range: %s", settings.GOOGLE_SHEET_ID, settings.GOOGLE_SHEET_RANGE)
    sh = client.open_by_key(settings.GOOGLE_SHEET_ID)

    ws_name, ws_range = settings.GOOGLE_SHEET_RANGE.split("!", 1)
    ws = sh.worksheet(ws_name)

    data = ws.get(
        ws_range,
        value_render_option="UNFORMATTED_VALUE",
        date_time_render_option="SERIAL_NUMBER",
    )

    if not data:
        logger.warning("No se obtuvieron datos del rango.")
        return pd.DataFrame()

    header, *rows = data

    # 🔑 Alinear filas al número de columnas del header
    max_cols = len(header)
    fixed_rows = []
    for i, r in enumerate(rows):
        if len(r) < max_cols:
            r = r + [""] * (max_cols - len(r))
            logger.warning("Fila %d con menos columnas (%d vs %d). Se completó.", i+2, len(r), max_cols)
        elif len(r) > max_cols:
            logger.warning("Fila %d con más columnas (%d vs %d). Se truncó.", i+2, len(r), max_cols)
            r = r[:max_cols]
        fixed_rows.append(r)

    df = pd.DataFrame(fixed_rows, columns=header)

    # Limpiar strings
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()

    # Tipificar la columna de fecha
    date_col = settings.SALES_DATE_COL or "Date"
    if date_col in df.columns:
        df = _parse_date_column(df, date_col)

    # Intentar convertir a numérico
    for col in df.columns:
        if col == date_col:
            continue
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = _coerce_numeric_series(df[col], threshold=0.85)

    logger.info("Filas leídas: %d", len(df))
    return df
