# app/tools/export_csv.py
"""
Guarda a CSV:
- RAW: filas tal cual se leyeron del Google Sheet.
- ANALISIS: dataset ya limpio y recortado a la ventana, igual que el usado por el prompt.

Se guardan en EXPORT_DIR (default: /app/out) con nombres legibles.
"""

from __future__ import annotations
import os
import sys
from datetime import datetime
import pandas as pd

# Ajustá estos imports a tu estructura real:
from app.servicios.sheets import read_sheet_to_df
from app.config import settings
from app.core.prompt_builder import (
    _ensure_datetime,
    _clean_amount_and_docs,
    _resolve_col,
    _parse_money_cell,  # por si necesitás parsear algo adicional
)

# ----------------------------
# Utilidades de fecha/ventana
# ----------------------------
def _now_anchor() -> pd.Timestamp:
    tz = os.getenv("TIMEZONE")  # ej: "America/Argentina/Cordoba"
    try:
        if tz:
            return pd.Timestamp.now(tz=tz).normalize().tz_convert(None)
    except Exception:
        pass
    return pd.Timestamp.today().normalize()

def _window_slice(df: pd.DataFrame, date_col: str, *, lookback_days: int, anchor_mode: str = "today") -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    s = pd.to_datetime(df[date_col], errors="coerce")
    as_of = s.max().normalize() if anchor_mode == "max_data" else _now_anchor()
    start = as_of - pd.Timedelta(days=lookback_days - 1)
    mask = (s >= start) & (s <= as_of)
    out = df.loc[mask].copy()
    out[date_col] = s.loc[out.index]
    return out, start, as_of

# ----------------------------
# Export principal
# ----------------------------
def export_csv() -> list[str]:
    # Carpeta de salida
    out_dir = os.getenv("EXPORT_DIR", "/app/out")
    os.makedirs(out_dir, exist_ok=True)

    # Lee el sheet "crudo"
    df_raw = read_sheet_to_df()

    # Detecta columnas de fecha/importe según tus settings y sinónimos
    date_hint = settings.SALES_DATE_COL or "Date"
    amt_hint  = settings.SALES_AMOUNT_COL or "Amount"

    date_col = _resolve_col(df_raw, date_hint, [date_hint, "date", "fecha", "invoice date", "order date", "fecha venta"])
    amt_col  = _resolve_col(df_raw, amt_hint,  [amt_hint,  "amount", "importe", "monto", "revenue", "net sales", "total", "total ventas"])

    if date_col is None:
        raise RuntimeError(f"No se pudo resolver columna de fecha (hint={date_hint}). Columnas: {list(df_raw.columns)}")
    if amt_col is None:
        raise RuntimeError(f"No se pudo resolver columna de importe (hint={amt_hint}). Columnas: {list(df_raw.columns)}")

    # Para metadatos de nombre de archivo, calculamos rango del RAW sin modificarlo
    s_raw_dates = pd.to_datetime(df_raw[date_col], errors="coerce")
    raw_min = s_raw_dates.min()
    raw_max = s_raw_dates.max()

    # Guardar RAW sin alterar contenido
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    raw_name = f"sheet_raw_{(raw_min.date() if pd.notna(raw_min) else 'na')}_{(raw_max.date() if pd.notna(raw_max) else 'na')}_{ts}.csv"
    raw_path = os.path.join(out_dir, raw_name)
    df_raw.to_csv(raw_path, index=False, encoding="utf-8-sig")

    # Construir DF de análisis (limpieza + ventana) EXACTO al que usa el pipeline
    df_clean = df_raw.copy()
    # Tipificar fecha y limpiar montos + reglas de documentos
    df_clean[date_col] = _ensure_datetime(df_clean[date_col])
    doc_col = "TipoDocumento" if "TipoDocumento" in df_clean.columns else None
    df_clean = _clean_amount_and_docs(df_clean, amt_col, doc_col)

    # Ventana estricta (por defecto): últimos N días
    lookback_days = int(getattr(settings, "SALES_LOOKBACK_DAYS", 90))
    anchor_mode = (os.getenv("LOOKBACK_ANCHOR", "today") or "today").lower()  # "today" | "max_data"
    strict = (os.getenv("STRICT_LOOKBACK", "1").strip().lower() in ("1", "true", "yes"))

    df_analysis, start, as_of = _window_slice(df_clean, date_col, lookback_days=lookback_days, anchor_mode=anchor_mode)
    if df_analysis.empty and not strict:
        # Fallback no estricto: si quedó vacío, usa todo df_clean
        df_analysis = df_clean
        start = pd.to_datetime(df_analysis[date_col]).min().normalize()
        as_of  = pd.to_datetime(df_analysis[date_col]).max().normalize()

    # Guardar ANALISIS
    an_name = f"sheet_analysis_{start.date()}_{as_of.date()}_{ts}.csv"
    an_path = os.path.join(out_dir, an_name)
    df_analysis.to_csv(an_path, index=False, encoding="utf-8-sig")

    # Extra: manifest con contexto útil
    manifest = (
        f"EXPORT_DIR={out_dir}\n"
        f"DATE_COL={date_col}\n"
        f"AMOUNT_COL={amt_col}\n"
        f"LOOKBACK_DAYS={lookback_days}\n"
        f"ANCHOR_MODE={anchor_mode}\n"
        f"STRICT_LOOKBACK={strict}\n"
        f"RAW_RANGE={raw_min} → {raw_max}\n"
        f"ANALYSIS_RANGE={start} → {as_of}\n"
        f"RAW_ROWS={len(df_raw)} | ANALYSIS_ROWS={len(df_analysis)}\n"
        f"RAW_FILE={raw_path}\n"
        f"ANALYSIS_FILE={an_path}\n"
    )
    man_path = os.path.join(out_dir, f"sheet_export_manifest_{ts}.txt")
    with open(man_path, "w", encoding="utf-8") as f:
        f.write(manifest)

    print("== EXPORT CSV ==")
    print(manifest)
    return [raw_path, an_path, man_path]

if __name__ == "__main__":
    try:
        export_csv()
    except Exception as e:
        print("ERROR en export_csv:", repr(e))
        sys.exit(1)
