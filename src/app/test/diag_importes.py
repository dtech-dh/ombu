# app/test/diag_importes.py  (o donde lo estés usando)
from app.servicios.sheets import read_sheet_to_df
from app.config import settings
from app.core.prompt_builder import (
    _clean_amount_and_docs,
    _ensure_datetime,
    _resolve_col,
    _norm,
    _parse_money_cell,  # lo usamos para el "bruto" sin reglas de documentos
)
import pandas as pd, os

def _now_anchor():
    """Devuelve 'hoy' normalizado respetando TIMEZONE si está seteado."""
    tz = os.getenv("TIMEZONE")  # ej: America/Argentina/Cordoba
    try:
        if tz:
            return pd.Timestamp.now(tz=tz).normalize().tz_convert(None)
    except Exception:
        pass
    return pd.Timestamp.today().normalize()

# === Carga ===
df = read_sheet_to_df()

# === Resolver columnas ===
dc = settings.SALES_DATE_COL or "Date"
ac = settings.SALES_AMOUNT_COL or "Amount"
dc = _resolve_col(df, dc, [dc, "date", "fecha", "invoice date", "order date", "fecha venta"])
ac = _resolve_col(df, ac, [ac, "amount", "importe", "monto", "revenue", "net sales", "total", "total ventas"])

if dc is None or ac is None:
    raise SystemExit(f"No pude resolver columnas de fecha/importe. Columnas: {list(df.columns)}")

# === Tipificar fecha y limpiar montos + reglas de documentos ===
df[dc] = _ensure_datetime(df[dc])
doc_col = "TipoDocumento" if "TipoDocumento" in df.columns else None
df_clean = _clean_amount_and_docs(df, ac, doc_col)

# === Ventana estricta: sólo últimos N días, anclados a 'today' (o max_data) ===
lookback_days = int(getattr(settings, "SALES_LOOKBACK_DAYS", 90))
anchor_mode = (os.getenv("LOOKBACK_ANCHOR", "today") or "today").lower()  # today | max_data

s_clean = pd.to_datetime(df_clean[dc], errors="coerce")
as_of = s_clean.max().normalize() if anchor_mode == "max_data" else _now_anchor()
start = as_of - pd.Timedelta(days=lookback_days - 1)

mask_win_clean = (s_clean >= start) & (s_clean <= as_of)
rows_before = len(df_clean)
df_clean = df_clean.loc[mask_win_clean].copy()   # <- acá sacamos TODO lo fuera de rango
rows_after = len(df_clean)

print(f"As-of: {as_of.date()} | Ventana: {start.date()} → {as_of.date()}")
print(f"Filas ventana (limpio): {rows_after} (descartadas fuera de rango: {rows_before - rows_after})")

# === Totales ===
# BRUTO (sin reglas de documentos), pero casteando el texto a número de forma robusta:
if ac in df.columns:
    s_raw_date = pd.to_datetime(df[dc], errors="coerce")
    raw_win = df.loc[(s_raw_date >= start) & (s_raw_date <= as_of), [dc, ac]].copy()
    # sólo casteo de formato monetario; no excluyo/nego documentos:
    bruto_total = raw_win[ac].apply(_parse_money_cell).sum()
else:
    bruto_total = float("nan")

# LIMPIO (con reglas de documentos y montos ya normalizados):
limpio_total = df_clean[ac].sum() if not df_clean.empty else 0.0

print("TOTAL BRUTO 90d (sin reglas docs/negativos):", f"{bruto_total:,.2f}")
print("TOTAL LIMPIO 90d (con reglas):", f"{limpio_total:,.2f}")

# Desglose por TipoDocumento (si existe)
if doc_col and doc_col in df_clean.columns and not df_clean.empty:
    print("\nTop TipoDocumento (limpio, 90d):")
    print(df_clean.groupby(doc_col)[ac].sum().sort_values(ascending=False).head(20))

# Ejemplos de montos mínimos (para revisar signos/parseo)
cols_show = [c for c in [dc, ac, doc_col] if c]
if not df_clean.empty:
    sample = df_clean[cols_show].sort_values(ac).head(10)
    print("\nEjemplos montos mínimos (limpio):")
    print(sample.to_string(index=False))
