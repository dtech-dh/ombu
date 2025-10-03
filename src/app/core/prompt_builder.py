# app/core/prompt_builder.py
import logging
import os
import re
import json
import unicodedata
import pandas as pd
import numpy as np
from datetime import date, datetime
from app.config import settings
from app.core.risk import compute_risk, format_risk_markdown
from app.core.logistics import compute_truck_metrics  # métricas de camión

logger = logging.getLogger(__name__)

# ======================================================
# Utilidades JSON
# ======================================================
def _json_default(o):
    import pandas as pd
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.isoformat()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)

# ======================================================
# Normalización de textos
# ======================================================
def _norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    # quitar signos que molestan
    for ch in ["(", ")", "[", "]", "{", "}", "$", "%", "#", "!", "?", ":", ";", "'", '"']:
        s = s.replace(ch, " ")
    s = s.replace("-", " ").replace("_", " ").replace("/", " ")
    s = " ".join(s.split())
    return s

def _resolve_col(df: pd.DataFrame, desired: str | None, synonyms: list[str]) -> str | None:
    """Resuelve nombre de columna real por heurística/sinónimos."""
    cols = list(df.columns)
    if not cols:
        return None
    norm_map = {_norm(c): c for c in cols}

    # pedido explícito
    if desired and desired in df.columns:
        return desired
    if desired:
        nd = _norm(desired)
        if nd in norm_map:
            return norm_map[nd]
        for nc, real in norm_map.items():
            if nd and nd in nc:
                return real

    # sinónimos
    for syn in synonyms:
        if syn in df.columns:
            return syn
        ns = _norm(syn)
        if ns in norm_map:
            return norm_map[ns]
    for syn in synonyms:
        ns = _norm(syn)
        for nc, real in norm_map.items():
            if ns and ns in nc:
                return real
    return None

def _ensure_datetime(series):
    """Convierte a datetime con tolerancia."""
    try:
        return pd.to_datetime(series, errors="coerce", dayfirst=True)
    except Exception:
        return pd.to_datetime(series, errors="coerce")

# ======================================================
# Limpieza robusta de montos
# ======================================================
_currency_chars = re.compile(r"[^\d,.\-\(\)]")

def _parse_money_cell(val):
    """
    Convierte '$1.234,56', '(1,234.56)', '1.234.567', '532.5', etc. a float.
    Paréntesis => negativo. Heurística de separadores , .
    """
    # ya numérico
    if isinstance(val, (int, float, np.number)):
        try:
            return float(val)
        except Exception:
            return np.nan

    if val is None:
        return np.nan
    s = str(val).strip()
    if s == "":
        return np.nan

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    # eliminar símbolos de moneda/espacios
    s = _currency_chars.sub("", s)

    # heurística separadores
    if "," in s and "." in s:
        # Si la última coma está a la derecha de la última punto => notación EU
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        tail = s.split(",")[-1]
        # cola de 2 o 3 dígitos => decimal
        if len(tail) in (2, 3):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    # si solo hay puntos, asumimos US y quitamos separadores de miles (si hubiera)
    # float() se encarga

    try:
        out = float(s)
        return -out if neg else out
    except Exception:
        return np.nan

def _clean_amount_and_docs(df: pd.DataFrame, amount_col: str, doc_col: str | None):
    """
    Normaliza montos y aplica filtros por tipo de documento.
    - Incluye solo doc_types configurados (por defecto: ventas).
    - Niega signos para Notas de crédito.
    """
    if amount_col not in df.columns:
        return df

    df = df.copy()
    # Parseo robusto de monto (si viene texto/mixto)
    if not pd.api.types.is_numeric_dtype(df[amount_col]):
        df[amount_col] = df[amount_col].apply(_parse_money_cell)
    else:
        df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")

    if doc_col and doc_col in df.columns:
        include_raw = os.getenv(
            "SALES_INCLUDE_DOCS",
            "Invoice,Sales Receipt,Credit Memo,Factura,Recibo,Nota de crédito",
        )
        negate_raw = os.getenv("NEGATE_DOC_TYPES", "Credit Memo,Nota de crédito")
        exclude_raw = os.getenv(
            "EXCLUDE_DOC_TYPES",
            "Payment,Journal,Estimate,Sales Order,Pago,Diario,Presupuesto,Pedido",
        )

        def _normdoc(x: str) -> str:
            return _norm(x)

        include = {_normdoc(x) for x in include_raw.split(",") if x.strip()}
        negate = {_normdoc(x) for x in negate_raw.split(",") if x.strip()}
        exclude = {_normdoc(x) for x in exclude_raw.split(",") if x.strip()}

        before = len(df)
        df = df[~df[doc_col].map(lambda x: _normdoc(str(x)) in exclude)].copy()
        if include:
            df = df[df[doc_col].map(lambda x: _normdoc(str(x)) in include)].copy()

        def _maybe_neg(row):
            d = _normdoc(str(row.get(doc_col)))
            val = row.get(amount_col)
            if pd.notna(val) and d in negate and val > 0:
                return -val
            return val

        df[amount_col] = df.apply(_maybe_neg, axis=1)
        logger.info(
            "Filtro TipoDocumento: filas %s → %s (include=%s | exclude=%s | negate=%s)",
            before, len(df), include_raw, exclude_raw, negate_raw
        )

    df = df.dropna(subset=[amount_col])
    return df

# ======================================================
# Filtros de negocio (Marca / Clase / IDs)
# ======================================================
def _filter_brand(df: pd.DataFrame, product_col: str | None) -> pd.DataFrame:
    """
    Mantiene solo filas de la marca configurada (por defecto: 'Dos Hermanos').
    Si no hay columna de producto o BRAND_FILTER no está seteado, no filtra.
    """
    brand_raw = os.getenv("BRAND_FILTER", "dos hermanos").strip().lower()
    if not product_col or product_col not in df.columns or not brand_raw:
        return df
    m = df[product_col].astype(str).str.lower()
    keep = m.str.contains(brand_raw, na=False)
    before, after = len(df), int(keep.sum())
    if after != before:
        logger.info("Filtro marca '%s': filas %s → %s", brand_raw, before, after)
    return df[keep].copy()

def _exclude_class_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Excluye valores de la columna Class según env CLASS_EXCLUDE (coma-separado).
    Default: 'Bulk Rice' (pedido explícito).
    """
    class_col = None
    for c in ["Class", "Clase", "Category", "Categoria"]:
        if c in df.columns:
            class_col = c
            break
    if not class_col:
        return df

    exclude_raw = os.getenv("CLASS_EXCLUDE", "Bulk Rice")
    vals = [v.strip() for v in exclude_raw.split(",") if v.strip()]
    if not vals:
        return df

    before = len(df)
    mask = ~df[class_col].astype(str).str.strip().isin(vals)
    out = df[mask].copy()
    removed = before - len(out)
    if removed > 0:
        logger.info("Excluidas %s filas por %s in %s", removed, class_col, vals)
    return out

# ======================================================
# Preparación del DF final de análisis (USADO POR PROMPT)
# ======================================================
def _now_anchor() -> pd.Timestamp:
    tz = os.getenv("TIMEZONE")
    try:
        if tz:
            return pd.Timestamp.now(tz=tz).normalize().tz_convert(None)
    except Exception:
        pass
    return pd.Timestamp.today().normalize()

def _prepare_analysis_df(df: pd.DataFrame):
    """
    Devuelve (base_df_final, meta) con EXACTAMENTE el DF que usa el prompt.
    Aplica:
      - Resolver columnas (fecha, monto, qty, cliente, grupos, ruta).
      - Parsear fecha/monto.
      - Filtrar documentos.
      - Excluir IDs configurados.
      - Filtrar por marca y excluir clase (p.ej. 'Bulk Rice').
      - Ventana de lookback (por defecto 90 días).
    """
    dc_req = settings.SALES_DATE_COL or "Date"
    ac_req = settings.SALES_AMOUNT_COL or "Amount"
    qc_req = settings.SALES_QTY_COL
    g1_req = settings.GROUP_BY_1
    g2_req = settings.GROUP_BY_2

    date_syn = [dc_req, "date", "fecha", "order date", "fecha venta", "invoice date"]
    amt_syn  = [ac_req, "revenue", "amount", "importe", "monto", "total", "venta", "sales", "net sales", "total ventas", "importe total"]
    qty_syn  = [qc_req, "qty", "quantity", "unidades", "cantidad"] if qc_req else []
    g1_syn   = [g1_req, "region", "zona", "sucursal", "canal", "category", "categoria"] if g1_req else []
    g2_syn   = [g2_req, "sku", "producto", "item", "id producto", "ref"] if g2_req else []
    cust_syn = ["Customer", "customer", "cliente", "client", "buyer", "account", "razon social", "nombre cliente"]
    route_syn = ["Truck", "Camion", "Route", "Ruta", "SalesRep"]

    # Resolver columnas
    dc = _resolve_col(df, dc_req, date_syn)
    ac = _resolve_col(df, ac_req, amt_syn)
    qc = _resolve_col(df, qc_req, qty_syn) if qc_req else None
    g1 = _resolve_col(df, g1_req, g1_syn) if g1_req else None
    g2 = _resolve_col(df, g2_req, g2_syn) if g2_req else None
    cu = _resolve_col(df, "Customer", cust_syn)
    rc = _resolve_col(df, os.getenv("ROUTE_COL", "") or None, route_syn)

    if dc is None or ac is None:
        raise RuntimeError(
            f"Columnas requeridas faltantes. Fecha={dc_req}->{dc} | Importe={ac_req}->{ac}. Disponibles: {list(df.columns)}"
        )

    df = df.copy()
    # Fecha y Monto
    df[dc] = _ensure_datetime(df[dc])
    df = df.dropna(subset=[dc])
    df = _clean_amount_and_docs(df, ac, "TipoDocumento" if "TipoDocumento" in df.columns else None)
    if qc:
        df[qc] = pd.to_numeric(df[qc], errors="coerce")

    # Excluir IDs específicos (p/ej. 1389644)
    exclude_id_col = os.getenv("EXCLUDE_ID_COL", "ID")
    exclude_ids_raw = os.getenv("EXCLUDE_IDS", "1389644")  # default incluye tu caso
    exclude_ids = {x.strip() for x in exclude_ids_raw.split(",") if x.strip()}
    if exclude_id_col in df.columns and exclude_ids:
        before = len(df)
        df = df[~df[exclude_id_col].astype(str).isin(exclude_ids)].copy()
        removed = before - len(df)
        if removed:
            logger.info("Excluidas %s filas por %s in %s", removed, exclude_id_col, sorted(exclude_ids))

    # Filtro por marca (solo Dos Hermanos, configurable)
    prod_col_guess = g2 if (g2 and g2 in df.columns) else ("Producto" if "Producto" in df.columns else None)
    df = _filter_brand(df, prod_col_guess)

    # Excluir clase (p.ej. 'Bulk Rice')
    df = _exclude_class_values(df)

    # Ventana
    lookback_days = int(getattr(settings, "SALES_LOOKBACK_DAYS", 90))
    anchor_mode = (os.getenv("LOOKBACK_ANCHOR", "today") or "today").lower()  # today | max_data
    strict = (os.getenv("STRICT_LOOKBACK", "1").strip().lower() in ("1", "true", "yes"))

    if anchor_mode == "max_data":
        as_of = pd.to_datetime(df[dc]).max().normalize()
    else:
        as_of = _now_anchor()

    base_start = as_of - pd.Timedelta(days=lookback_days - 1)
    base_df = df[(df[dc] >= base_start) & (df[dc] <= as_of)].copy()

    if base_df.empty and not strict:
        base_df = df.copy()
        base_start = pd.to_datetime(df[dc]).min().normalize()

    meta = {
        "as_of_date": str(as_of.date()),
        "lookback_days": lookback_days,
        "base_period": {"start": str(base_start.date()), "end": str(as_of.date())},
        "resolved_columns": {
            "date": dc, "amount": ac, "quantity": qc,
            "group1": g1, "group2": g2, "customer": cu, "route": rc
        },
        "currency": settings.CURRENCY,
    }
    return base_df, meta

def get_analysis_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """API pública: devuelve el DF FINAL de análisis y la metadata (rango/columnas)."""
    return _prepare_analysis_df(df)

# ======================================================
# Métricas y Prompt
# ======================================================
def _monthly_breakdowns(base_df: pd.DataFrame, dc: str, ac: str, g2: str | None):
    """Devuelve agregados mensuales y por producto-mes (top 10 productos por revenue)."""
    if base_df.empty:
        return {"monthly_totals": [], "monthly_by_product_top": []}

    tmp = base_df.copy()
    tmp["__ym"] = pd.to_datetime(tmp[dc]).dt.to_period("M").astype(str)
    monthly_totals = (
        tmp.groupby("__ym")[ac]
           .sum()
           .reset_index()
           .rename(columns={"__ym": "year_month", ac: "revenue"})
           .sort_values("year_month")
           .to_dict(orient="records")
    )

    monthly_by_product_top = []
    if g2 and g2 in tmp.columns:
        prod_month = (
            tmp.groupby([g2, "__ym"])[ac]
               .sum()
               .reset_index()
               .rename(columns={"__ym": "year_month", ac: "revenue"})
        )
        # Top 10 productos por revenue total en la ventana
        top10 = (
            prod_month.groupby(g2)["revenue"]
                     .sum()
                     .sort_values(ascending=False)
                     .head(10)
                     .index
        )
        monthly_by_product_top = prod_month[prod_month[g2].isin(top10)].sort_values([g2, "year_month"]).to_dict(orient="records")

    return {"monthly_totals": monthly_totals, "monthly_by_product_top": monthly_by_product_top}

def _customer_stats(base_df: pd.DataFrame, dc: str, ac: str, cu: str | None):
    """Cantidad de clientes y frecuencia (aprox mediana días entre pedidos por cliente)."""
    if not cu or cu not in base_df.columns or base_df.empty:
        return {"unique_customers": None, "median_days_between_orders": None, "avg_orders_per_customer": None}

    df = base_df.copy()
    df["__day"] = pd.to_datetime(df[dc]).dt.normalize()

    unique_customers = int(df[cu].nunique())

    # órdenes por cliente (días distintos con ventas)
    orders_per_cust = (
        df.groupby([cu, "__day"])[ac]
          .sum()
          .reset_index()
          .groupby(cu)["__day"]
          .size()
    )
    avg_orders_per_customer = float(orders_per_cust.mean()) if len(orders_per_cust) > 0 else None

    # mediana de días entre pedidos por cliente
    gaps = (
        df[[cu, "__day"]].drop_duplicates().sort_values([cu, "__day"])
          .groupby(cu)["__day"]
          .apply(lambda s: s.sort_values().diff().median())
    )
    if not gaps.empty and pd.api.types.is_timedelta64_dtype(gaps):
        median_days_between_orders = float(np.nanmedian(gaps.dt.days.dropna())) if gaps.notna().any() else None
    else:
        median_days_between_orders = None

    return {
        "unique_customers": unique_customers,
        "median_days_between_orders": median_days_between_orders,
        "avg_orders_per_customer": avg_orders_per_customer,
    }

def _summaries_from_base(base_df: pd.DataFrame, meta: dict) -> dict:
    dc = meta["resolved_columns"]["date"]
    ac = meta["resolved_columns"]["amount"]
    qc = meta["resolved_columns"]["quantity"]
    cu = meta["resolved_columns"]["customer"]
    g1 = meta["resolved_columns"]["group1"]
    g2 = meta["resolved_columns"]["group2"]

    as_of = pd.to_datetime(meta["base_period"]["end"])
    lookback_days = meta["lookback_days"]

    def window_summary(days: int):
        start = as_of - pd.Timedelta(days=days - 1)
        prev_start = start - pd.Timedelta(days=days)
        prev_end = start - pd.Timedelta(days=1)

        cur = base_df[(base_df[dc] >= start) & (base_df[dc] <= as_of)]
        prev = base_df[(base_df[dc] >= prev_start) & (base_df[dc] <= prev_end)]

        cur_rev = float(cur[ac].sum()) if ac in cur.columns else 0.0
        prev_rev = float(prev[ac].sum()) if ac in prev.columns else 0.0
        cur_qty = float(cur[qc].sum()) if (qc and qc in cur.columns) else None
        prev_qty = float(prev[qc].sum()) if (qc and qc in prev.columns) else None

        def delta(a, b):
            if b in (0, None):
                return None
            try:
                return (a - b) / b
            except Exception:
                return None

        return {
            "days": int(days),
            "period": {"start": str(start.date()), "end": str(as_of.date())},
            "revenue": {"current": cur_rev, "previous": prev_rev, "pct_change": delta(cur_rev, prev_rev)},
            "quantity": (
                {"current": cur_qty, "previous": prev_qty, "pct_change": delta(cur_qty, prev_qty)}
                if cur_qty is not None else None
            ),
        }

    # Top agrupaciones
    top_g1, top_g2 = [], []
    if g1 and g1 in base_df.columns:
        top_g1 = (
            base_df.groupby(g1, dropna=False)[ac]
                  .sum()
                  .sort_values(ascending=False)
                  .head(5)
                  .reset_index()
                  .rename(columns={ac: "revenue"})
                  .to_dict(orient="records")
        )
    if g2 and g2 in base_df.columns:
        top_g2 = (
            base_df.groupby(g2, dropna=False)[ac]
                  .sum()
                  .sort_values(ascending=False)
                  .head(5)
                  .reset_index()
                  .rename(columns={ac: "revenue"})
                  .to_dict(orient="records")
        )

    # Aperturas mensuales
    monthly = _monthly_breakdowns(base_df, dc, ac, g2)

    # Clientes / frecuencia
    cust = _customer_stats(base_df, dc, ac, cu)

    payload = {
        "as_of_date": meta["as_of_date"],
        "lookback_days": int(lookback_days),
        "base_period": meta["base_period"],
        "currency": meta["currency"],
        "resolved_columns": meta["resolved_columns"],
        "overall": {
            "total_revenue": float(base_df[ac].sum() if ac in base_df.columns else 0.0),
            "total_quantity": float(base_df[qc].sum()) if (qc and qc in base_df.columns) else None,
        },
        "windows": {
            "trailing_7d": window_summary(7),
            "trailing_30d": window_summary(30),
        },
        "top_group1": top_g1,
        "top_group2": top_g2,
        "daily_outliers": [],  # (opcional) podés poblarlo luego
        "sample_rows": base_df.head(10).to_dict(orient="records"),
        "monthly_totals": monthly["monthly_totals"],
        "monthly_by_product_top": monthly["monthly_by_product_top"],
        "customer_stats": cust,
    }

    # === Métricas de camión/ruta ===
    route_col = meta["resolved_columns"].get("route")
    if route_col:
        payload["truck_metrics"] = compute_truck_metrics(
            base_df=base_df,
            date_col=dc,
            amount_col=ac,
            customer_col=cu,
            as_of=pd.to_datetime(meta["base_period"]["end"]),
            top_customers_per_truck=10
        )

    logger.info("Columnas resueltas: %s", payload["resolved_columns"])
    logger.info(
        "Base de análisis: %s → %s (lookback=%s días)",
        payload["base_period"]["start"], payload["base_period"]["end"], lookback_days
    )
    return payload

def build_prompt_from_dataframe(df: pd.DataFrame) -> str:
    """
    Construye el prompt a partir del DF original (lee/limpia/filtra/ventana),
    calcula métricas y adjunta pistas determinísticas de riesgo + logística.
    Devuelve un prompt que instruye al LLM a producir **texto narrativo en español**,
    no JSON.
    """
    base_df, meta = _prepare_analysis_df(df)

    if base_df.empty:
        return "__RAW__\n**Reporte sin datos**\n- Motivo: sin filas en ventana configurada."

    metrics = _summaries_from_base(base_df, meta)

    # Riesgo determinístico (pista adicional para el modelo)
    resolved = metrics.get("resolved_columns", {})
    risk = compute_risk(
        base_df,
        date_col=resolved.get("date", "Date"),
        amount_col=resolved.get("amount", "Amount"),
        qty_col=resolved.get("quantity", "Qty"),
        product_col=resolved.get("group2", "Producto") or "Producto",
        customer_col=resolved.get("customer", "Customer") or "Customer",
        as_of=metrics.get("as_of_date"),
        lookback_days=int(metrics.get("lookback_days", 90)),
    )
    risk_md = format_risk_markdown(risk)
    metrics["risk_hints"] = risk

    json_blob = json.dumps(metrics, ensure_ascii=False, separators=(",", ":"), default=_json_default)

    # === Instrucciones al modelo (claras y finales) ===
    # Indicamos que el JSON es solo referencia interna y pedimos un correo en castellano.
    prompt = f"""
Generá un **correo** breve y profesional en **español** (Argentina), usando el símbolo de moneda "{settings.CURRENCY}" con separadores de miles y dos decimales.
**NO** devuelvas JSON, **NO** devuelvas código, **NO** uses bloques ```; devolvé **solo el texto del correo**.

Estructura esperada (subtítulos en negrita):
- **Resumen**: 1–2 frases con lo más importante del período.
- **Clientes**: cantidad de clientes; frecuencia de compra (mediana de días entre pedidos y promedio de pedidos por cliente).
- **Logística**:
  - Rendimiento por camión: ingresos totales, viajes, promedio por viaje, paradas medianas.
  - Cadencia de visitas: mediana de días entre viajes; y por camión, top clientes con mediana de días entre visitas y días desde la última visita (si aplica).
- **Riesgos**: sintetizá si hay señales de riesgo (cuentas por cobrar/aging, top vencidos). Si no hay datos, escribí "No hay datos suficientes".

Reglas importantes:
- No inventes datos ni placeholders.
- Usá un tono claro, conservador y profesional.
- Si una sección no tiene datos, indicá "No hay datos suficientes".
- No repitas el JSON ni cifras redundantes; priorizá la lectura ejecutiva.

Datos (referencia interna; **no los devuelvas**):
{json_blob}

Pistas internas de riesgo (no las copies literal, usalas para enriquecer la sección de Riesgos):
{risk_md}
""".strip()

    return prompt
