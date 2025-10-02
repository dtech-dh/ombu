# app/ai.py
import logging
import os
import json
import unicodedata
from datetime import timedelta
import httpx
import pandas as pd
from openai import OpenAI
from ..config import settings
from ..core.risk import compute_risk, format_risk_markdown


logger = logging.getLogger(__name__)


import numpy as np
from datetime import date, datetime

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

def _make_openai_client() -> OpenAI:
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    http_client = httpx.Client(proxies=proxy) if proxy else httpx.Client()
    return OpenAI(api_key=settings.OPENAI_API_KEY, http_client=http_client)

def _norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    for ch in [",", ".", "(", ")", "[", "]", "{", "}", "$", "%", "#", "!", "?", ":", ";", "'", '"']:
        s = s.replace(ch, " ")
    s = s.replace("-", " ").replace("_", " ").replace("/", " ")
    s = " ".join(s.split())
    return s

def _resolve_col(df: pd.DataFrame, desired: str, synonyms: list[str]) -> str | None:
    cols = list(df.columns)
    if not cols:
        return None
    norm_map = {_norm(c): c for c in cols}
    if desired in df.columns:
        return desired
    nd = _norm(desired)
    if nd in norm_map:
        return norm_map[nd]
    for nc, real in norm_map.items():
        if nd and nd in nc:
            return real
    for syn in synonyms:
        ns = _norm(syn)
        if syn in df.columns:
            return syn
        if ns in norm_map:
            return norm_map[ns]
    for syn in synonyms:
        ns = _norm(syn)
        for nc, real in norm_map.items():
            if ns and ns in nc:
                return real
    return None

def _ensure_datetime(series):
    try:
        return pd.to_datetime(series, errors="coerce", dayfirst=True)
    except Exception:
        return pd.to_datetime(series, errors="coerce")

def _summaries(df: pd.DataFrame) -> dict:
    dc_req = settings.SALES_DATE_COL
    ac_req = settings.SALES_AMOUNT_COL
    qc_req = settings.SALES_QTY_COL
    g1_req = settings.GROUP_BY_1
    g2_req = settings.GROUP_BY_2

    date_syn = [dc_req, "date", "fecha", "order date", "fecha venta"]
    amt_syn  = [ac_req, "revenue", "amount", "importe", "monto", "total", "venta", "sales", "net sales", "total ventas", "importe total"]
    qty_syn  = [qc_req, "qty", "quantity", "unidades", "cantidad"]
    g1_syn   = [g1_req, "region", "zona", "sucursal", "canal", "category", "categoria"]
    g2_syn   = [g2_req, "sku", "producto", "item", "id producto", "ref"]

    dc = _resolve_col(df, dc_req, date_syn)
    ac = _resolve_col(df, ac_req, amt_syn)
    qc = _resolve_col(df, qc_req, qty_syn) if qc_req else None
    g1 = _resolve_col(df, g1_req, g1_syn) if g1_req else None
    g2 = _resolve_col(df, g2_req, g2_syn) if g2_req else None

    unresolved = []
    if dc is None: unresolved.append(f"Fecha ({dc_req})")
    if ac is None: unresolved.append(f"Importe ({ac_req})")
    if unresolved:
        return {"error": "Columnas requeridas faltantes", "missing": unresolved, "available": list(df.columns)}

    df = df.copy()
    df[dc] = _ensure_datetime(df[dc])
    df = df.dropna(subset=[dc])
    df[ac] = pd.to_numeric(df[ac], errors="coerce")
    df = df.dropna(subset=[ac])
    if qc:
        df[qc] = pd.to_numeric(df[qc], errors="coerce")

    if df.empty:
        return {"error": "DF vacío tras tipificar", "available": list(df.columns)}

    as_of = pd.to_datetime(df[dc]).max().normalize()
    lookback_days = getattr(settings, "SALES_LOOKBACK_DAYS", 90)
    base_start = as_of - pd.Timedelta(days=lookback_days - 1)
    base_df = df[(df[dc] >= base_start) & (df[dc] <= as_of)]
    if base_df.empty:
        base_df = df.copy()
        base_start = pd.to_datetime(df[dc]).min().normalize()

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

    payload = {
        "as_of_date": str(as_of.date()),
        "lookback_days": int(lookback_days),
        "base_period": {"start": str(base_start.date()), "end": str(as_of.date())},
        "currency": settings.CURRENCY,
        "resolved_columns": {"date": dc, "amount": ac, "quantity": qc, "group1": g1, "group2": g2},
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
        "daily_outliers": [],
        "sample_rows": base_df.head(10).to_dict(orient="records"),
    }
    logger.info("Columnas resueltas: %s", payload["resolved_columns"])
    logger.info("Base de análisis: %s → %s (lookback=%s días)",
                payload["base_period"]["start"], payload["base_period"]["end"], lookback_days)
    return payload

def build_prompt_from_dataframe(df: pd.DataFrame) -> str:
    metrics = _summaries(df)
    # --- Inserción: cálculo de riesgo determinístico (últimos N días)
    resolved = metrics.get("resolved_columns", {})
    risk = compute_risk(
        df,
        date_col=resolved.get("date", "Date"),
        amount_col=resolved.get("amount", "Amount"),
        qty_col=resolved.get("quantity", "Qty"),
        product_col=resolved.get("group2", "Producto") or "Producto",
        customer_col="Customer",
        as_of=metrics.get("as_of_date"),
        lookback_days=int(metrics.get("lookback_days", 90)),
    )
    risk_md = format_risk_markdown(risk)

    # Guardamos las pistas en el JSON del prompt para la IA (serializable)
    metrics["risk_hints"] = risk

    if "error" in metrics:
        msg = [
            "**Reporte sin datos (preparación fallida)**",
            f"- Motivo: {metrics.get('error')}",
        ]
        if "missing" in metrics:
            msg.append(f"- Faltan columnas: {', '.join(metrics['missing'])}")
        if "available" in metrics:
            msg.append(f"- Columnas disponibles: {', '.join(map(str, metrics['available']))}")
        return "__RAW__\n" + "\n".join(msg)

    json_blob = json.dumps( metrics, ensure_ascii=False, separators=(",", ":"), default=_json_default  )

    header = (
        f"La base de análisis son los últimos {metrics.get('lookback_days')} días "
        f"({metrics['base_period']['start']} → {metrics['base_period']['end']})."
    )

    prompt = f"""
{header}
Tenés métricas de ventas en formato JSON (Argentina, lenguaje español). Pensá paso a paso internamente y **devolvé sólo** el resultado final en el formato solicitado.

JSON:
```
{json_blob}
```

Reglas importantes:
- No inventes datos ni placeholders. Si una sección no tiene datos suficientes, escribí: "No hay datos suficientes".
- Usá el símbolo de moneda "{settings.CURRENCY}" y separadores de miles.

Formato de salida (Markdown mínimo, sin metadatos ni explicar tu proceso):
- **Resumen ejecutivo**
- **Insights claves**
- **Riesgos / outliers**
- **Acciones priorizadas (5)**
- **Notas**
""".strip()
    prompt += "\n\n" + risk_md

    return prompt

def get_ai_report(prompt: str) -> str:
    if prompt.startswith("__RAW__"):
        return prompt.split("\n", 1)[1].strip()

    logger.info("Consultando OpenAI...")
    client = _make_openai_client()
    resp = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        temperature=settings.OPENAI_TEMPERATURE,
        max_tokens=settings.OPENAI_MAX_TOKENS,
        messages=[
            {"role": "system", "content": "Sos un asistente de análisis para negocio. Responde claro, conciso y accionable; no muestres tu proceso."},
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content.strip()
    logger.info("OpenAI completó la respuesta.")
    return content
