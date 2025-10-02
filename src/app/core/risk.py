# app/risk.py
from __future__ import annotations
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)

# ---------- Helpers de configuración ----------
def _get_float(env: str, default: float) -> float:
    try:
        return float(os.getenv(env, str(default)))
    except Exception:
        return default

def _get_int(env: str, default: int) -> int:
    try:
        return int(os.getenv(env, str(default)))
    except Exception:
        return default

def _get_bool(env: str, default: bool) -> bool:
    val = os.getenv(env)
    if val is None:
        return default
    return val.strip() in ("1", "true", "True", "yes", "YES")

def _get_list_int(env: str, default: List[int]) -> List[int]:
    raw = os.getenv(env)
    if not raw:
        return default
    out = []
    for p in raw.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            pass
    return out or default

# ---------- Dataclasses de salida ----------
@dataclass
class RiskSection:
    title: str
    bullets: List[str]

@dataclass
class RiskReport:
    has_findings: bool
    sections: List[RiskSection]
    notes: List[str]
    skipped_reasons: List[str]

    def to_dict(self) -> Dict:
        # 100% JSON serializable (sólo strings y listas)
        return {
            "has_findings": bool(self.has_findings),
            "sections": [{"title": s.title, "bullets": list(s.bullets)} for s in self.sections],
            "notes": list(self.notes),
            "skipped_reasons": list(self.skipped_reasons),
        }

# ---------- Cálculo principal ----------
def compute_risk(
    df: pd.DataFrame,
    *,
    date_col: str = "Date",
    amount_col: str = "Amount",
    qty_col: str = "Qty",
    product_col: str = "Producto",
    customer_col: str = "Customer",
    as_of: Optional[date] = None,
    lookback_days: int = 90,
) -> Dict:
    """
    Motor determinístico de “riesgos / outliers” basado en los últimos N días.
    Devuelve un dict JSON-serializable para adjuntar al prompt o imprimir.
    """
    enable = _get_bool("RISK_ENABLE", True)
    if not enable:
        return RiskReport(False, [], [], ["RISK_ENABLE=0"]).to_dict()

    # Umbrales configurables (.env)
    thr_top_sku = _get_float("RISK_TOP_SKU_SHARE", 0.40)           # 40%
    thr_top_cust = _get_float("RISK_TOP_CUST_SHARE", 0.35)         # 35%
    thr_price_jump = _get_float("RISK_PRICE_JUMP_PCT", 0.20)       # 20%
    thr_z = _get_float("RISK_ZSCORE_THRESHOLD", 3.0)               # z-score robusto
    min_tx_per_sku = _get_int("RISK_MIN_TX_PER_SKU", 4)
    min_active_days = _get_int("RISK_MIN_ACTIVE_DAYS", 5)
    aging_buckets = _get_list_int("RISK_AGING_BUCKETS", [30, 60, 90])
    default_terms = _get_int("RISK_DEFAULT_TERMS_DAYS", 30)

    notes: List[str] = []
    skipped: List[str] = []
    sections: List[RiskSection] = []

    # Columnas mínimas
    needed = [date_col, amount_col, qty_col, product_col, customer_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return RiskReport(False, [], [], [f"Faltan columnas: {', '.join(missing)}"]).to_dict()

    # Fechas y ventana
    sdate = pd.to_datetime(df[date_col], errors="coerce")
    if not sdate.notna().any():
        return RiskReport(False, [], [], ["No hay fechas válidas para riesgo"]).to_dict()

    if as_of is None:
        as_of_pd = sdate.max().normalize()
    else:
        as_of_pd = pd.to_datetime(as_of).normalize()
    start = as_of_pd - pd.Timedelta(days=lookback_days - 1)

    base = df.copy()
    base[date_col] = sdate
    base = base[(base[date_col] >= start) & (base[date_col] <= as_of_pd)].copy()
    base[amount_col] = pd.to_numeric(base[amount_col], errors="coerce")
    base[qty_col] = pd.to_numeric(base[qty_col], errors="coerce")

    if base.empty:
        return RiskReport(False, [], [], ["Sin datos en la ventana temporal"]).to_dict()

    # ---------------- 1) Concentración (SKU / Cliente)
    sec_conc: List[str] = []
    by_sku = base.groupby(product_col, dropna=False)[amount_col].sum().sort_values(ascending=False)
    total_amt = float(by_sku.sum()) or 1.0
    if len(by_sku) > 0:
        top1_share = float(by_sku.iloc[0]) / total_amt
        top3_share = float(by_sku.head(3).sum()) / total_amt
        line = f"Top SKU = {by_sku.index[0]!s}: {by_sku.iloc[0]:,.0f} USD ({top1_share:.1%} del total)"
        if top1_share >= thr_top_sku:
            line += " ⚠️"
        sec_conc.append(line)
        sec_conc.append(f"Top 3 SKUs concentran {top3_share:.1%} del total")
        if top3_share >= max(0.65, thr_top_sku + 0.20):
            sec_conc[-1] += " ⚠️"
    by_cust = base.groupby(customer_col, dropna=False)[amount_col].sum().sort_values(ascending=False)
    if len(by_cust) > 0:
        top1_share_c = float(by_cust.iloc[0]) / (float(by_cust.sum()) or 1.0)
        line = f"Top cliente = {by_cust.index[0]!s}: {by_cust.iloc[0]:,.0f} USD ({top1_share_c:.1%} del total)"
        if top1_share_c >= thr_top_cust:
            line += " ⚠️"
        sec_conc.append(line)

    if sec_conc:
        sections.append(RiskSection("Concentración", sec_conc))

    # ---------------- 2) Volatilidad / outliers (z-score robusto por día)
    sec_vol: List[str] = []
    daily = base.groupby(base[date_col].dt.date).agg(
        amount_sum=(amount_col, "sum"),
        qty_sum=(qty_col, "sum"),
    ).sort_index()

    def robust_z(x: pd.Series) -> pd.Series:
        med = x.median()
        mad = (x - med).abs().median()
        if mad == 0:
            std = x.std(ddof=0) or 1.0
            return (x - med) / std
        return 0.6745 * (x - med) / mad

    if len(daily) >= min_active_days:
        z_amt = robust_z(daily["amount_sum"])
        z_qty = robust_z(daily["qty_sum"])
        spikes_amt = daily[abs(z_amt) >= thr_z].copy()
        spikes_qty = daily[abs(z_qty) >= thr_z].copy()

        def _fmt_spikes(spikes: pd.DataFrame, col: str, z: pd.Series, label: str) -> List[str]:
            out = []
            tmp = spikes.assign(z=z.loc[spikes.index]).copy()
            tmp = tmp.reindex(tmp["z"].abs().sort_values(ascending=False).index)
            for d, row in tmp.head(5).iterrows():
                out.append(f"{label} {d}: {row[col]:,.0f} (z={row['z']:.1f}) ⚠️")
            return out

        if not spikes_amt.empty:
            sec_vol += _fmt_spikes(spikes_amt, "amount_sum", z_amt, "Monto outlier el")
        if not spikes_qty.empty:
            sec_vol += _fmt_spikes(spikes_qty, "qty_sum", z_qty, "Qty outlier el")

    if sec_vol:
        sections.append(RiskSection("Volatilidad / outliers", sec_vol))
    else:
        skipped.append("Sin masa crítica para outliers diarios o sin picos significativos")

    # ---------------- 3) Anomalías de precio por SKU (30d vs 31-60d)
    sec_price: List[str] = []
    price_col = None
    for cand in ("SalesPrice", "UnitPrice", "Price"):
        if cand in base.columns:
            price_col = cand
            break
    if price_col:
        base["_day"] = base[date_col].dt.date
        last30_start = (as_of_pd - pd.Timedelta(days=29)).date()
        prev30_start = (as_of_pd - pd.Timedelta(days=59)).date()
        prev30_end = (as_of_pd - pd.Timedelta(days=30)).date()

        last30 = base[base["_day"] >= last30_start]
        prev30 = base[(base["_day"] >= prev30_start) & (base["_day"] <= prev30_end)]

        g1 = last30.groupby(product_col)[price_col].median()
        g0 = prev30.groupby(product_col)[price_col].median()
        joined = pd.concat([g0.rename("m_prev"), g1.rename("m_last")], axis=1).dropna()
        joined = joined[(joined["m_prev"] > 0) & (joined["m_last"] > 0)]

        if not joined.empty:
            joined["pct"] = (joined["m_last"] - joined["m_prev"]) / joined["m_prev"]
            # Impacto económico aproximado por SKU en los últimos 30d
            impact = last30.groupby(product_col)[amount_col].sum()
            joined = joined.join(impact.rename("amt_last30"), how="left").fillna({"amt_last30": 0})
            # Filtrar por masa crítica mínima
            tx_count = last30.groupby(product_col).size()
            joined = joined.join(tx_count.rename("n_tx_last30"), how="left")
            joined = joined[joined["n_tx_last30"] >= min_tx_per_sku]

            # Flags de salto de precio
            abn = joined[joined["pct"].abs() >= thr_price_jump].sort_values(
                ["amt_last30", "pct"], ascending=[False, False]
            )
            for sku, row in abn.head(8).iterrows():
                sign = "↑" if row["pct"] > 0 else "↓"
                sec_price.append(
                    f"Precio {sign} {sku}: {row['pct']:.1%} (mediana 30d={row['m_last']:.2f} vs prev30={row['m_prev']:.2f}, impacto 30d≈{row['amt_last30']:,.0f} USD) ⚠️"
                )

        if sec_price:
            sections.append(RiskSection("Anomalías de precio", sec_price))
        else:
            skipped.append("Sin saltos de precio > umbral o masa crítica insuficiente (RISK_PRICE_JUMP_PCT)")
    else:
        skipped.append("Columna de precio no encontrada (SalesPrice/UnitPrice/Price)")

    # ---------------- 4) Riesgo de cobranza (aging simple con términos por defecto)
    sec_ar: List[str] = []
    if "Balance" in base.columns:
        base["Balance"] = pd.to_numeric(base["Balance"], errors="coerce").fillna(0.0)
        ar = base[base["Balance"] > 0].copy()
        ar["_days"] = (as_of_pd - ar[date_col]).dt.days
        # Buckets 0-30, 31-60, 61-90, >90 (o según config)
        b = sorted(aging_buckets)
        labels = []
        edges = [0] + b + [10_000]  # límite grande
        for i in range(len(edges)-1):
            a, c = edges[i], edges[i+1]
            if c == 10_000:
                labels.append(f">{edges[i]}")
            else:
                labels.append(f"{a}-{c}")
        ar["_bucket"] = pd.cut(ar["_days"], bins=edges, labels=labels, right=True, include_lowest=True)
        aging_tbl = ar.groupby("_bucket")["Balance"].sum().reindex(labels).fillna(0.0)

        # Vencidos según términos por defecto
        overdue = ar[ar["_days"] > default_terms]
        top_over = overdue.groupby(customer_col)["Balance"].sum().sort_values(ascending=False).head(10)

        sec_ar.append("Aging por buckets (USD): " + ", ".join(f"{k}: {v:,.0f}" for k, v in aging_tbl.items()))
        if not top_over.empty:
            for cust, bal in top_over.items():
                sec_ar.append(f"Cliente con saldo vencido >{default_terms} días: {cust} = {bal:,.0f} USD ⚠️")

        if sec_ar:
            sections.append(RiskSection("Cobranzas (AR aging)", sec_ar))
    else:
        skipped.append("Sin columna Balance para aging de cobranzas")

    has_findings = any(len(s.bullets) > 0 for s in sections)
    if not has_findings and not skipped:
        skipped.append("No se detectaron señales por encima de umbrales configurados")

    # Notas útiles
    notes.append(f"Ventana analizada: {start.date()} → {as_of_pd.date()} (lookback={lookback_days}d)")
    notes.append(f"Umbrales: top_sku≥{thr_top_sku:.0%}, top_cliente≥{thr_top_cust:.0%}, Δprecio≥{thr_price_jump:.0%}, z≥{thr_z:g}")

    return RiskReport(has_findings, sections, notes, skipped).to_dict()

# ---------- Render a Markdown ----------
def format_risk_markdown(report: Dict) -> str:
    if not report:
        return "**Riesgos / outliers (motor determinístico)**\nNo hay información."
    has = report.get("has_findings")
    sections = report.get("sections", [])
    notes = report.get("notes", [])
    skipped = report.get("skipped_reasons", [])

    lines = ["**Riesgos / outliers (motor determinístico)**"]
    if sections:
        for sec in sections:
            title = sec.get("title", "Sección")
            bullets = sec.get("bullets", [])
            if not bullets:
                continue
            lines.append(f"- **{title}**")
            for b in bullets:
                lines.append(f"  - {b}")

    if not any(sec.get("bullets") for sec in sections):
        lines.append("No se detectaron riesgos por encima de umbrales en esta corrida.")

    if skipped:
        lines.append("- **Chequeos omitidos / razones**")
        for s in skipped:
            lines.append(f"  - {s}")

    if notes:
        lines.append("- **Notas**")
        for n in notes:
            lines.append(f"  - {n}")

    return "\n".join(lines)

# app/core/risk.py (añadir/actualizar)

import pandas as pd
import numpy as np

def _coerce_num_inplace(df, cols):
    for c in cols:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def _resolve_simple(df, preferred, synonyms):
    """Devuelve el nombre de columna si existe (preferred o sinónimos), sino None."""
    if preferred and preferred in df.columns:
        return preferred
    for s in (synonyms or []):
        if s in df.columns:
            return s
    return None

def _compute_ar_aging(df: pd.DataFrame, *, date_col: str, balance_col: str,
                      customer_col: str | None = None, invoice_col: str | None = None,
                      as_of: pd.Timestamp) -> dict:
    """
    Calcula aging de CxC evitando doble conteo:
    - Agrupa por factura (invoice_col) y toma la ÚLTIMA observación por fecha.
    - Si no hay invoice_col, agrupa por (customer_col, date) de forma conservadora.
    - Suma SOLO saldos positivos (deudores). Los negativos (notas de crédito) no inflan buckets.
    """
    df = df.copy()
    # sanity coercion
    _coerce_num_inplace(df, [balance_col])
    df[balance_col] = df[balance_col].fillna(0.0)
    df = df[np.isfinite(df[balance_col])]

    # Filtrar filas sin fecha (no se puede agear)
    sdate = pd.to_datetime(df[date_col], errors="coerce")
    df = df.loc[sdate.notna()].copy()
    df[date_col] = sdate

    # 1) Deduplicación por factura
    if invoice_col and invoice_col in df.columns:
        # Última observación por (invoice[, customer]) según fecha
        sort_cols = [date_col]
        gcols = [invoice_col]
        if customer_col and customer_col in df.columns:
            gcols = [customer_col, invoice_col]

        df_sorted = df.sort_values(sort_cols)
        last_per_invoice = df_sorted.groupby(gcols, dropna=False).tail(1)
        base = last_per_invoice[[date_col, balance_col]]
        if customer_col and customer_col in last_per_invoice.columns:
            base[customer_col] = last_per_invoice[customer_col].values
        if invoice_col in last_per_invoice.columns:
            base[invoice_col] = last_per_invoice[invoice_col].values
    else:
        # Fallback: tomar última por (customer) si no hay número de factura
        gcols = [customer_col] if (customer_col and customer_col in df.columns) else []
        if gcols:
            df_sorted = df.sort_values([date_col])
            base = df_sorted.groupby(gcols, dropna=False).tail(1)[[date_col, balance_col] + gcols]
        else:
            # Como último recurso, usar todo tal cual (menos ideal)
            base = df[[date_col, balance_col]].copy()

    # 2) Días de atraso y sólo saldos positivos
    base["days_overdue"] = (as_of - base[date_col]).dt.days
    pos = base[base[balance_col] > 0].copy()

    # 3) Buckets
    b0_30   = float(pos.loc[(pos["days_overdue"] >= 0) & (pos["days_overdue"] <= 30), balance_col].sum())
    b30_60  = float(pos.loc[(pos["days_overdue"] >= 31) & (pos["days_overdue"] <= 60), balance_col].sum())
    b60_90  = float(pos.loc[(pos["days_overdue"] >= 61) & (pos["days_overdue"] <= 90), balance_col].sum())
    bgt90   = float(pos.loc[(pos["days_overdue"] >= 91), balance_col].sum())
    total   = b0_30 + b30_60 + b60_90 + bgt90

    # 4) Top vencidos
    cols_show = []
    if customer_col and customer_col in base.columns: cols_show.append(customer_col)
    if invoice_col and invoice_col in base.columns: cols_show.append(invoice_col)
    cols_show += [balance_col, "days_overdue"]

    top_overdue = (
        pos.sort_values(["days_overdue", balance_col], ascending=[False, False])
           [cols_show].head(10).to_dict(orient="records")
    )

    return {
        "buckets": {"0_30": b0_30, "30_60": b30_60, "60_90": b60_90, ">90": bgt90, "total": total},
        "top_overdue": top_overdue,
        "count_invoices": int(len(pos)),
    }

def compute_risk(df, date_col, amount_col, qty_col=None,
                 product_col=None, customer_col=None,
                 as_of=None, lookback_days=90,
                 balance_col=None, price_col=None, **kwargs):
    # ... tu código existente arriba ...

    # === AR AGING (evitar doble conteo por líneas) ===
    # Resolver columnas si no vinieron:
    if not balance_col:
        balance_col = _resolve_simple(df, "Balance",
                        ["Balance","Saldo","AR","Accounts Receivable","Cuentas por cobrar"])
    invoice_col = _resolve_simple(df, "Num", ["Num","Invoice","Nro","Numero","Número"])

    # Timestamp as_of
    as_of_ts = pd.to_datetime(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()

    ar_section = None
    if balance_col and balance_col in df.columns:
        try:
            ar_section = _compute_ar_aging(
                df,
                date_col=date_col,
                balance_col=balance_col,
                customer_col=customer_col if (customer_col and customer_col in df.columns) else None,
                invoice_col=invoice_col if (invoice_col and invoice_col in df.columns) else None,
                as_of=as_of_ts,
            )
        except Exception as e:
            ar_section = {"error": f"aging_failed: {e}"}
    else:
        ar_section = {"note": "No hay columna de balance disponible para aging."}

    # Agregalo a tu payload final antes de devolver:
    out = {
        # ... el resto de tu payload de riesgo ...
        "ar_aging": ar_section,
    }
    return out
