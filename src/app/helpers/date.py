# app/filters.py
from __future__ import annotations
import os
from typing import Tuple, Optional
import pandas as pd

def _now_anchor() -> pd.Timestamp:
    """
    Devuelve 'hoy' normalizado (00:00) según TZ opcional (env TIMEZONE).
    Si no hay TZ válida, usa hora local del contenedor.
    """
    tz = os.getenv("TIMEZONE")  # ej: "America/Argentina/Cordoba"
    try:
        if tz:
            # normaliza a 00:00 y vuelve naive (sin TZ) para comparaciones con series naive
            return pd.Timestamp.now(tz=tz).normalize().tz_convert(None)
    except Exception:
        pass
    return pd.Timestamp.today().normalize()

def filter_by_lookback(
    df: pd.DataFrame,
    date_col: str,
    lookback_days: int,
    *,
    anchor: str = "today",  # "today" | "max_data"
) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    Filtra df para dejar solo filas con date_col en [start, as_of],
    donde start = as_of - (lookback_days - 1).

    anchor="today": usa la fecha de hoy como 'as_of'.
    anchor="max_data": usa la fecha máxima presente en los datos como 'as_of'.

    Devuelve: (df_filtrado, start, as_of)
    """
    s = pd.to_datetime(df[date_col], errors="coerce")
    if anchor.lower() == "max_data":
        as_of = s.max().normalize()
    else:
        as_of = _now_anchor()

    start = as_of - pd.Timedelta(days=lookback_days - 1)
    mask = (s >= start) & (s <= as_of)
    out = df.loc[mask].copy()
    # tipifica la columna a datetime coerced filtrada
    out[date_col] = s.loc[out.index]
    return out, start, as_of
