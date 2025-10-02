# app/core/logistics.py
import os
import pandas as pd
import numpy as np

def _resolve_route_col(df: pd.DataFrame) -> str | None:
    """
    Intenta resolver la columna de camión/ruta. 
    - Si existe env ROUTE_COL, usa ese nombre (resolviendo sinónimos).
    - Si no, prueba candidatos comunes: Truck, Camion, Route, Ruta, SalesRep.
    """
    def _norm(s: str) -> str:
        return "".join(ch for ch in s.lower().strip())

    # Prioridad: env
    route_env = (os.getenv("ROUTE_COL", "") or "").strip()
    candidates = []
    if route_env:
        candidates.append(route_env)

    # Candidatos por defecto
    candidates += ["Truck", "Camion", "Route", "Ruta", "SalesRep"]

    # Mapa normalizado -> real
    norm_map = {_norm(c): c for c in df.columns}

    for cand in candidates:
        n = _norm(cand)
        if cand in df.columns:
            return cand
        if n in norm_map:
            return norm_map[n]
        # substring contain
        for nc, real in norm_map.items():
            if n and n in nc:
                return real
    return None


def compute_truck_metrics(base_df: pd.DataFrame, date_col: str, amount_col: str,
                          customer_col: str | None, as_of: pd.Timestamp,
                          top_customers_per_truck: int = 10) -> dict:
    """
    Calcula métricas de logística:
      - per_truck: ingresos totales, cantidad de viajes, ingreso promedio/viaje,
                   paradas medianas por viaje, clientes únicos, mediana días entre viajes.
      - cadence_by_customer_top: por camión, top N clientes por revenue con mediana días entre visitas y días desde última visita.

    Requisitos mínimos:
      - date_col
      - amount_col
      - customer_col (para paradas y cadencia; si no existe, se omiten algunos campos)
      - route_col (detectado automáticamente). Si no existe, devuelve {}.
    """
    if base_df.empty:
        return {}

    route_col = _resolve_route_col(base_df)
    if route_col is None:
        return {}

    df = base_df.copy()
    df["__day"] = pd.to_datetime(df[date_col]).dt.normalize()

    # === Ingresos por viaje (camión + día) ===
    trip_rev = (
        df.groupby([route_col, "__day"])[amount_col]
          .sum()
          .reset_index()
          .rename(columns={amount_col: "trip_revenue"})
    )

    # Paradas por viaje (clientes únicos ese día y camión)
    if customer_col and customer_col in df.columns:
        trip_stops = (
            df.groupby([route_col, "__day"])[customer_col]
              .nunique()
              .reset_index()
              .rename(columns={customer_col: "stops"})
        )
        trip = trip_rev.merge(trip_stops, on=[route_col, "__day"], how="left")
    else:
        trip = trip_rev.copy()
        trip["stops"] = np.nan

    # === Resumen por camión ===
    per_truck = (
        trip.groupby(route_col)
            .agg(
                total_revenue=("trip_revenue", "sum"),
                trips=("trip_revenue", "size"),
                avg_revenue_per_trip=("trip_revenue", "mean"),
                median_stops_per_trip=("stops", "median"),
            )
            .reset_index()
            .rename(columns={route_col: "truck"})
    )

    # Clientes únicos por camión
    if customer_col and customer_col in df.columns:
        uniq_cust = (
            df.groupby(route_col)[customer_col].nunique()
              .reset_index()
              .rename(columns={customer_col: "unique_customers", route_col: "truck"})
        )
        per_truck = per_truck.merge(uniq_cust, on="truck", how="left")
    else:
        per_truck["unique_customers"] = np.nan

    # Mediana de días entre viajes por camión
    trip_dates = (
        trip[[route_col, "__day"]]
        .drop_duplicates()
        .sort_values([route_col, "__day"])
    )
    gap = (
        trip_dates.groupby(route_col)["__day"]
                  .apply(lambda s: s.sort_values().diff().median())
                  .reset_index()
                  .rename(columns={"__day": "median_gap"})
    )
    if not gap.empty and pd.api.types.is_timedelta64_dtype(gap["median_gap"]):
        gap["median_days_between_trips"] = gap["median_gap"].dt.days
    else:
        gap["median_days_between_trips"] = np.nan
    gap = gap.drop(columns=["median_gap"])
    gap = gap.rename(columns={route_col: "truck"})
    per_truck = per_truck.merge(gap, on="truck", how="left")

    # === Cadencia por cliente (top por camión) ===
    cadence_by_customer_top = []
    if customer_col and customer_col in df.columns:
        # Mediana de días entre visitas por (camión, cliente)
        pair_dates = (
            df[[route_col, customer_col, "__day"]]
            .drop_duplicates()
            .sort_values([route_col, customer_col, "__day"])
        )
        pair_gap = (
            pair_dates.groupby([route_col, customer_col])["__day"]
                      .apply(lambda s: s.sort_values().diff().median())
                      .reset_index()
                      .rename(columns={"__day": "median_gap"})
        )
        if not pair_gap.empty and pd.api.types.is_timedelta64_dtype(pair_gap["median_gap"]):
            pair_gap["median_days_between_visits"] = pair_gap["median_gap"].dt.days
        else:
            pair_gap["median_days_between_visits"] = np.nan
        pair_gap = pair_gap.drop(columns=["median_gap"])

        # Revenue por (camión, cliente)
        rev_pair = (
            df.groupby([route_col, customer_col])[amount_col]
              .sum()
              .reset_index()
              .rename(columns={amount_col: "revenue"})
        )

        # Última visita y días desde última visita
        last_visit = (
            df.groupby([route_col, customer_col])["__day"]
              .max()
              .reset_index()
              .rename(columns={"__day": "last_visit"})
        )
        last_visit["days_since_last"] = (pd.to_datetime(as_of) - pd.to_datetime(last_visit["last_visit"])).dt.days

        cad = pair_gap.merge(rev_pair, on=[route_col, customer_col], how="left")
        cad = cad.merge(last_visit, on=[route_col, customer_col], how="left")

        # Ordenar por revenue dentro de cada camión y cortar top N
        cad = cad.sort_values([route_col, "revenue"], ascending=[True, False])
        cad["truck"] = cad[route_col]
        cad = cad.drop(columns=[route_col])
        # top N por camión
        cad = (
            cad.groupby("truck", group_keys=False)
               .apply(lambda g: g.head(top_customers_per_truck))
               .reset_index(drop=True)
        )
        cadence_by_customer_top = cad.to_dict(orient="records")

    return {
        "route_col": _resolve_route_col(base_df),
        "per_truck": per_truck.to_dict(orient="records"),
        "cadence_by_customer_top": cadence_by_customer_top,
    }
