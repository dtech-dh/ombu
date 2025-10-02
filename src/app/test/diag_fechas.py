# app/diag_fechas.py
import pandas as pd
from ..servicios.sheets import read_sheet_to_df
from ..config import settings

def main():
    df = read_sheet_to_df()
    dc = settings.SALES_DATE_COL
    print("Columnas DF:", list(df.columns))
    if dc in df.columns:
        s = pd.to_datetime(df[dc], errors="coerce")
        print("Fecha min:", s.min())
        print("Fecha max:", s.max())
        as_of = s.max().normalize()
        lookback = int(getattr(settings, "SALES_LOOKBACK_DAYS", 90))
        base_start = as_of - pd.Timedelta(days=lookback-1)
        base_df = df[(s >= base_start) & (s <= as_of)]
        print(f"Ventana base {base_start.date()} → {as_of.date()}  (lookback={lookback} días)")
        print("Filas en ventana:", len(base_df))
        print(base_df.head(10).to_string(index=False))
    else:
        print(f"La columna de fecha '{dc}' no está en el DataFrame")

if __name__ == "__main__":
    main()
