# app/test/diag_balances.py
from app.servicios.sheets import read_sheet_to_df
from app.core.prompt_builder import _resolve_col, _ensure_datetime
import pandas as pd
import numpy as np

def pct_non_numeric(s: pd.Series) -> float:
    try:
        s2 = pd.to_numeric(s, errors="coerce")
        return float(s2.isna().mean())
    except Exception:
        return 1.0

def sample_weird(s: pd.Series, n=10):
    s = s.astype(str)
    mask = ~s.str.match(r"^-?\d+([.,]\d+)?$")
    return s[mask].dropna().unique()[:n]

def main():
    df = read_sheet_to_df()
    ac = _resolve_col(df, "Amount", ["Amount","Importe","Monto","Revenue","Net Sales","Total"])
    bc = _resolve_col(df, "Balance", ["Balance","Saldo","AR","Accounts Receivable","Cuentas por cobrar"])
    qc = _resolve_col(df, "Qty", ["Qty","Quantity","Unidades","Cantidad"])
    dc = _resolve_col(df, "Date", ["Date","Fecha","Order Date","Invoice Date"])

    print("Cols:", list(df.columns))
    for col in [ac, bc, qc]:
        if col:
            print(f"\n>>> {col} dtype={df[col].dtype}")
            print(f"   % no-numérico (to_numeric NaN): {pct_non_numeric(df[col]):.2%}")
            print(f"   ejemplos raros: {sample_weird(df[col])}")

    # Sumas como están (peligro si son strings)
    if ac:
        try:
            print("\nSUM(Amount) SIN COERCER:", df[ac].sum())
        except Exception as e:
            print("ERROR sum Amount:", e)
        # Suma segura
        print("SUM(Amount) COERCE:", pd.to_numeric(df[ac], errors="coerce").sum())

    if bc:
        try:
            print("\nSUM(Balance) SIN COERCER:", df[bc].sum())
        except Exception as e:
            print("ERROR sum Balance:", e)
        print("SUM(Balance) COERCE:", pd.to_numeric(df[bc], errors="coerce").sum())

if __name__ == "__main__":
    main()
