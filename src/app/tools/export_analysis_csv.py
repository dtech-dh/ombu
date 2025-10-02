# app/tools/export_analysis_csv.py
from __future__ import annotations
import os
from datetime import datetime
import pandas as pd

from app.servicios.sheets import read_sheet_to_df
from app.core.prompt_builder import get_analysis_dataframe

def main():
    out_dir = os.getenv("EXPORT_DIR", "/app/out")
    os.makedirs(out_dir, exist_ok=True)

    df_raw = read_sheet_to_df()
    df_final, meta = get_analysis_dataframe(df_raw)

    start = meta["base_period"]["start"]
    end   = meta["base_period"]["end"]
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    fname = f"analysis_final_{start}_{end}_{ts}.csv"
    path = os.path.join(out_dir, fname)

    # CSV exacto (sin truncar ni excluir más filas)
    df_final.to_csv(path, index=False, encoding="utf-8-sig", sep=";")

    print("== EXPORT ANALYSIS CSV ==")
    print("Filas:", len(df_final))
    print("Rango:", start, "→", end)
    print("Archivo:", path)

if __name__ == "__main__":
    main()
