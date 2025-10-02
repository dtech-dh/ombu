# app/main.py
import logging
import logging.config
from datetime import datetime
import os
import traceback
import pandas as pd

from .config import settings
from .servicios.sheets import read_sheet_to_df
from .core.prompt_builder import build_prompt_from_dataframe
from .servicios.openai_ai import get_openai_report
from .servicios.anthropic_ai import get_anthropic_report
from .servicios.emailer import send_email

def setup_logging():
    use_conf = os.getenv("USE_LOGCONF", "0")
    if use_conf != "1":
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        for name in ["prompt_builder", "openai_ai", "anthropic_ai", "sheets", "main", "googleapiclient.discovery_cache"]:
            logging.getLogger(name).setLevel(os.getenv("LOG_LEVEL", "INFO"))
        return
    conf_path = os.path.join(os.path.dirname(__file__), "logging.conf")
    if os.path.exists(conf_path):
        logging.config.fileConfig(conf_path)
    else:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

def _md_to_html(md_text: str) -> str:
    try:
        import markdown2
        return markdown2.markdown(md_text)
    except Exception:
        from html import escape
        return f"<div style='white-space:pre-wrap'>{escape(md_text)}</div>"

def _send_report_email(subject_prefix: str, provider_tag: str, md_body: str):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    subject = f"{subject_prefix} [{provider_tag}] {now_str}"
    send_email(subject=subject, html_body=_md_to_html(md_body))

def main():
    setup_logging()
    logger = logging.getLogger("main")
    subject_prefix = getattr(settings, "REPORT_SUBJECT_PREFIX", "Reporte de Ventas")

    logger.info("== Iniciando gsheets-ai-mailer ==")
    try:
        df = read_sheet_to_df()
    except Exception as e:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        subject = f"{subject_prefix} (ERROR LEYENDO SHEET) {now_str}"
        body = f"Error leyendo Google Sheet:\n\n{e}\n\n{traceback.format_exc()}"
        try:
            send_email(subject=subject, html_body=_md_to_html(body))
        except Exception:
            logger.exception("Fallo enviando email de error.")
        raise

    if df.empty:
        logger.warning("DataFrame vacío. No se generará reporte de IA.")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        subject = f"{subject_prefix} (SIN DATOS) {now_str}"
        body = "No se encontraron filas válidas para analizar tras la preparación del DataFrame.\n" f"Columnas: {list(df.columns)}"
        try:
            send_email(subject=subject, html_body=_md_to_html(body))
        except Exception:
            logger.exception("Fallo enviando email de 'sin datos'.")
        logger.info("== Proceso finalizado ==")
        return

    logger.info("Filas leídas del Sheet: %d | Columnas: %d", len(df), len(df.columns))
    dc = settings.SALES_DATE_COL
    if dc in df.columns:
        try:
            s = pd.to_datetime(df[dc], errors="coerce")
            fmin = s.min().date() if s.notna().any() else None
            fmax = s.max().date() if s.notna().any() else None
            logger.info("Rango de fechas en DF: %s → %s", fmin, fmax)
        except Exception:
            logger.exception("No pude calcular rango de fechas del DF.")

    # Prompt único (separado de los proveedores)
    prompt = build_prompt_from_dataframe(df)

    # Si es RAW (validación), lo enviamos igual por ambos proveedores
    if isinstance(prompt, str) and prompt.startswith("__RAW__"):
        report_md = prompt.split("\n", 1)[1].strip()
        _send_report_email(subject_prefix, "OpenAI", report_md)
        _send_report_email(subject_prefix, "Claude", report_md)
        logger.info("Reporte RAW enviado (validación de datos) por ambos proveedores.")
        logger.info("== Proceso finalizado ==")
        return

    # OpenAI
    try:
        report_openai = get_openai_report(prompt)
        _send_report_email(subject_prefix, "OpenAI", report_openai)
        logger.info("Reporte OpenAI enviado.")
    except Exception:
        logger.exception("Error generando o enviando el reporte OpenAI.")

    # Anthropic (segunda opinión)
    try:
        report_claude = get_anthropic_report(prompt)
        if isinstance(report_claude, str) and report_claude.startswith("__RAW__"):
            report_claude = report_claude.split("\n", 1)[1].strip()
        _send_report_email(subject_prefix, "Claude", report_claude)
        logger.info("Reporte Anthropic (Claude) enviado.")
    except Exception:
        logger.exception("Error generando o enviando el reporte Anthropic.")

    logger.info("== Proceso finalizado ==")

if __name__ == "__main__":
    main()
