"""Estágio 1b — Ingestão de voos ANAC VRA (Voo Regular Ativo).

Fonte: ANAC Dados Abertos — Voo Regular Ativo (VRA)
URL:   https://www.gov.br/anac/pt-br/acesso-a-informacao/dados-abertos/
         areas-de-atuacao/voos-e-operacoes-aereas/voo-regular-ativo-vra
Acesso: 2026-10-10

Formato original:
  - CSV com separador ';', encoding UTF-8-BOM
  - Primeira linha: metadado ("Atualizado em: YYYY-MM-DD")
  - Segunda linha: cabeçalho
  - Dados a partir da terceira linha

Schema original → normalizado:
  ICAO Empresa Aérea       → airline_icao
  Número Voo               → flight_number
  Código Autorização (DI)  → auth_code
  Código Tipo Linha        → route_type  (N=nacional, I=internacional, C=cargueiro)
  ICAO Aeródromo Origem    → origin_icao
  ICAO Aeródromo Destino   → dest_icao
  Partida Prevista         → scheduled_dep
  Partida Real             → actual_dep
  Chegada Prevista         → scheduled_arr
  Chegada Real             → actual_arr
  Situação Voo             → flight_status (REALIZADO | CANCELADO | ...)
  Código Justificativa     → delay_code

Colunas derivadas:
  delay_arr_min    — atraso de chegada em minutos (actual_arr − scheduled_arr)
  delay_dep_min    — atraso de partida em minutos (actual_dep − scheduled_dep)
  is_delayed       — booleano: delay_arr_min > 15 min (limiar IATA)
  arr_hour_local   — hora local (America/Sao_Paulo) da chegada real
  source_file      — nome do arquivo CSV de origem (rastreabilidade)
  source_month     — YYYY-MM extraído do nome do arquivo
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd
import requests

from tese_pipeline.config import load_config, set_global_seed
from tese_pipeline.logging_conf import configure_logging
from tese_pipeline.provenance import write_provenance

LOG = logging.getLogger(__name__)

# Mapeamento de nomes originais → snake_case normalizado
COL_MAP = {
    "ICAO Empresa Aérea":      "airline_icao",
    "Número Voo":               "flight_number",
    "Código Autorização (DI)":  "auth_code",
    "Código Tipo Linha":        "route_type",
    "ICAO Aeródromo Origem":    "origin_icao",
    "ICAO Aeródromo Destino":   "dest_icao",
    "Partida Prevista":         "scheduled_dep",
    "Partida Real":             "actual_dep",
    "Chegada Prevista":         "scheduled_arr",
    "Chegada Real":             "actual_arr",
    "Situação Voo":             "flight_status",
    "Código Justificativa":     "delay_code",
}

DATETIME_COLS = ["scheduled_dep", "actual_dep", "scheduled_arr", "actual_arr"]
DELAY_THRESHOLD_MIN = 15   # limiar IATA para voo "atrasado"
LOCAL_TZ = "America/Sao_Paulo"


def download_csv(url: str, dest_dir: Path, retries: int = 3) -> Path:
    """Baixa um CSV de URL para dest_dir. Retorna o caminho local."""
    filename = Path(unquote(urlparse(url).path)).name
    dest = dest_dir / filename
    if dest.exists():
        LOG.info("já existe localmente: %s (pulando download)", filename)
        return dest
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=120, stream=True)
            r.raise_for_status()
            dest.write_bytes(r.content)
            size_kb = dest.stat().st_size // 1024
            LOG.info("baixado: %s (%d KB)", filename, size_kb)
            return dest
        except requests.RequestException as exc:
            LOG.warning("tentativa %d/%d falhou para %s: %s", attempt, retries, filename, exc)
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("inalcançável")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def read_vra_csv(path: Path) -> pd.DataFrame:
    """Lê um CSV VRA com encoding UTF-8-BOM, ';' como separador e skiprows=1."""
    df = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        skiprows=1,           # pula linha de metadado ("Atualizado em: ...")
        on_bad_lines="skip",
        dtype=str,            # lê tudo como string; normaliza depois
    )
    # strip em nomes de colunas e valores
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()
    return df


def normalize(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """Renomeia colunas, converte tipos e deriva features."""
    # renomeia somente colunas presentes
    rename = {k: v for k, v in COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    # converte datetimes (formato: YYYY-MM-DD HH:MM:SS)
    for col in DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%Y-%m-%d %H:%M:%S", errors="coerce")

    # deriva atrasos
    if "actual_arr" in df.columns and "scheduled_arr" in df.columns:
        df["delay_arr_min"] = (
            df["actual_arr"] - df["scheduled_arr"]
        ).dt.total_seconds().div(60).round(1)
    else:
        df["delay_arr_min"] = float("nan")

    if "actual_dep" in df.columns and "scheduled_dep" in df.columns:
        df["delay_dep_min"] = (
            df["actual_dep"] - df["scheduled_dep"]
        ).dt.total_seconds().div(60).round(1)
    else:
        df["delay_dep_min"] = float("nan")

    df["is_delayed"] = df["delay_arr_min"] > DELAY_THRESHOLD_MIN

    # hora local de chegada (para join com meteo e features temporais)
    if "actual_arr" in df.columns:
        df["arr_hour_local"] = (
            df["actual_arr"]
            .dt.tz_localize("UTC")
            .dt.tz_convert(LOCAL_TZ)
            .dt.floor("h")
        )

    # rastreabilidade
    df["source_file"] = source_file
    # extrai YYYY-MM do nome do arquivo (ex: VRA_20251.csv → 2025-01)
    stem = Path(source_file).stem  # "VRA_20251"
    digits = "".join(filter(str.isdigit, stem))  # "20251"
    if len(digits) >= 5:
        year, month = digits[:4], digits[4:]
        df["source_month"] = f"{year}-{month.zfill(2)}"
    else:
        df["source_month"] = ""

    return df


def main() -> None:
    configure_logging(log_file="reports/logs/ingest_anac.log")
    params = load_config("params.yaml")
    set_global_seed(params["seed"])
    cfg = load_config("config/flights.yaml")

    anac_cfg = cfg.get("anac", {})
    urls: list[str] = anac_cfg.get("urls", [])
    airport: str = anac_cfg.get("airport_filter", "SBPA")
    keep_status: list[str] = anac_cfg.get("keep_status", ["REALIZADO"])

    if not urls:
        raise ValueError("config/flights.yaml: 'anac.urls' está vazio.")

    # diretório de cache local para os CSVs brutos
    raw_dir = Path("data/raw/anac_vra")
    raw_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    file_hashes: dict[str, str] = {}

    for url in urls:
        csv_path = download_csv(url, raw_dir)
        file_hashes[csv_path.name] = sha256_file(csv_path)

        df_raw = read_vra_csv(csv_path)
        df = normalize(df_raw, source_file=csv_path.name)

        # filtra aeroporto e status
        if "dest_icao" in df.columns:
            df = df[df["dest_icao"].str.upper() == airport.upper()]
        if "flight_status" in df.columns:
            df = df[df["flight_status"].isin(keep_status)]

        n = len(df)
        LOG.info("%s → %d chegadas %s (%s)", csv_path.name, n, airport, keep_status)
        frames.append(df)
        time.sleep(0.5)   # respeita servidor ANAC

    if not frames:
        raise RuntimeError(f"Nenhum voo {airport} encontrado nos CSVs.")

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values("actual_arr", na_position="last").reset_index(drop=True)

    # estatísticas de missing
    for col in DATETIME_COLS + ["delay_arr_min"]:
        if col in result.columns:
            pct = result[col].isna().mean() * 100
            LOG.info("missing %s: %.2f%%", col, pct)

    out = Path("data/raw/flights_anac.parquet")
    result.to_parquet(out, index=False)

    write_provenance(
        out,
        source="anac:vra",
        params={
            "anac": {
                "airport_filter": airport,
                "keep_status": keep_status,
                "n_files": len(urls),
                "file_hashes": file_hashes,
                "source_url": "https://www.gov.br/anac/pt-br/acesso-a-informacao/dados-abertos/areas-de-atuacao/voos-e-operacoes-aereas/voo-regular-ativo-vra",
                "accessed": "2026-10-10",
            }
        },
    )
    LOG.info(
        "ANAC VRA salvo: %d chegadas | %d meses | colunas: %s",
        len(result), len(frames), result.columns.tolist(),
    )


if __name__ == "__main__":
    main()
