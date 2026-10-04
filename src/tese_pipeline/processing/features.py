"""Estágio 3 — Processamento e feature engineering (join meteo x voos)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from tese_pipeline.config import load_config, set_global_seed
from tese_pipeline.logging_conf import configure_logging
from tese_pipeline.provenance import write_provenance

LOG = logging.getLogger(__name__)


def build(meteo: pd.DataFrame, flights: pd.DataFrame) -> pd.DataFrame:
    """Une meteorologia e voos por (aeroporto, hora UTC) e deriva features.

    Regra anti-leakage: usar apenas informação disponível no instante do evento.
    """
    meteo = meteo.copy()
    flights = flights.copy()

    meteo["ts_hour"] = pd.to_datetime(meteo["time"], utc=True).dt.floor("h")
    flights["ts_hour"] = pd.to_datetime(flights["firstSeen"], unit="s", utc=True).dt.floor("h")

    df = flights.merge(
        meteo,
        left_on=["_airport", "ts_hour"],
        right_on=["location", "ts_hour"],
        how="left",
        validate="m:1",  # flagra explosão de linhas no join
    )

    sem_match = df["location"].isna().mean() * 100
    LOG.info("voos sem match meteorológico: %.2f%%", sem_match)

    df["hour"] = df["ts_hour"].dt.hour
    df["dow"] = df["ts_hour"].dt.dayofweek
    df["month"] = df["ts_hour"].dt.month
    df["is_rain"] = (df["precipitation"].fillna(0) > 0).astype(int)
    return df


def main() -> None:
    configure_logging(log_file="reports/logs/process.log")
    params = load_config("params.yaml")
    set_global_seed(params["seed"])

    meteo = pd.read_parquet("data/raw/meteo.parquet")
    flights = pd.read_parquet("data/raw/flights.parquet")
    df = build(meteo, flights)

    out = Path("data/processed/features.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    write_provenance(out, source="processing:features", params={"seed": params["seed"]})
    LOG.info("features: %d linhas, %d colunas", df.shape[0], df.shape[1])


if __name__ == "__main__":
    main()
