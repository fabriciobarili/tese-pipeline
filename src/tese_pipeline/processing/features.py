"""Estágio 3 — Processamento e feature engineering (join meteo x voos + H3)."""

from __future__ import annotations

import logging
from pathlib import Path

import h3
import pandas as pd

from tese_pipeline.config import load_config, set_global_seed
from tese_pipeline.logging_conf import configure_logging
from tese_pipeline.provenance import write_provenance

LOG = logging.getLogger(__name__)


def add_h3_features(df: pd.DataFrame, icao_coords: dict, res: int, macro_res: int) -> pd.DataFrame:
    """Adiciona células H3 do aeroporto e features derivadas.

    ``icao_coords``: dict mapeando ICAO → {latitude, longitude}
    Resolução padrão vem de ``params.yaml: geospatial.h3_resolution``.
    """
    def cell(airport: str) -> str | None:
        coords = icao_coords.get(airport)
        if coords is None:
            return None
        return h3.latlng_to_cell(coords["latitude"], coords["longitude"], res)

    df = df.copy()
    df["h3_airport"] = df["_airport"].apply(cell)
    df["h3_airport_r7"] = df["h3_airport"].apply(
        lambda c: h3.cell_to_parent(c, macro_res) if c else None
    )
    return df


def build(meteo: pd.DataFrame, flights: pd.DataFrame, icao_coords: dict,
          h3_res: int, h3_macro: int) -> pd.DataFrame:
    """Une meteorologia e voos por (aeroporto, hora local) e deriva features.

    Regra anti-leakage: usar apenas informação disponível no instante do evento.

    Schema esperado de flights (ANAC VRA normalizado):
      actual_arr      — datetime UTC da chegada real (tz-naive, UTC implícito)
      arr_hour_local  — hora local arredondada para baixo (gerado em flights_anac.py)
      dest_icao       — ICAO do aeródromo de destino (filtrado para SBPA)
      delay_arr_min   — atraso de chegada em minutos
      is_delayed      — bool: delay_arr_min > 15
      route_type      — N/I/C
      airline_icao    — companhia aérea
    """
    meteo = meteo.copy()
    flights = flights.copy()

    # meteorologia: índice por (aeroporto, hora local)
    meteo["ts_hour"] = pd.to_datetime(meteo["time"]).dt.tz_localize("UTC").dt.tz_convert("America/Sao_Paulo").dt.floor("h")

    # voos ANAC: arr_hour_local já foi gerado pelo módulo de ingestão
    if "arr_hour_local" in flights.columns:
        flights["ts_hour"] = pd.to_datetime(flights["arr_hour_local"])
    else:
        flights["ts_hour"] = pd.to_datetime(flights["actual_arr"], utc=True).dt.tz_convert("America/Sao_Paulo").dt.floor("h")

    # para join, airport = dest_icao
    if "dest_icao" in flights.columns:
        flights["_airport"] = flights["dest_icao"].str.upper()
    elif "_airport" not in flights.columns:
        flights["_airport"] = "SBPA"

    df = flights.merge(
        meteo,
        left_on=["_airport", "ts_hour"],
        right_on=["location", "ts_hour"],
        how="left",
        validate="m:1",
    )

    sem_match = df["location"].isna().mean() * 100
    LOG.info("voos sem match meteorológico: %.2f%%", sem_match)

    # features temporais
    df["hour"] = df["ts_hour"].dt.hour
    df["dow"] = df["ts_hour"].dt.dayofweek
    df["month"] = df["ts_hour"].dt.month
    df["is_rain"] = (df["precipitation"].fillna(0) > 0).astype(int)

    # features de atraso (ANAC VRA)
    for col in ("delay_arr_min", "delay_dep_min", "is_delayed", "route_type", "airline_icao"):
        if col not in df.columns:
            df[col] = None

    # features geoespaciais H3
    df = add_h3_features(df, icao_coords, h3_res, h3_macro)

    return df


def main() -> None:
    configure_logging(log_file="reports/logs/process.log")
    params = load_config("params.yaml")
    set_global_seed(params["seed"])
    flights_cfg = load_config("config/flights.yaml")

    meteo = pd.read_parquet("data/raw/meteo.parquet")
    flights = pd.read_parquet("data/raw/flights_anac.parquet")
    df = build(
        meteo, flights,
        icao_coords=flights_cfg["icao_coords"],
        h3_res=params["geospatial"]["h3_resolution"],
        h3_macro=params["geospatial"]["h3_macro_resolution"],
    )

    out = Path("data/processed/features.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    write_provenance(out, source="processing:features", params={
        "seed": params["seed"],
        "h3_resolution": params["geospatial"]["h3_resolution"],
    })
    LOG.info("features: %d linhas, %d colunas", df.shape[0], df.shape[1])


if __name__ == "__main__":
    main()
