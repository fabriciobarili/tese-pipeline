"""Estágio 4 — Geração de corridas sintéticas com indexação H3.

As premissas do processo gerador vivem em ``config/synthesis.yaml`` e são
justificadas em ``docs/decisions/0002-premissas-de-sintese.md``.
Destinos são sorteados por saltos H3 (``max_destination_k``) a partir do
aeroporto de origem, garantindo realismo geoespacial parametrizado.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h3
import numpy as np
import pandas as pd

from tese_pipeline.config import load_config, set_global_seed
from tese_pipeline.logging_conf import configure_logging
from tese_pipeline.provenance import write_provenance

LOG = logging.getLogger(__name__)


def sample_destination(origin_cell: str, rng: np.random.Generator, max_k: int) -> str:
    """Sorteia uma célula de destino dentro de ``max_k`` saltos H3 da origem."""
    candidates = list(h3.grid_disk(origin_cell, max_k))
    return rng.choice(candidates)


def generate(features: pd.DataFrame, cfg: dict, rng: np.random.Generator,
             max_k: int) -> pd.DataFrame:
    """Gera corridas sintéticas a partir de eventos de voo + condição meteorológica.

    Cada corrida inclui célula H3 de origem (aeroporto), célula H3 de destino
    (sorteada no raio ``max_k``), distância em saltos H3 e distância em km.
    """
    peak = set(cfg.get("peak_hours", [7, 8, 17, 18]))
    rows: list[dict] = []
    for ev in features.itertuples(index=False):
        n = rng.poisson(cfg["demand"]["rides_per_arrival_mean"])
        h3_origin: str | None = getattr(ev, "h3_airport", None)
        for _ in range(n):
            dur = rng.normal(cfg["duration_minutes"]["base_mean"],
                             cfg["duration_minutes"]["base_sd"])
            if getattr(ev, "is_rain", 0) == 1:
                dur *= cfg["duration_minutes"]["rain_multiplier"]
            if getattr(ev, "hour", -1) in peak:
                dur *= cfg["duration_minutes"]["peak_hour_multiplier"]
            dur += rng.normal(0, cfg["noise_sd"])

            # destino geoespacial via H3
            h3_dest: str | None = None
            dist_cells: int | None = None
            dist_km: float | None = None
            if h3_origin:
                h3_dest = sample_destination(h3_origin, rng, max_k)
                dist_cells = h3.grid_distance(h3_origin, h3_dest)
                lat_o, lng_o = h3.cell_to_latlng(h3_origin)
                lat_d, lng_d = h3.cell_to_latlng(h3_dest)
                dist_km = round(h3.great_circle_distance(
                    (lat_o, lng_o), (lat_d, lng_d), unit="km"
                ), 3)
            else:
                dist_km = max(0.5, rng.normal(cfg["distance_km"]["mean"],
                                              cfg["distance_km"]["sd"]))

            dist_km = dist_km or max(0.5, rng.normal(cfg["distance_km"]["mean"],
                                                      cfg["distance_km"]["sd"]))
            rows.append({
                "origin_flight": getattr(ev, "icao24", None),
                "airport": getattr(ev, "_airport", None),
                "ts_hour": getattr(ev, "ts_hour", None),
                "hour": getattr(ev, "hour", None),
                "dow": getattr(ev, "dow", None),
                "month": getattr(ev, "month", None),
                "is_rain": getattr(ev, "is_rain", 0),
                "temperature_2m": getattr(ev, "temperature_2m", None),
                "wind_speed_10m": getattr(ev, "wind_speed_10m", None),
                "h3_origin": h3_origin,
                "h3_destination": h3_dest,
                "h3_origin_r7": h3.cell_to_parent(h3_origin, 7) if h3_origin else None,
                "dist_cells": dist_cells,      # saltos H3 (feature sem leakage de rota)
                "distance_km": round(dist_km, 2),
                "duration_min": round(max(1.0, dur), 2),
            })
    return pd.DataFrame(rows)


def main() -> None:
    configure_logging(log_file="reports/logs/synthesize.log")
    params = load_config("params.yaml")
    set_global_seed(params["seed"])
    rng = np.random.default_rng(params["seed"])
    cfg = load_config("config/synthesis.yaml")
    geo = params["geospatial"]

    features = pd.read_parquet("data/processed/features.parquet")
    df = generate(features, cfg, rng, max_k=geo["max_destination_k"])
    if len(df) > cfg["n_rides"]:
        df = df.sample(cfg["n_rides"], random_state=params["seed"]).reset_index(drop=True)

    out = Path("data/synthetic/rides.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    write_provenance(out, source="synthesis:rides", params={
        "synthesis": cfg,
        "geospatial": geo,
    })
    LOG.info("corridas sintéticas: %d | colunas: %s", len(df), list(df.columns))


if __name__ == "__main__":
    main()
