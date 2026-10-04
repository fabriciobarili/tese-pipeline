"""Estágio 4 — Geração de corridas sintéticas.

As premissas do processo gerador vivem em ``config/synthesis.yaml`` e são
justificadas em ``docs/decisions/0002-premissas-de-sintese.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tese_pipeline.config import load_config, set_global_seed
from tese_pipeline.logging_conf import configure_logging
from tese_pipeline.provenance import write_provenance

LOG = logging.getLogger(__name__)


def generate(features: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Gera corridas sintéticas a partir de eventos de voo + condição meteorológica."""
    peak = set(cfg.get("peak_hours", [7, 8, 17, 18]))
    rows: list[dict] = []
    for ev in features.itertuples(index=False):
        n = rng.poisson(cfg["demand"]["rides_per_arrival_mean"])
        for _ in range(n):
            dur = rng.normal(cfg["duration_minutes"]["base_mean"], cfg["duration_minutes"]["base_sd"])
            if getattr(ev, "is_rain", 0) == 1:
                dur *= cfg["duration_minutes"]["rain_multiplier"]
            if getattr(ev, "hour", -1) in peak:
                dur *= cfg["duration_minutes"]["peak_hour_multiplier"]
            dur += rng.normal(0, cfg["noise_sd"])
            dist = max(0.5, rng.normal(cfg["distance_km"]["mean"], cfg["distance_km"]["sd"]))
            rows.append(
                {
                    "origin_flight": getattr(ev, "icao24", None),
                    "airport": getattr(ev, "_airport", None),
                    "ts_hour": getattr(ev, "ts_hour", None),
                    "hour": getattr(ev, "hour", None),
                    "dow": getattr(ev, "dow", None),
                    "month": getattr(ev, "month", None),
                    "is_rain": getattr(ev, "is_rain", 0),
                    "temperature_2m": getattr(ev, "temperature_2m", None),
                    "wind_speed_10m": getattr(ev, "wind_speed_10m", None),
                    "distance_km": round(dist, 2),
                    "duration_min": round(max(1.0, dur), 2),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    configure_logging(log_file="reports/logs/synthesize.log")
    params = load_config("params.yaml")
    set_global_seed(params["seed"])
    rng = np.random.default_rng(params["seed"])
    cfg = load_config("config/synthesis.yaml")

    features = pd.read_parquet("data/processed/features.parquet")
    df = generate(features, cfg, rng)
    if len(df) > cfg["n_rides"]:
        df = df.sample(cfg["n_rides"], random_state=params["seed"]).reset_index(drop=True)

    out = Path("data/synthetic/rides.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    write_provenance(out, source="synthesis:rides", params={"synthesis": cfg})
    LOG.info("corridas sintéticas: %d", len(df))


if __name__ == "__main__":
    main()
