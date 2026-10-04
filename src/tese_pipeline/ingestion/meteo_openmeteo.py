"""Estágio 1 — Ingestão meteorológica (Open-Meteo Historical Weather API)."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from tese_pipeline.config import load_config, set_global_seed
from tese_pipeline.logging_conf import configure_logging
from tese_pipeline.provenance import write_provenance

LOG = logging.getLogger(__name__)
HIST_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_location(loc: dict, cfg: dict, retries: int = 3) -> pd.DataFrame:
    """Baixa a série horária de uma localidade, com retry e backoff."""
    params = {
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "start_date": cfg["start_date"],
        "end_date": cfg["end_date"],
        "hourly": ",".join(cfg["hourly"]),
        "timezone": cfg["timezone"],
    }
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(HIST_URL, params=params, timeout=60)
            r.raise_for_status()
            hourly = r.json()["hourly"]
            df = pd.DataFrame(hourly)
            df["location"] = loc["name"]
            df["time"] = pd.to_datetime(df["time"])
            return df
        except requests.RequestException as exc:
            LOG.warning("falha %s (tentativa %d/%d): %s", loc["name"], attempt, retries, exc)
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("inalcançável")


def main() -> None:
    configure_logging(log_file="reports/logs/ingest_meteo.log")
    params = load_config("params.yaml")
    set_global_seed(params["seed"])
    cfg = load_config("config/meteo.yaml")

    frames = [fetch_location(loc, cfg) for loc in cfg["locations"]]
    df = pd.concat(frames, ignore_index=True)

    for col in cfg["hourly"]:
        if col in df:
            pct = df[col].isna().mean() * 100
            LOG.info("missing %s: %.2f%%", col, pct)

    out = Path("data/raw/meteo.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    write_provenance(out, source="open-meteo:archive-api", params={"meteo": cfg})
    LOG.info("meteo salvo: %d linhas, %d locais", len(df), len(cfg["locations"]))


if __name__ == "__main__":
    main()
