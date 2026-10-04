"""Ingestão meteorológica para a Região Metropolitana de Porto Alegre.

Reutiliza a lógica de meteo_openmeteo.py com config dedicada (config/meteo_rm.yaml)
e saída em data/raw/meteo_rm.parquet. Inclui shortwave_radiation para cálculo de Tmrt.
"""

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
    """Baixa a série horária de uma localidade com retry e backoff."""
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
            r = requests.get(HIST_URL, params=params, timeout=90)
            r.raise_for_status()
            hourly = r.json()["hourly"]
            df = pd.DataFrame(hourly)
            df["location"] = loc["name"]
            df["latitude"]  = loc["latitude"]
            df["longitude"] = loc["longitude"]
            df["time"] = pd.to_datetime(df["time"])
            return df
        except requests.RequestException as exc:
            LOG.warning("falha %s (tentativa %d/%d): %s", loc["name"], attempt, retries, exc)
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("inalcançável")


def main() -> None:
    configure_logging(log_file="reports/logs/ingest_meteo_rm.log")
    params = load_config("params.yaml")
    set_global_seed(params["seed"])
    cfg = load_config("config/meteo_rm.yaml")

    frames = []
    for loc in cfg["locations"]:
        df = fetch_location(loc, cfg)
        frames.append(df)
        for col in cfg["hourly"]:
            if col in df.columns:
                pct = df[col].isna().mean() * 100
                if pct > 0:
                    LOG.info("missing %s/%s: %.2f%%", loc["name"], col, pct)
        LOG.info("%s: %d horas baixadas", loc["name"], len(df))
        time.sleep(0.5)   # respeita rate limit Open-Meteo

    result = pd.concat(frames, ignore_index=True)

    out = Path("data/raw/meteo_rm.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)

    write_provenance(
        out,
        source="open-meteo:archive-api:ERA5",
        params={
            "meteo_rm": {
                "n_locations": len(cfg["locations"]),
                "locations": [l["name"] for l in cfg["locations"]],
                "variables": cfg["hourly"],
                "start_date": cfg["start_date"],
                "end_date": cfg["end_date"],
                "timezone": cfg["timezone"],
            }
        },
    )
    LOG.info(
        "meteo_rm salvo: %d linhas | %d localidades | período: %s → %s",
        len(result), len(cfg["locations"]), cfg["start_date"], cfg["end_date"],
    )


if __name__ == "__main__":
    main()
