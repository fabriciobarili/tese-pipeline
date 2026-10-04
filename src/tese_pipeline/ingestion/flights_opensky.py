"""Estágio 2 — Ingestão de voos (OpenSky Network, OAuth2 client credentials)."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from tese_pipeline.config import load_config, set_global_seed
from tese_pipeline.logging_conf import configure_logging
from tese_pipeline.provenance import write_provenance

LOG = logging.getLogger(__name__)
TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network/"
    "protocol/openid-connect/token"
)
BASE = "https://opensky-network.org/api"


def get_token() -> str:
    """Obtém um token OAuth2 via client credentials (variáveis de ambiente)."""
    try:
        client_id = os.environ["OPENSKY_CLIENT_ID"]
        client_secret = os.environ["OPENSKY_CLIENT_SECRET"]
    except KeyError as exc:
        raise RuntimeError(
            "Defina OPENSKY_CLIENT_ID e OPENSKY_CLIENT_SECRET no ambiente."
        ) from exc
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def to_unix(iso_date: str) -> int:
    return int(datetime.fromisoformat(iso_date).replace(tzinfo=timezone.utc).timestamp())


def fetch_window(kind: str, airport: str, begin: int, end: int, token: str) -> list[dict]:
    """Busca voos de chegada/partida numa janela; 404 = janela sem voos."""
    r = requests.get(
        f"{BASE}/flights/{kind}",
        params={"airport": airport, "begin": begin, "end": end},
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    if r.status_code == 404:
        return []
    r.raise_for_status()
    return r.json()


def main() -> None:
    configure_logging(log_file="reports/logs/ingest_flights.log")
    params = load_config("params.yaml")
    set_global_seed(params["seed"])
    cfg = load_config("config/flights.yaml")
    token = get_token()

    rows: list[dict] = []
    step = timedelta(days=cfg["chunk_days"])
    for airport in cfg["airports"]:
        for kind in cfg["kinds"]:
            cur = datetime.fromisoformat(cfg["start_date"])
            end_dt = datetime.fromisoformat(cfg["end_date"])
            while cur < end_dt:
                nxt = min(cur + step, end_dt)
                data = fetch_window(
                    kind, airport, to_unix(cur.isoformat()), to_unix(nxt.isoformat()), token
                )
                for d in data:
                    d["_airport"], d["_kind"] = airport, kind
                rows.extend(data)
                LOG.info("%s %s %s->%s: %d voos", airport, kind, cur.date(), nxt.date(), len(data))
                cur = nxt
                time.sleep(1)  # respeita rate limit

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["icao24", "firstSeen"])

    out = Path("data/raw/flights.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    write_provenance(out, source="opensky:flights", params={"flights": cfg})
    LOG.info("voos salvos: %d registros", len(df))


if __name__ == "__main__":
    main()
