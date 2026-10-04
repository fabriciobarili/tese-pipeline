"""Testes do estágio de processamento (join meteo x voos)."""

from __future__ import annotations

import pandas as pd

from tese_pipeline.processing.features import build


def _meteo() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00"], utc=True),
            "temperature_2m": [20.0, 21.0],
            "precipitation": [0.0, 1.5],
            "wind_speed_10m": [5.0, 6.0],
            "location": ["SBGR", "SBGR"],
        }
    )


def _flights() -> pd.DataFrame:
    base = pd.Timestamp("2024-01-01 10:30", tz="UTC").timestamp()
    return pd.DataFrame(
        {
            "icao24": ["abc123"],
            "firstSeen": [int(base)],
            "_airport": ["SBGR"],
            "_kind": ["arrival"],
        }
    )


def test_build_nao_explode_linhas() -> None:
    df = build(_meteo(), _flights())
    assert len(df) == 1  # join m:1 não duplica voos


def test_is_rain_derivada() -> None:
    df = build(_meteo(), _flights())
    # voo às 10:30 -> hora 10 -> precipitação 0.0 -> is_rain 0
    assert df["is_rain"].iloc[0] == 0
    assert df["hour"].iloc[0] == 10
