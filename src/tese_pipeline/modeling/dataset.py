"""Preparação de dados para modelagem, com split temporal (sem leakage)."""

from __future__ import annotations

import pandas as pd

DROP_COLS = ["ts_hour", "origin_flight", "time", "location", "_kind", "callsign"]


def make_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X e y, descarta colunas não-preditivas e aplica one-hot."""
    df = df.sort_values("ts_hour") if "ts_hour" in df else df
    y = df[target]
    x = df.drop(columns=[target, *[c for c in DROP_COLS if c in df]], errors="ignore")
    x = pd.get_dummies(x, drop_first=True)
    return x, y


def temporal_split(
    df: pd.DataFrame, time_col: str = "ts_hour", test_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporal: treino no passado, teste no futuro. Nunca aleatório."""
    df = df.sort_values(time_col)
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
