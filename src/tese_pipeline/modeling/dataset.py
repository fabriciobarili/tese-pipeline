"""Preparação de dados para modelagem, com split temporal (sem leakage)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from tese_pipeline.modeling import feateng_granularidade as fe

# Colunas não-preditivas: identificadores, timestamps e chaves de rastreabilidade.
# h3_origin / h3_destination / h3_origin_r7 são features válidas (one-hot ou label).
DROP_COLS = ["ts_hour", "origin_flight", "time", "location", "_kind", "callsign"]


def make_xy(
    df: pd.DataFrame, target: str, fe_cfg: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X e y, enriquece as features (05_6), descarta não-preditivas e aplica one-hot.

    ``fe_cfg`` (bloco ``feature_eng`` do params.yaml) controla o enriquecimento climático e
    cíclico; ``None`` usa os padrões. ``enabled: false`` desliga o enriquecimento.
    """
    df = df.sort_values("ts_hour") if "ts_hour" in df else df
    cfg = fe_cfg or {}
    if cfg.get("enabled", True):
        kwargs = {k: cfg[k] for k in ("time_col", "climate_cols", "horizontes", "ciclicas")
                  if k in cfg}
        df = fe.enrich_rides(df, **kwargs)
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
