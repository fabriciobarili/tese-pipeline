"""Funções de limpeza reutilizadas no estágio de processamento."""

from __future__ import annotations

import pandas as pd


def drop_implausible_durations(df: pd.DataFrame, col: str = "duration_min") -> pd.DataFrame:
    """Remove linhas com duração não-positiva (defensivo; documentar no data card)."""
    if col in df:
        return df[df[col] > 0].copy()
    return df
