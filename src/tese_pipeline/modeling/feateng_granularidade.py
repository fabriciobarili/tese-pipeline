"""Utilitários de engenharia de features (porta do notebook ``05_6_FeatureEng_...``).

Funções **puras** aplicadas sobre os dados que a pipeline já produz (ex.: ``rides.parquet``),
consumidas por ``modeling/dataset.make_xy`` e, por tabela, por ``train.py``/``optimize.py``.
Não fazem I/O nem trocam a base de dados — recebem e devolvem ``DataFrame``.

Lógica trazida do ``05_6_FeatureEng_Individual_vs_Agrupado.ipynb`` (parte de features):
  - ``enrich_climate_temporal`` — sobre a série HORÁRIA do clima: médias móveis
    multi-horizonte por TEMPO (3h/6h/24h, robustas a lacunas), delta de 3h, razão física
    vento/temperatura, e ``lag``/``diff`` de 1 janela (defasagem causal — só passado).
  - ``add_cyclical`` — codificação cíclica sin/cos de variáveis temporais (hora/dia/mês),
    evitando o salto artificial 23->0 que o valor cru introduz nas árvores.
  - ``enrich_rides`` — conveniência: deriva a série horária das corridas, enriquece e mescla
    as colunas derivadas de volta a cada corrida, somando as cíclicas.

Todas as derivações são backward-looking (rolling por tempo trailing; ``shift`` para trás),
então não há vazamento do futuro para dentro das features de uma linha.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

DEFAULT_HORIZONTES: tuple[str, ...] = ("3h", "6h", "24h")
DEFAULT_CLIMATE_COLS: tuple[str, ...] = (
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "precipitation", "surface_pressure",
)
DEFAULT_CICLICAS: dict[str, int] = {"hour": 24, "dow": 7, "month": 12}


def add_cyclical(df: pd.DataFrame, periods: Mapping[str, int]) -> pd.DataFrame:
    """Adiciona ``{col}_sin``/``{col}_cos`` para cada coluna temporal presente em ``periods``."""
    df = df.copy()
    for col, period in periods.items():
        if col in df.columns:
            ang = 2 * np.pi * df[col].astype(float) / period
            df[f"{col}_sin"] = np.sin(ang)
            df[f"{col}_cos"] = np.cos(ang)
    return df


def enrich_climate_temporal(serie_h: pd.DataFrame, climate_cols: Sequence[str],
                            horizontes: Sequence[str] = DEFAULT_HORIZONTES) -> pd.DataFrame:
    """Enriquece uma série HORÁRIA de clima (indexada por tempo) com features derivadas.

    Médias móveis por TEMPO (lidam com lacunas); ``delta_3h`` e ``lag``/``diff`` assumem passo
    horário — por isso a série é reindexada para frequência horária completa antes dos shifts.
    """
    s = serie_h.sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    full.name = s.index.name
    s = s.reindex(full)
    for v in [c for c in climate_cols if c in s.columns]:
        for h in horizontes:
            s[f"{v}_media_{h}"] = s[v].rolling(h).mean()
        s[f"{v}_delta_3h"] = s[v] - s[v].shift(3)
        s[f"{v}_lag"] = s[v].shift(1)
        s[f"diff_{v}"] = s[v] - s[v].shift(1)
    if "wind_speed_10m" in s.columns and "temperature_2m" in s.columns:
        s["razao_vento_temp"] = s["wind_speed_10m"] / (s["temperature_2m"].abs() + 1e-3)
    return s


def enrich_rides(df: pd.DataFrame, time_col: str = "ts_hour",
                 climate_cols: Sequence[str] = DEFAULT_CLIMATE_COLS,
                 horizontes: Sequence[str] = DEFAULT_HORIZONTES,
                 ciclicas: Mapping[str, int] = DEFAULT_CICLICAS) -> pd.DataFrame:
    """Aplica o enriquecimento do 05_6 a um ``DataFrame`` de corridas, mantendo a base.

    Deriva a série horária única do clima (``time_col`` + ``climate_cols`` presentes), calcula
    as features temporais e mescla de volta por ``time_col``; depois soma as cíclicas. Colunas
    climáticas ausentes são ignoradas (ex.: ``surface_pressure`` não existe em ``rides.parquet``).
    """
    if time_col not in df.columns:
        LOG.warning("enrich_rides: coluna de tempo '%s' ausente; sem enriquecimento.", time_col)
        return add_cyclical(df, ciclicas)
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    presentes = [c for c in climate_cols if c in df.columns]
    if presentes:
        serie = (df[[time_col, *presentes]].drop_duplicates(time_col)
                 .set_index(time_col).sort_index())
        enriched = enrich_climate_temporal(serie, presentes, horizontes)
        derivadas = [c for c in enriched.columns if c not in presentes]
        df = df.merge(enriched[derivadas].reset_index(), on=time_col, how="left")
        LOG.info("enrich_rides: %d features climáticas derivadas de %s", len(derivadas),
                 presentes)
    else:
        LOG.warning("enrich_rides: nenhuma coluna climática de %s presente.", list(climate_cols))
    return add_cyclical(df, ciclicas)
