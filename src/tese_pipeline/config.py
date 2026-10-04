"""Carregamento de configuração e controle de determinismo.

Toda a configuração científica vive em ``params.yaml`` e ``config/*.yaml``.
Este módulo centraliza a leitura e a fixação de seeds para garantir
reprodutibilidade em todos os estágios da pipeline.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Carrega um arquivo YAML de configuração e retorna um dicionário."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Configuração não encontrada: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Configuração inválida (esperado mapa YAML): {p}")
    return data


def set_global_seed(seed: int) -> None:
    """Fixa as seeds globais para reprodutibilidade bit-a-bit do que for possível.

    Propaga a seed para ``random``, ``numpy`` e ``PYTHONHASHSEED``. Os módulos de
    LightGBM e Optuna recebem a seed explicitamente em seus próprios parâmetros.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
