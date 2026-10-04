"""Configuração central de logging para toda a pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> None:
    """Configura o logging raiz com formato padronizado e, opcionalmente, arquivo.

    Logs de execução longa (ingestão, Optuna) devem apontar para ``reports/logs/``
    como evidência de auditoria.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=_FORMAT, handlers=handlers, force=True)
