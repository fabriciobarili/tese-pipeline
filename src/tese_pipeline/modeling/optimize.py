"""Estágio 6a — Otimização de hiperparâmetros do LightGBM com Optuna.

Validação cruzada temporal (TimeSeriesSplit) para evitar vazamento. Os trials
são persistidos em SQLite (``reports/metrics/optuna.db``) para auditoria e retomada.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from tese_pipeline.config import load_config, set_global_seed
from tese_pipeline.logging_conf import configure_logging
from tese_pipeline.modeling.dataset import make_xy

LOG = logging.getLogger(__name__)


def objective(trial: optuna.Trial, x: pd.DataFrame, y: pd.Series, folds: int, seed: int) -> float:
    params = {
        "objective": "regression",
        "metric": "rmse",
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    tscv = TimeSeriesSplit(n_splits=folds)
    scores: list[float] = []
    for tr, va in tscv.split(x):
        dtr = lgb.Dataset(x.iloc[tr], y.iloc[tr])
        dva = lgb.Dataset(x.iloc[va], y.iloc[va])
        model = lgb.train(
            params, dtr, valid_sets=[dva], num_boost_round=2000,
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred = model.predict(x.iloc[va])
        scores.append(float(mean_squared_error(y.iloc[va], pred) ** 0.5))
    return float(np.mean(scores))


def main() -> None:
    configure_logging(log_file="reports/logs/optimize.log")
    p = load_config("params.yaml")
    set_global_seed(p["seed"])
    m = p["model"]

    df = pd.read_parquet("data/synthetic/rides.parquet")
    x, y = make_xy(df, m["target"])

    Path("reports/metrics").mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=p["seed"])
    study = optuna.create_study(
        direction=m["optuna"]["direction"],
        sampler=sampler,
        storage="sqlite:///reports/metrics/optuna.db",
        study_name="lgbm",
        load_if_exists=True,
    )
    study.optimize(
        lambda t: objective(t, x, y, m["cv_folds"], p["seed"]),
        n_trials=m["optuna"]["n_trials"],
        timeout=m["optuna"]["timeout_s"],
    )

    Path("reports/metrics/best_params.json").write_text(
        json.dumps(study.best_params, indent=2), encoding="utf-8"
    )
    LOG.info("melhor RMSE CV: %.4f", study.best_value)


if __name__ == "__main__":
    main()
