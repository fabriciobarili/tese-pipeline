"""Estágio 6b — Treino final do LightGBM e avaliação fora da amostra (split temporal)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tese_pipeline.config import load_config, set_global_seed
from tese_pipeline.logging_conf import configure_logging
from tese_pipeline.modeling.dataset import make_xy, temporal_split

LOG = logging.getLogger(__name__)


def main() -> None:
    configure_logging(log_file="reports/logs/train.log")
    p = load_config("params.yaml")
    set_global_seed(p["seed"])
    m = p["model"]

    df = pd.read_parquet("data/synthetic/rides.parquet")
    train_df, test_df = temporal_split(df, test_frac=m["test_frac"])
    x_tr, y_tr = make_xy(train_df, m["target"])
    x_te, y_te = make_xy(test_df, m["target"])
    x_te = x_te.reindex(columns=x_tr.columns, fill_value=0)  # alinha one-hot

    best = json.loads(Path("reports/metrics/best_params.json").read_text(encoding="utf-8"))
    params = {
        "objective": "regression", "metric": "rmse", "seed": p["seed"],
        "deterministic": True, "force_row_wise": True, "verbosity": -1, **best,
    }
    model = lgb.train(params, lgb.Dataset(x_tr, y_tr), num_boost_round=2000)

    pred = model.predict(x_te)
    baseline = np.full_like(y_te, fill_value=float(y_tr.mean()), dtype=float)
    eval_metrics = {
        "rmse": float(mean_squared_error(y_te, pred) ** 0.5),
        "mae": float(mean_absolute_error(y_te, pred)),
        "r2": float(r2_score(y_te, pred)),
        "baseline_rmse": float(mean_squared_error(y_te, baseline) ** 0.5),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
    }

    Path("models").mkdir(exist_ok=True)
    model.save_model("models/lgbm.txt")
    Path("reports/metrics").mkdir(parents=True, exist_ok=True)
    Path("reports/metrics/eval.json").write_text(
        json.dumps(eval_metrics, indent=2), encoding="utf-8"
    )
    LOG.info("avaliação: %s", eval_metrics)


if __name__ == "__main__":
    main()
