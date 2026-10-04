"""Estágio 6c — Explicabilidade com SHAP sobre o LightGBM treinado."""

from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from tese_pipeline.config import load_config, set_global_seed  # noqa: E402
from tese_pipeline.logging_conf import configure_logging  # noqa: E402
from tese_pipeline.modeling.dataset import make_xy  # noqa: E402

LOG = logging.getLogger(__name__)


def main() -> None:
    configure_logging(log_file="reports/logs/explain.log")
    p = load_config("params.yaml")
    set_global_seed(p["seed"])
    ex = p["explain"]

    model = lgb.Booster(model_file="models/lgbm.txt")
    df = pd.read_parquet("data/synthetic/rides.parquet")
    x, _ = make_xy(df, p["model"]["target"])
    sample = x.sample(min(ex["shap_sample"], len(x)), random_state=p["seed"])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    out = Path("reports/shap")
    out.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(shap_values, sample, show=False, max_display=ex["top_features"])
    plt.tight_layout()
    plt.savefig(out / "summary.png", dpi=150)
    plt.close()

    importance = (
        pd.Series(abs(shap_values).mean(axis=0), index=sample.columns)
        .sort_values(ascending=False)
    )
    importance.to_json(out / "mean_abs_shap.json", indent=2)
    LOG.info("top features: %s", list(importance.head(ex["top_features"]).index))


if __name__ == "__main__":
    main()
