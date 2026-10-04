"""Estágio: Análise de padrões de atraso — adaptado de:

  Martins et al. (2025). "Flight delay prediction: Evaluating machine learning
  algorithms for enhanced accuracy." PMC12685205. Sec. 3.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12685205/

Adaptações em relação ao paper original:
  - Dataset: ANAC VRA 2025 (chegadas SBPA) em vez do Kaggle BTS 2018-2022
  - Features adicionais: meteorologia Open-Meteo ERA5 no aeroporto de chegada
    (precipitation, wind_speed, temperature_2m, weather_code) — ausente no paper
  - Classificadores: Random Forest, Decision Tree, LightGBM (substitui SVC/KNN/NB
    por eficiência em tabular data), Logistic Regression como baseline
  - Saída extra: base de conhecimento em JSON com regras, importâncias e padrões

Target: delay_category
  no_delay   ← delay_arr_min ≤ 0
  small      ← 0  < delay_arr_min < 15   (limiar IATA)
  medium     ← 15 ≤ delay_arr_min < 45
  large      ← delay_arr_min ≥ 45
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from tese_pipeline.config import load_config, set_global_seed
from tese_pipeline.logging_conf import configure_logging
from tese_pipeline.provenance import write_provenance

LOG = logging.getLogger(__name__)

# Limites do target (min) — alinhados com o paper (sec. 3.1)
DELAY_BINS   = [float("-inf"), 0, 15, 45, float("inf")]
DELAY_LABELS = ["no_delay", "small", "medium", "large"]

# Features candidatas — paper usa 25 via SelectKBest; aqui seleção por importância
BASE_FEATURES = [
    # temporais
    "hour", "dow", "month",
    # atraso de partida propagado (causa mais comum de atraso de chegada)
    "delay_dep_min",
    # meteorologia no aeroporto de chegada (SBPA) — contribuição adicional
    "precipitation", "wind_speed_10m", "temperature_2m", "weather_code",
    "is_rain",
    # características operacionais
    "route_type_enc",     # N/I/C codificado
    "airline_icao_enc",   # companhia
    "origin_icao_enc",    # aeroporto de origem
]


def build_delay_dataset(flights: pd.DataFrame, meteo: pd.DataFrame) -> pd.DataFrame:
    """Une ANAC VRA + meteorologia e constrói as features para classificação."""
    df = flights.copy()

    # join com meteorologia por hora local
    if "arr_hour_local" in df.columns:
        df["ts_hour"] = pd.to_datetime(df["arr_hour_local"])
    else:
        df["ts_hour"] = pd.to_datetime(df["actual_arr"], utc=True) \
            .dt.tz_convert("America/Sao_Paulo").dt.floor("h")

    meteo = meteo.copy()
    meteo["ts_hour"] = (
        pd.to_datetime(meteo["time"])
        .dt.tz_localize("UTC")
        .dt.tz_convert("America/Sao_Paulo")
        .dt.floor("h")
    )
    meteo["location"] = meteo["location"].str.upper()

    df = df.merge(
        meteo,
        left_on=["dest_icao", "ts_hour"],
        right_on=["location", "ts_hour"],
        how="left",
    )

    # features temporais
    df["hour"]  = df["ts_hour"].dt.hour
    df["dow"]   = df["ts_hour"].dt.dayofweek
    df["month"] = df["ts_hour"].dt.month
    df["is_rain"] = (df["precipitation"].fillna(0) > 0).astype(int)

    # label encoding para categóricas
    for col, new_col in [
        ("route_type", "route_type_enc"),
        ("airline_icao", "airline_icao_enc"),
        ("origin_icao", "origin_icao_enc"),
    ]:
        if col in df.columns:
            le = LabelEncoder()
            df[new_col] = le.fit_transform(df[col].fillna("UNKNOWN"))
        else:
            df[new_col] = -1

    # target
    df["delay_category"] = pd.cut(
        df["delay_arr_min"].fillna(0),
        bins=DELAY_BINS,
        labels=DELAY_LABELS,
        right=True,
    ).astype(str)

    return df


def evaluate_classifier(name: str, clf, X_train, y_train, X_test, y_test,
                        cv: StratifiedKFold) -> dict:
    """Treina, aplica CV 10-fold e avalia no teste. Retorna dict de métricas."""
    # cross-val no treino
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # ROC-AUC multiclass (OvR)
    try:
        y_prob = clf.predict_proba(X_test)
        classes = sorted(set(y_test))
        roc_auc = roc_auc_score(
            pd.get_dummies(y_test)[classes],
            y_prob[:, :len(classes)],
            average="weighted",
            multi_class="ovr",
        )
    except Exception:
        roc_auc = float("nan")

    metrics = {
        "classifier": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "mcc": round(matthews_corrcoef(y_test, y_pred), 4),
        "roc_auc_weighted": round(roc_auc, 4),
        "cv_f1_mean": round(cv_scores.mean(), 4),
        "cv_f1_std": round(cv_scores.std(), 4),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }
    LOG.info(
        "%s  acc=%.3f  f1=%.3f  mcc=%.3f  roc=%.3f  cv_f1=%.3f±%.3f",
        name.ljust(25),
        metrics["accuracy"], metrics["f1_weighted"], metrics["mcc"],
        metrics["roc_auc_weighted"], metrics["cv_f1_mean"], metrics["cv_f1_std"],
    )
    return metrics


def build_knowledge_base(results: list[dict], rf: RandomForestClassifier,
                         feature_names: list[str], dt: DecisionTreeClassifier) -> dict:
    """Extrai regras e importâncias para a base de conhecimento."""
    best = max(results, key=lambda r: r["f1_weighted"])

    importances = dict(sorted(
        zip(feature_names, rf.feature_importances_),
        key=lambda x: -x[1]
    ))

    # regras do Decision Tree (max_depth=4 para legibilidade)
    dt_rules = export_text(dt, feature_names=feature_names, max_depth=4)

    kb = {
        "best_classifier": best["classifier"],
        "best_f1_weighted": best["f1_weighted"],
        "best_accuracy": best["accuracy"],
        "feature_importances_rf": {k: round(v, 5) for k, v in importances.items()},
        "top_5_features": list(importances.keys())[:5],
        "decision_tree_rules": dt_rules,
        "all_classifiers": [
            {k: v for k, v in r.items() if k != "report"}
            for r in results
        ],
        "class_distribution_test": {},
        "methodology_ref": (
            "Adapted from: Martins et al. (2025) 'Flight delay prediction: "
            "Evaluating machine learning algorithms for enhanced accuracy.' "
            "PMC12685205. https://pmc.ncbi.nlm.nih.gov/articles/PMC12685205/"
        ),
    }
    return kb


def main() -> None:
    configure_logging(log_file="reports/logs/delay_analysis.log")
    params = load_config("params.yaml")
    set_global_seed(params["seed"])

    flights = pd.read_parquet("data/raw/flights_anac.parquet")
    meteo   = pd.read_parquet("data/raw/meteo.parquet")

    LOG.info("Voos carregados: %d | Meteo linhas: %d", len(flights), len(meteo))

    df = build_delay_dataset(flights, meteo)

    # remove linhas sem delay calculado
    df = df.dropna(subset=["delay_arr_min"])
    LOG.info("Dataset após limpeza: %d linhas", len(df))
    LOG.info("Distribuição de classes:\n%s", df["delay_category"].value_counts().to_string())

    features = [f for f in BASE_FEATURES if f in df.columns]
    LOG.info("Features usadas (%d): %s", len(features), features)

    X = df[features].fillna(0)
    y = df["delay_category"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features)

    # split temporal 80/20 (mesma filosofia anti-leakage da pipeline principal)
    df = df.reset_index(drop=True)
    cut = int(len(df) * 0.80)
    X_train, X_test = X_scaled.iloc[:cut], X_scaled.iloc[cut:]
    y_train, y_test = y.iloc[:cut], y.iloc[cut:]

    LOG.info("Treino: %d | Teste: %d", len(X_train), len(X_test))

    # rebalanceamento com SMOTE (paper sec. 3.2)
    ros = RandomOverSampler(random_state=params["seed"])
    smote = SMOTE(random_state=params["seed"], k_neighbors=3)
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
    try:
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    except Exception as e:
        LOG.warning("SMOTE falhou (%s), usando ROS", e)
        X_train_smote, y_train_smote = X_train_ros, y_train_ros

    LOG.info("Treino após ROS:   %d", len(X_train_ros))
    LOG.info("Treino após SMOTE: %d", len(X_train_smote))

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=params["seed"])

    classifiers_ros = [
        ("Random Forest (ROS)",   RandomForestClassifier(n_estimators=100, random_state=params["seed"]), X_train_ros, y_train_ros),
        ("Decision Tree (ROS)",   DecisionTreeClassifier(max_depth=8, random_state=params["seed"]), X_train_ros, y_train_ros),
        ("LightGBM (ROS)",        lgb.LGBMClassifier(n_estimators=100, random_state=params["seed"], verbose=-1), X_train_ros, y_train_ros),
        ("Logistic Reg. (ROS)",   LogisticRegression(max_iter=500, random_state=params["seed"]), X_train_ros, y_train_ros),
        ("Random Forest (SMOTE)", RandomForestClassifier(n_estimators=100, random_state=params["seed"]), X_train_smote, y_train_smote),
        ("Decision Tree (SMOTE)", DecisionTreeClassifier(max_depth=8, random_state=params["seed"]), X_train_smote, y_train_smote),
        ("LightGBM (SMOTE)",      lgb.LGBMClassifier(n_estimators=100, random_state=params["seed"], verbose=-1), X_train_smote, y_train_smote),
    ]

    results = []
    rf_model = dt_model = None

    for name, clf, Xtr, ytr in classifiers_ros:
        res = evaluate_classifier(name, clf, Xtr, ytr, X_test, y_test, cv)
        results.append(res)
        if "Random Forest (ROS)" in name and rf_model is None:
            rf_model = clf
        if "Decision Tree (ROS)" in name and dt_model is None:
            dt_model = clf

    # base de conhecimento
    kb = build_knowledge_base(results, rf_model, features, dt_model)
    kb["class_distribution_test"] = y_test.value_counts().to_dict()

    # salva resultados
    out_dir = Path("reports/delay_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "knowledge_base.json").write_text(
        json.dumps(kb, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    all_metrics = [
        {k: v for k, v in r.items() if k != "report"} for r in results
    ]
    (out_dir / "classifier_metrics.json").write_text(
        json.dumps(all_metrics, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # relatório Markdown
    report_lines = [
        "# Análise de Padrões de Atraso — SBPA 2025\n",
        f"**Referência metodológica:** PMC12685205 (Martins et al., 2025)\n",
        f"**Dataset:** ANAC VRA 2025 — chegadas SBPA ({len(df)} voos)\n\n",
        "## Distribuição do Target\n",
        "| Categoria | N | % |\n|---|---|---|\n",
    ]
    for cat, n in sorted(df["delay_category"].value_counts().items()):
        pct = n / len(df) * 100
        report_lines.append(f"| {cat} | {n} | {pct:.1f}% |\n")

    report_lines += [
        "\n## Desempenho dos Classificadores\n",
        "| Classificador | Acurácia | F1 (weighted) | MCC | ROC-AUC |\n|---|---|---|---|---|\n",
    ]
    for r in sorted(results, key=lambda x: -x["f1_weighted"]):
        report_lines.append(
            f"| {r['classifier']} | {r['accuracy']} | {r['f1_weighted']} | {r['mcc']} | {r['roc_auc_weighted']} |\n"
        )

    report_lines += [
        f"\n## Top-5 Features (Random Forest)\n",
        "| Rank | Feature | Importância |\n|---|---|---|\n",
    ]
    for i, (feat, imp) in enumerate(list(kb["feature_importances_rf"].items())[:5], 1):
        report_lines.append(f"| {i} | `{feat}` | {imp:.5f} |\n")

    report_lines += [
        "\n## Regras de Decisão (Decision Tree, max_depth=4)\n",
        "```\n", kb["decision_tree_rules"], "\n```\n",
    ]

    (out_dir / "delay_analysis_report.md").write_text(
        "".join(report_lines), encoding="utf-8"
    )

    write_provenance(
        out_dir / "knowledge_base.json",
        source="delay_analysis:anac_vra+open_meteo",
        params={
            "features": features,
            "n_classifiers": len(classifiers_ros),
            "resampling": ["ROS", "SMOTE"],
            "cv_folds": 10,
            "methodology_ref": "PMC12685205",
        },
    )

    LOG.info("Base de conhecimento salva em %s", out_dir)
    LOG.info("Melhor classificador: %s (F1=%.4f)", kb["best_classifier"], kb["best_f1_weighted"])


if __name__ == "__main__":
    main()
