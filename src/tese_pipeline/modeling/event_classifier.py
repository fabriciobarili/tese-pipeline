"""Classificador de corridas sob evento de demanda (Alvo A).

Prevê, para cada corrida sintética, se ela ocorre sob um evento de demanda:
  - nenhum
  - voo_atrasado     (evento de chegada atrasada no SBPA)
  - estresse_termico (evento de desconforto térmico UTCI no SBPA)

Pilha: LightGBM + SMOTE (rebalanceamento) + SHAP (explicabilidade) e uma
narração determinística em linguagem natural. O alvo é derivado de ``event_name``;
nenhuma feature vaza o rótulo (``event_name``/``chained_ride`` ficam de fora).

Dados: amostra estratificada do GCS em ``_tmp_big`` (config/event_classifier.yaml).
Treino é rebalanceado (subamostra da majoritária + SMOTE); o TESTE mantém a
distribuição natural, para avaliação honesta.
"""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

from tese_pipeline.config import load_config, set_global_seed
from tese_pipeline.logging_conf import configure_logging

LOG = logging.getLogger(__name__)
EARTH_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distância great-circle (km) vetorizada."""
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_KM * 2 * np.arcsin(np.sqrt(a))


def to_target(event_name: pd.Series) -> pd.Series:
    s = event_name.astype(str)
    out = np.where(s.str.startswith("Estresse Termico"), "estresse_termico",
                   np.where(s == "Voo Atrasado SBPA", "voo_atrasado", "nenhum"))
    return pd.Series(out, index=event_name.index)


def load_data(cfg: dict) -> pd.DataFrame:
    # 'month' é partição (vem do caminho month=N), não está dentro do arquivo;
    # derivamos de 'date'.
    cols = ["event_name", "date", "hour", "weekday", "trip_distance_km",
            "trip_time_min", "pickup_time_min", "h3_distance_to_call", "trip_seq",
            "origin_lat", "origin_lng", "driver_rating", "driver_prior_trips",
            "driver_rating_class"]
    src = cfg["data"].get("source", cfg["data"].get("sample_dir"))
    if str(src).startswith("gs://"):
        # dataset completo no GCS (particionado por mês). Lê só as colunas
        # necessárias e converte liberando memória (self_destruct) — essencial
        # para 267M linhas em VM de alta memória.
        import pyarrow.dataset as ds
        from pyarrow.fs import FileSystem
        fs, root = FileSystem.from_uri(src)
        dataset = ds.dataset(root, filesystem=fs, format="parquet", partitioning="hive")
        table = dataset.to_table(columns=cols)
        LOG.info("lendo do GCS: %s | %d linhas", src, table.num_rows)
        df = table.to_pandas(split_blocks=True, self_destruct=True)
        del table
    else:
        files = sorted(glob.glob(f"{src}/*.parquet"))
        if not files:
            raise FileNotFoundError(f"sem shards em {src}")
        df = pd.concat((pd.read_parquet(f, columns=cols) for f in files),
                       ignore_index=True)
        LOG.info("amostra local carregada: %d linhas de %d shards", len(df), len(files))
    return df


def build_features(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feat = cfg["features"]
    df = df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.month.astype("int16")
    df["dist_to_airport_km"] = haversine_km(
        df["origin_lat"].to_numpy(), df["origin_lng"].to_numpy(),
        feat["airport_lat"], feat["airport_lng"]).round(3)
    le = LabelEncoder()
    df["driver_rating_class_enc"] = le.fit_transform(df["driver_rating_class"].astype(str))

    y = to_target(df["event_name"])
    features = list(feat["numeric"]) + ["driver_rating_class_enc"]
    X = df[features].fillna(0.0).astype("float32")
    return X, y, features


def resample_train(X: pd.DataFrame, y: pd.Series, cfg: dict, seed: int):
    """Subamostra majoritárias e aplica SMOTE nas minoritárias até o alvo/classe."""
    target = int(cfg["resampling"]["target_per_class"])
    k = int(cfg["resampling"]["smote_k_neighbors"])
    counts = y.value_counts().to_dict()
    LOG.info("treino (natural): %s", counts)

    under = {c: min(n, target) for c, n in counts.items() if n > target}
    if under:
        X, y = RandomUnderSampler(sampling_strategy=under, random_state=seed).fit_resample(X, y)
    over = {c: target for c in counts if counts[c] < target}
    if over:
        X, y = SMOTE(sampling_strategy=over, k_neighbors=k, random_state=seed).fit_resample(X, y)
    LOG.info("treino (rebalanceado): %s", pd.Series(y).value_counts().to_dict())
    return X, y


def evaluate(clf, X_test, y_test, classes) -> dict:
    y_pred = clf.predict(X_test)
    # predict_proba segue a ordem de clf.classes_; reordena para `classes`
    proba = clf.predict_proba(X_test)
    order = [list(clf.classes_).index(c) for c in classes]
    proba = proba[:, order]
    yb = label_binarize(y_test, classes=classes)
    roc = roc_auc_score(yb, proba, average="weighted", multi_class="ovr")
    rep = classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()
    m = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
        "mcc": round(matthews_corrcoef(y_test, y_pred), 4),
        "roc_auc_ovr_weighted": round(roc, 4),
        "per_class": {c: {k: round(rep[c][k], 4) for k in ("precision", "recall", "f1-score", "support")}
                      for c in classes},
        "confusion_matrix": {"labels": classes, "matrix": cm},
    }
    return m


def shap_importances(clf, X_sample, features, classes, out_dir: Path, top: int) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    expl = shap.TreeExplainer(clf)
    sv = expl.shap_values(X_sample)
    # normaliza para lista de arrays (n_samples, n_features), uma por classe.
    # SHAP >=0.5 retorna ndarray 3-D (n, features, classes); versões antigas,
    # uma lista. A ordem das classes segue clf.classes_.
    if isinstance(sv, list):
        sv_list = sv
    elif getattr(sv, "ndim", 2) == 3:
        sv_list = [sv[:, :, k] for k in range(sv.shape[2])]
    else:
        sv_list = [sv]
    model_classes = list(clf.classes_)

    glob_imp = np.mean([np.abs(s).mean(axis=0) for s in sv_list], axis=0)
    glob_imp = np.asarray(glob_imp).ravel()
    order = np.argsort(-glob_imp)
    ranking = {features[int(i)]: float(round(float(glob_imp[int(i)]), 5)) for i in order}

    per_class = {}
    if len(sv_list) == len(model_classes):
        for k, c in enumerate(model_classes):
            imp = np.abs(sv_list[k]).mean(axis=0).ravel()
            per_class[c] = dict(sorted(zip(features, imp), key=lambda x: -x[1]))

    try:
        plt.figure()
        plot_arg = sv_list if len(sv_list) > 1 else sv_list[0]
        shap.summary_plot(plot_arg, X_sample, feature_names=features, show=False,
                          max_display=top, plot_type="bar")
        plt.tight_layout(); plt.savefig(out_dir / "shap_summary.png", dpi=140, bbox_inches="tight")
        plt.close()
    except Exception as e:  # noqa: BLE001
        LOG.warning("summary_plot falhou (%s); seguindo com o ranking numérico", e)

    return {"global": ranking,
            "per_class": {c: {k: float(round(v, 5)) for k, v in list(d.items())[:top]}
                          for c, d in per_class.items()}}


def narrate(metrics: dict, shap_imp: dict, classes: list[str]) -> str:
    top_global = list(shap_imp["global"].keys())[:5]
    L = ["# Narração — Classificação de corridas sob evento de demanda\n",
         f"O modelo (LightGBM + SMOTE) distingue as {len(classes)} classes "
         f"({', '.join(classes)}) com F1-macro de {metrics['f1_macro']} e "
         f"ROC-AUC (OvR) de {metrics['roc_auc_ovr_weighted']}.\n",
         f"As variáveis mais associadas às predições são: {', '.join(top_global)}.\n"]
    for c in classes:
        pc = metrics["per_class"][c]
        feats = list(shap_imp["per_class"].get(c, {}).keys())[:4]
        L.append(f"- **{c}**: recall {pc['recall']}, precisão {pc['precision']} "
                 f"(suporte {int(pc['support'])}). Fatores mais associados: "
                 f"{', '.join(feats) if feats else 'n/d'}.")
    L.append("\nObservação: linguagem associativa; as relações refletem o processo "
             "gerador sintético (eventos ancorados no aeroporto SBPA e em janelas "
             "horárias/sazonais), não causalidade observada no mundo real.")
    return "\n".join(L)


def main() -> None:
    configure_logging(log_file="reports/logs/event_classifier.log")
    params = load_config("params.yaml")
    seed = params["seed"]
    set_global_seed(seed)
    cfg = load_config("config/event_classifier.yaml")
    classes = cfg["target"]["classes"]

    df = load_data(cfg)
    X, y, features = build_features(df, cfg)
    del df
    dist = y.value_counts(normalize=True).mul(100).round(2).to_dict()
    LOG.info("alvo (distribuição natural %%): %s", dist)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["data"]["test_frac"], stratify=y, random_state=seed)
    LOG.info("split: treino=%d | teste=%d (teste mantém distribuição natural)",
             len(X_tr), len(X_te))

    X_rb, y_rb = resample_train(X_tr, y_tr, cfg, seed)
    del X_tr, y_tr

    lp = cfg["model"]["lgbm"]
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=len(classes),
                             verbose=-1, **lp)

    # CV leve (macro-F1) num subconjunto do treino rebalanceado.
    # n_jobs=1 no cross_val_score: o paralelismo fica DENTRO do LightGBM; evita
    # sobre-subscrição (cross_val_score n_jobs=-1 × LGBM n_jobs=-1 trava a VM).
    folds = int(cfg["model"].get("cv_folds", 0) or 0)
    if folds > 1:
        n_cv = min(300000, len(X_rb))
        idx = np.random.RandomState(seed).choice(len(X_rb), n_cv, replace=False)
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        cv_scores = cross_val_score(clf, X_rb.iloc[idx], pd.Series(y_rb).iloc[idx],
                                    cv=cv, scoring="f1_macro", n_jobs=1)
        cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())
        LOG.info("CV f1_macro: %.4f ± %.4f", cv_mean, cv_std)
    else:
        cv_mean = cv_std = float("nan")
        LOG.info("CV desativada (cv_folds<=1)")

    LOG.info("treinando modelo final em %d linhas...", len(X_rb))
    clf.fit(X_rb, y_rb)

    # avaliação pode ser cara em teste gigante; opcionalmente subamostra
    # (mantendo a distribuição natural via estratificação).
    eval_cap = int(cfg["model"].get("eval_max_rows", 0) or 0)
    if eval_cap and len(X_te) > eval_cap:
        X_te, _, y_te, _ = train_test_split(X_te, y_te, train_size=eval_cap,
                                            stratify=y_te, random_state=seed)
        LOG.info("teste subamostrado para %d (avaliação tratável)", len(X_te))

    LOG.info("avaliando em %d linhas de teste...", len(X_te))
    metrics = evaluate(clf, X_te, y_te, classes)
    metrics["cv_f1_macro_mean"] = round(cv_mean, 4) if cv_mean == cv_mean else None
    metrics["cv_f1_macro_std"] = round(cv_std, 4) if cv_std == cv_std else None
    metrics["target_distribution_pct"] = dist
    metrics["n_test"] = int(len(X_te))
    LOG.info("TESTE: acc=%.3f f1_macro=%.3f f1_w=%.3f mcc=%.3f roc=%.3f",
             metrics["accuracy"], metrics["f1_macro"], metrics["f1_weighted"],
             metrics["mcc"], metrics["roc_auc_ovr_weighted"])

    out_dir = Path("reports/event_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # salva métricas e modelo ANTES do SHAP (etapa cara já concluída)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    clf.booster_.save_model(str(out_dir / "lgbm_event.txt"))
    LOG.info("métricas e modelo salvos em %s", out_dir)

    # SHAP (não deve derrubar o run se falhar)
    try:
        n_shap = min(int(cfg["explain"]["shap_sample"]), len(X_te))
        X_shap = X_te.sample(n_shap, random_state=seed)
        shap_imp = shap_importances(clf, X_shap, features, classes, out_dir,
                                    int(cfg["explain"]["top_features"]))
        (out_dir / "shap_importances.json").write_text(json.dumps(shap_imp, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "narration.md").write_text(narrate(metrics, shap_imp, classes), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        LOG.warning("SHAP/narração falhou (%s); métricas e modelo já estão salvos", e)

    LOG.info("artefatos salvos em %s", out_dir)
    for c in classes:
        pc = metrics["per_class"][c]
        LOG.info("  %-18s recall=%.3f precision=%.3f f1=%.3f (n=%d)",
                 c, pc["recall"], pc["precision"], pc["f1-score"], int(pc["support"]))


if __name__ == "__main__":
    main()
