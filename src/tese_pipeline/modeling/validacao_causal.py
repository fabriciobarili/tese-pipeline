"""Validação causal via SHAP — recuperação da estrutura injetada no gerador sintético.

NÃO é inferência causal no mundo real: é verificar se o classificador de eventos
(``event_classifier``) **recupera**, via SHAP, as relações que o processo gerador injetou —
sobretudo o **ancoramento espacial** dos eventos no SBPA (``dist_to_airport_km``) e os padrões
temporais. A força "causal" vem de o DGP ser conhecido e controlado.

Passos (ver Seção 5.3 do artigo):
  2. Importância SHAP global por classe (ranking).
  3. Dependence plot do driver espacial por classe (sinal/forma).
  4. Teste de INTERVENÇÃO: relocar radialmente a origem da corrida e varrer ``dist_to_airport_km``
     recomputando lat/lng de forma consistente; a probabilidade do evento deve cair monotonicamente.
  5. Score de concordância: ranking esperado (pré-registrado) × ranking SHAP + monotonicidade.

Rodar:  python -m tese_pipeline.modeling.validacao_causal
"""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from tese_pipeline.config import load_config, set_global_seed  # noqa: E402
from tese_pipeline.logging_conf import configure_logging  # noqa: E402
from tese_pipeline.modeling.event_classifier import (  # noqa: E402
    build_features,
    resample_train,
)

LOG = logging.getLogger(__name__)

OUT_DIR = Path("reports/causal_validation")
DATA_SOURCE = "_tmp_big"
N_SHARDS = 40                 # amostra enxuta p/ a validação (tratável localmente)
TARGET_PER_CLASS = 60000      # rebalanceamento do treino (menor que o run completo, por velocidade)
EARTH_KM = 6371.0088
COLS = ["event_name", "date", "hour", "weekday", "trip_distance_km", "trip_time_min",
        "pickup_time_min", "h3_distance_to_call", "trip_seq", "origin_lat", "origin_lng",
        "driver_rating", "driver_prior_trips", "driver_rating_class"]

# Expectativa PRÉ-REGISTRADA (antes de olhar o SHAP): quais features o gerador torna causais.
# Eventos ancorados no SBPA -> dist_to_airport_km é o driver espacial de AMBAS as classes de
# evento; a distinção clima×voo vem dos padrões TEMPORAIS (hora/mês/dia).
EXPECTED_DRIVERS = {
    "voo_atrasado": ["dist_to_airport_km", "hour", "weekday", "month"],
    "estresse_termico": ["dist_to_airport_km", "month", "hour", "weekday"],
}


def _bearing(lat1, lon1, lat2, lon2):
    la1, la2 = np.radians(lat1), np.radians(lat2)
    dlo = np.radians(lon2 - lon1)
    y = np.sin(dlo) * np.cos(la2)
    x = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(dlo)
    return np.degrees(np.arctan2(y, x))


def _ponto_a_distancia(lat1, lon1, bearing_deg, dist_km):
    br = np.radians(bearing_deg)
    d = np.asarray(dist_km) / EARTH_KM
    la1, lo1 = np.radians(lat1), np.radians(lon1)
    la2 = np.arcsin(np.sin(la1) * np.cos(d) + np.cos(la1) * np.sin(d) * np.cos(br))
    lo2 = lo1 + np.arctan2(np.sin(br) * np.sin(d) * np.cos(la1),
                           np.cos(d) - np.sin(la1) * np.sin(la2))
    return np.degrees(la2), np.degrees(lo2)


def _shap_por_classe(sv, k):
    """Normaliza o retorno do SHAP para a matriz (n, features) da classe k."""
    if isinstance(sv, list):
        return sv[k]
    if getattr(sv, "ndim", 2) == 3:
        return sv[:, :, k]
    return sv


def carregar_amostra(cfg: dict):
    files = sorted(glob.glob(f"{DATA_SOURCE}/*.parquet"))[:N_SHARDS]
    if not files:
        raise FileNotFoundError(f"sem shards em {DATA_SOURCE}")
    df = pd.concat((pd.read_parquet(f, columns=COLS) for f in files), ignore_index=True)
    LOG.info("amostra: %d linhas de %d shards", len(df), len(files))
    X, y, features = build_features(df, cfg)
    return X, y, features


def treinar(cfg: dict, X, y, seed: int):
    classes = cfg["target"]["classes"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cfg["data"]["test_frac"],
                                              stratify=y, random_state=seed)
    cfg = {**cfg, "resampling": {**cfg["resampling"], "target_per_class": TARGET_PER_CLASS}}
    X_rb, y_rb = resample_train(X_tr, y_tr, cfg, seed)
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=len(classes), verbose=-1,
                             **cfg["model"]["lgbm"])
    clf.fit(X_rb, y_rb)
    return clf, X_te, y_te, classes


def passo2_3_shap(clf, X_te, features, classes, seed, n_shap=4000):
    """Importância SHAP por classe (passo 2) + dependence do driver espacial (passo 3)."""
    Xs = X_te.sample(min(n_shap, len(X_te)), random_state=seed)
    sv = shap.TreeExplainer(clf).shap_values(Xs)
    model_classes = list(clf.classes_)
    ranking = {}
    for c in model_classes:
        k = model_classes.index(c)
        imp = np.abs(_shap_por_classe(sv, k)).mean(axis=0).ravel()
        ordem = np.argsort(-imp)
        ranking[c] = [{"feature": features[int(i)], "shap_abs": float(round(imp[int(i)], 5))}
                      for i in ordem]
    # dependence de dist_to_airport_km para as classes de evento
    di = features.index("dist_to_airport_km")
    for c in ("voo_atrasado", "estresse_termico"):
        if c not in model_classes:
            continue
        k = model_classes.index(c)
        vals = _shap_por_classe(sv, k)[:, di]
        plt.figure(figsize=(5, 3.5))
        plt.scatter(Xs["dist_to_airport_km"], vals, s=5, alpha=0.3)
        plt.axhline(0, color="gray", lw=0.8, ls="--")
        plt.xlabel("dist_to_airport_km (km)")
        plt.ylabel(f"SHAP → P({c})")
        plt.title(f"Dependence: distância ao SBPA × efeito ({c})")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"dependence_dist_{c}.png", dpi=140, bbox_inches="tight")
        plt.close()
    return ranking


def passo4_intervencao(clf, X_te, y_te, features, classes, seed, airport, n_rides=400):
    """Intervenção radial: relocar a origem a distâncias crescentes do SBPA e medir P(evento)."""
    model_classes = list(clf.classes_)
    grid = np.array([0.5, 1, 2, 3, 5, 7, 10, 13, 16, 20, 25, 30, 35, 40], dtype=float)
    curvas = {}
    for c in ("voo_atrasado", "estresse_termico"):
        if c not in model_classes:
            continue
        k = model_classes.index(c)
        pos = X_te[(y_te == c).values]
        if len(pos) == 0:
            continue
        base = pos.sample(min(n_rides, len(pos)), random_state=seed).reset_index(drop=True)
        brg = _bearing(airport[0], airport[1], base["origin_lat"].to_numpy(),
                       base["origin_lng"].to_numpy())
        media = []
        for d in grid:
            m = base.copy()
            lat2, lon2 = _ponto_a_distancia(airport[0], airport[1], brg, d)
            m["origin_lat"] = lat2
            m["origin_lng"] = lon2
            m["dist_to_airport_km"] = d
            proba = clf.predict_proba(m[features])[:, k]
            media.append(float(np.mean(proba)))
        rho, _ = spearmanr(grid, media)
        curvas[c] = {"grid_km": grid.tolist(), "prob_media": media,
                     "spearman_dist_prob": float(round(rho, 4)), "n_rides": int(len(base))}
        LOG.info("intervenção %s: Spearman(dist, P)=%.3f (esperado ~ -1)", c, rho)
    # figura
    plt.figure(figsize=(5.5, 3.8))
    for c, d in curvas.items():
        plt.plot(d["grid_km"], d["prob_media"], marker="o",
                 label=f"{c} (ρ={d['spearman_dist_prob']})")
    plt.xlabel("dist_to_airport_km (km) — intervenção")
    plt.ylabel("P(evento) média")
    plt.title("Intervenção espacial: afastar a origem do SBPA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "intervencao_distancia.png", dpi=140, bbox_inches="tight")
    plt.close()
    return curvas


def passo5_concordancia(ranking: dict, curvas: dict):
    conc = {}
    for c, esperado in EXPECTED_DRIVERS.items():
        if c not in ranking:
            continue
        rank_feats = [r["feature"] for r in ranking[c]]
        top5 = rank_feats[:5]
        dist_rank = rank_feats.index("dist_to_airport_km") + 1
        frac_top5 = len([f for f in top5 if f in esperado]) / len(top5)
        conc[c] = {
            "top5_shap": top5,
            "dist_to_airport_km_e_top1": rank_feats[0] == "dist_to_airport_km",
            "dist_to_airport_km_rank": int(dist_rank),
            "frac_top5_esperados": round(frac_top5, 3),
            "intervencao_spearman": curvas.get(c, {}).get("spearman_dist_prob"),
        }
    return conc


def main() -> None:
    configure_logging(log_file="reports/logs/validacao_causal.log")
    params = load_config("params.yaml")
    seed = params["seed"]
    set_global_seed(seed)
    cfg = load_config("config/event_classifier.yaml")
    cfg["data"]["source"] = DATA_SOURCE
    airport = (cfg["features"]["airport_lat"], cfg["features"]["airport_lng"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, features = carregar_amostra(cfg)
    clf, X_te, y_te, classes = treinar(cfg, X, y, seed)
    LOG.info("treino ok | teste=%d | classes=%s", len(X_te), classes)

    ranking = passo2_3_shap(clf, X_te, features, classes, seed)
    curvas = passo4_intervencao(clf, X_te, y_te, features, classes, seed, airport)
    conc = passo5_concordancia(ranking, curvas)

    (OUT_DIR / "shap_importancia_por_classe.json").write_text(
        json.dumps(ranking, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "intervencao_distancia.json").write_text(
        json.dumps(curvas, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "concordancia.json").write_text(
        json.dumps(conc, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("concordância: %s", json.dumps(conc, ensure_ascii=False))
    LOG.info("artefatos da validação causal em %s", OUT_DIR)


if __name__ == "__main__":
    main()
