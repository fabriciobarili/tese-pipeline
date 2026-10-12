"""Treino e exportação dos dois modelos binários pareados (clima/atraso) para o narrador.

Porta da seção de PRODUÇÃO do ``05_6_FeatureEng_Individual_vs_Agrupado``, adaptada à base do
pacote: eventos de corrida vêm de ``driver_trips`` (amostra estratificada no DuckDB), clima de
``data/raw/meteo.parquet`` (SBPA), voos de ``data/raw/flights_anac.parquet``.

Exporta para ``Work/narrador_artefatos/`` (consumidos pelo narrador):
  modelo_par_clima.joblib / _meta.json, modelo_par_atraso.joblib / _meta.json,
  climatologia_atraso_sbpa.csv, reputacao_slot_sbpa.csv.

Rodar:  python -m tese_pipeline.narrador.treina_pareados
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import average_precision_score, classification_report, f1_score

from tese_pipeline.config import load_config, set_global_seed
from tese_pipeline.logging_conf import configure_logging
from tese_pipeline.narrador import features as fe

LOG = logging.getLogger(__name__)

ART_DIR = Path("Work/narrador_artefatos")
METEO_PARQUET = "data/raw/meteo.parquet"
FLIGHTS_PARQUET = "data/raw/flights_anac.parquet"

# --- Parâmetros (sem constantes mágicas espalhadas; ajustáveis aqui) ---
# diretório de shards driver_trips (ou gs:// / data/synthetic/driver_trips)
DATA_SOURCE = "_tmp_big"
N_BUCKETS, POR_BUCKET = 100, 300  # amostragem estratificada bucket × classe
INICIO_TS, FIM_TS = 1735699200, 1767235200   # 2025-01-01 .. 2026-01-01 (UTC)
JANELA_PROD = "1h"
TEST_QUANTIL = 0.80
SCALE_POS_WEIGHT_ATRASO = 2.0
INCLUIR_POSFATO_ATRASO = False
LGBM_PARAMS = dict(objective="binary", n_estimators=3000, learning_rate=0.05, num_leaves=63,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                   n_jobs=-1, verbose=-1)
ES_ROUNDS = 40

COLS_META = {"y", "time", "time_window", "request_dt", "request_ts", "event_name",
             "dataset_type", "bucket"}
COLS_ESPACIAIS = ["lat", "lng", "origin_h3", "densidade_h3", "dist_aeroporto_km"]


def load_sample(seed_duck: float = 0.42) -> pd.DataFrame:
    """Amostra estratificada bucket × classe no DuckDB (setseed materializado)."""
    src = str(DATA_SOURCE)
    glob = f"{src.rstrip('/')}/**/*.parquet" if not src.endswith(".parquet") else src
    con = duckdb.connect()
    if src.startswith("gs://"):
        con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"SELECT setseed({seed_duck})").fetchall()
    query = f"""
    WITH base AS (
        SELECT request_ts, event_name, origin_h3,
            CASE
                WHEN event_name = '{fe.EVENT_VOO_EXACT}'        THEN 'DS_VOO'
                WHEN event_name LIKE '{fe.EVENT_CLIMA_PREFIX}%' THEN 'DS_CLIMA'
                WHEN event_name IS NULL                         THEN 'NULO'
                ELSE 'DS_OUTROS'
            END AS dataset_type,
            LEAST({N_BUCKETS - 1},
                  CAST((request_ts - {INICIO_TS}) * {N_BUCKETS}.0
                       / ({FIM_TS} - {INICIO_TS}) AS INTEGER)) AS bucket
        FROM read_parquet('{glob}', hive_partitioning=true, union_by_name=true)
        WHERE request_ts >= {INICIO_TS} AND request_ts < {FIM_TS}
    ),
    ranked AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY dataset_type, bucket
                                     ORDER BY RANDOM()) AS rn
        FROM base WHERE dataset_type <> 'NULO'
    )
    SELECT request_ts, event_name, origin_h3, dataset_type
    FROM ranked WHERE rn <= {POR_BUCKET}
    """
    df = con.execute(query).df()
    con.close()
    df["request_dt"] = pd.to_datetime(df["request_ts"], unit="s", utc=True)
    LOG.info("amostra: %s | %s", df.shape, df["dataset_type"].value_counts().to_dict())
    return df


def load_climate() -> pd.DataFrame:
    dfc = pd.read_parquet(METEO_PARQUET)
    dfc["time"] = pd.to_datetime(dfc["time"], utc=True, errors="coerce")
    return dfc.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)


def construir_base_producao(df_clima: pd.DataFrame, df_voos: pd.DataFrame,
                            splits: dict) -> dict:
    """Base de produção (1h): clima agregado + lag/diff + voos enriquecidos + temporal + prior."""
    w = JANELA_PROD
    c = df_clima.copy()
    c["time_window"] = c["time"].dt.floor(w)
    clima_agg = (c.drop(columns=["time"]).groupby("time_window").mean(numeric_only=True)
                 .sort_index())
    for col in fe.COLS_LAG_PROD:
        if col in clima_agg.columns:
            clima_agg[f"{col}_lag"] = clima_agg[col].shift(1)
            clima_agg[f"diff_{col}"] = clima_agg[col] - clima_agg[col].shift(1)
    clima_agg = clima_agg.reset_index()
    voos_agg = fe.agregar_voos(df_voos, w)

    ev = (pd.concat([splits["DS_OUTROS"], splits["DS_CLIMA"], splits["DS_VOO"]],
                    ignore_index=True).sort_values("request_dt").reset_index(drop=True))
    ev["time_window"] = ev["request_dt"].dt.floor(w)
    corte_prod = ev["request_dt"].quantile(TEST_QUANTIL)
    prior_df, fallback, rep_ctx = fe.construir_prior(df_voos, corte_utc=corte_prod)

    base = ev.merge(clima_agg, on="time_window", how="left").merge(voos_agg, on="time_window",
                                                                   how="left")
    for col in base.columns:
        if fe.eh_feature_voo(col):
            base[col] = base[col].fillna(0.0)
    fe.add_temporais(base, "request_dt")
    base = base.merge(prior_df, on=["dia_semana", "hora"], how="left")
    for col in fe.COLS_CLIMATOLOGIA:
        if col in base.columns:
            base[col] = base[col].fillna(fallback[col])
    return dict(base=base, corte_prod=corte_prod, prior_df=prior_df, fallback=fallback,
                rep_ctx=rep_ctx)


def selecionar_features(df: pd.DataFrame, cols_excluir: list[str]) -> list[str]:
    proibidas = set(cols_excluir) | COLS_META | set(COLS_ESPACIAIS)
    return [c for c in df.columns
            if c not in proibidas and pd.api.types.is_numeric_dtype(df[c])]


def preparar_par_ts(df: pd.DataFrame, classe_pos: str, pos_label: str, cols_excluir: list[str],
                    corte_ts: pd.Timestamp) -> dict:
    d = df[df["dataset_type"].isin(["DS_OUTROS", classe_pos])].copy()
    d = d.sort_values("request_dt").reset_index(drop=True)
    d["y"] = (d["dataset_type"] == classe_pos).astype(int)
    feats = selecionar_features(d, cols_excluir)
    tr = d[d["request_dt"] < corte_ts]
    te = d[d["request_dt"] >= corte_ts]
    return dict(Xtr=tr[feats].copy(), ytr=tr["y"].values, Xte=te[feats].copy(),
                yte=te["y"].values, feats=feats, pos_label=pos_label)


def treinar_lgbm_par(dados: dict, titulo: str, scale_pos_weight: float | None = None) -> dict:
    """Early stopping em sub-fatia temporal + refit em todo o treino (R3). macro-F1 + PR-AUC."""
    Xtr, ytr = dados["Xtr"], dados["ytr"]
    n = len(Xtr)
    k = max(1, int(n * 0.80))
    params = dict(random_state=42, **LGBM_PARAMS)
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = float(scale_pos_weight)
    if n >= 2 and 0 < k < n and len(np.unique(ytr[:k])) > 1:
        probe = LGBMClassifier(**params)
        probe.fit(Xtr.iloc[:k], ytr[:k], eval_set=[(Xtr.iloc[k:], ytr[k:])],
                  eval_metric="binary_logloss",
                  callbacks=[early_stopping(ES_ROUNDS), log_evaluation(0)])
        best_iter = int(probe.best_iteration_ or params["n_estimators"])
    else:
        best_iter = 300
    model = LGBMClassifier(**{**params, "n_estimators": best_iter})
    model.fit(Xtr, ytr)
    proba = model.predict_proba(dados["Xte"])[:, 1]
    pred = (proba >= 0.5).astype(int)
    macro = f1_score(dados["yte"], pred, average="macro")
    prauc = (average_precision_score(dados["yte"], proba)
             if len(np.unique(dados["yte"])) > 1 else float("nan"))
    LOG.info("=== %s === macro-F1=%.4f | PR-AUC=%.4f | best_iter=%d", titulo, macro, prauc,
             best_iter)
    LOG.info("\n%s", classification_report(dados["yte"], pred, digits=3, zero_division=0,
                                            target_names=["NAO_EVENTO", dados["pos_label"]]))
    return dict(model=model, macro_f1=float(macro), pr_auc=float(prauc), best_iter=best_iter,
                feats=dados["feats"])


def _exportar(r: dict, fallback: dict, model_path: Path, meta_path: Path,
              scale_pos_weight: float | None) -> dict:
    meta = {
        "janela": JANELA_PROD,
        "dist_aeroporto": False,
        "scale_pos_weight": scale_pos_weight,
        "features_lgbm": list(r["feats"]),
        "macro_f1_bal": round(r["macro_f1"], 4),
        "macro_f1_nat": round(r["macro_f1"], 4),
        "pr_auc": round(r["pr_auc"], 4),
        "best_iteration": int(r["best_iter"]),
        "posfato_atraso": bool(INCLUIR_POSFATO_ATRASO),
        "climatologia_atraso": [c for c in fe.COLS_CLIMATOLOGIA if c in r["feats"]],
        "climatologia_fallback": {c: float(fallback[c]) for c in fe.COLS_CLIMATOLOGIA},
        "taxa_atraso_global": float(fallback["taxa_atraso_hist"]),
        "obs": ("Teste balanceado por bucket (nao populacional); macro_f1_nat=bal. "
                "PR-AUC e a metrica primaria."),
    }
    joblib.dump(r["model"], model_path)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("exportado: %s (%d features)", model_path, len(meta["features_lgbm"]))
    return meta


def main() -> None:
    configure_logging(log_file="reports/logs/narrador_treino.log")
    params = load_config("params.yaml")
    set_global_seed(params["seed"])
    ART_DIR.mkdir(parents=True, exist_ok=True)

    df_comb = load_sample()
    splits = {k: df_comb[df_comb["dataset_type"] == k].copy()
              for k in ("DS_VOO", "DS_CLIMA", "DS_OUTROS")}
    df_clima = load_climate()
    df_voos = pd.read_parquet(FLIGHTS_PARQUET)

    prod = construir_base_producao(df_clima, df_voos, splits)
    base, corte, fallback = prod["base"], prod["corte_prod"], prod["fallback"]
    prod["prior_df"].to_csv(ART_DIR / "climatologia_atraso_sbpa.csv", index=False,
                            encoding="utf-8")
    prod["rep_ctx"].to_csv(ART_DIR / "reputacao_slot_sbpa.csv", index=False, encoding="utf-8")
    LOG.info("base_prod=%s | CORTE=%s | prior=%d slots", base.shape, corte, len(prod["prior_df"]))

    cols_voos = [c for c in base.columns if fe.eh_feature_voo(c)]
    # CLIMA: só físicas/temporais (sem voos, sem climatologia de atraso).
    excl_clima = sorted(set([c for c in base.columns if fe.eh_score_composto(c)]
                            + cols_voos + fe.COLS_CLIMATOLOGIA))
    # ATRASO: escores fora; voos pós-fato fora por padrão; climatologia prospectiva DENTRO.
    excl_atraso = sorted(set(c for c in base.columns if fe.eh_score_composto(c)))
    if not INCLUIR_POSFATO_ATRASO:
        excl_atraso = sorted(set(excl_atraso) | set(cols_voos))

    ns = fe.namespace_narrador(df_clima.select_dtypes("number").columns)
    for nome, (classe, pos_label, excl, spw, mpath, meta_path) in {
        "clima": ("DS_CLIMA", "CLIMA", excl_clima, None,
                  ART_DIR / "modelo_par_clima.joblib", ART_DIR / "modelo_par_clima_meta.json"),
        "atraso": ("DS_VOO", "ATRASO", excl_atraso, SCALE_POS_WEIGHT_ATRASO,
                   ART_DIR / "modelo_par_atraso.joblib",
                   ART_DIR / "modelo_par_atraso_meta.json"),
    }.items():
        d = preparar_par_ts(base, classe, pos_label, excl, corte)
        r = treinar_lgbm_par(d, f"PRODUÇÃO {pos_label}", scale_pos_weight=spw)
        fora = [f for f in r["feats"] if f not in ns]
        if fora:
            LOG.warning("[%s] features fora do namespace do narrador: %s", nome, fora)
        _exportar(r, fallback, mpath, meta_path, spw)

    LOG.info("artefatos do narrador em %s", ART_DIR)


if __name__ == "__main__":
    main()
