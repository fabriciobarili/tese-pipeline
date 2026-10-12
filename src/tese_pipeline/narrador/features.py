"""Construção de features compartilhada entre o treino dos modelos pareados e o narrador.

Centraliza o contrato de features (namespace, nomes, UTC, janela) para que treino e inferência
produzam EXATAMENTE as mesmas colunas — eliminando train/serve skew. Adaptado à base do pacote:
clima = ``meteo.parquet`` (SBPA, variáveis físicas; sem ``utci``/``surface_pressure``); voos =
``flights_anac.parquet`` (todas as chegadas realizadas em SBPA).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

# Aeroporto Salgado Filho (SBPA)
AER_LAT, AER_LNG = -29.9939, -51.1711

# Namespace de voos (mesmas listas do 05_5/05_6, para a ficha técnica e o contexto textual).
TOP_CIAS = ["TAM", "AZU", "GLO", "TAP", "ARG"]
TOP_ORIGENS = ["SBGR", "SBSP", "SBKP", "SBCF", "SBBR"]

# Variáveis climáticas com lag/diff no contrato (sem surface_pressure — ausente em meteo.parquet).
COLS_LAG_PROD = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]

# Features do prior prospectivo de atraso (climatologia + reputação por slot).
COLS_CLIMATOLOGIA = ["taxa_atraso_hist", "atraso_medio_hist", "chegadas_previstas_hist",
                     "rep_cia_slot", "rep_cia_origem_slot", "rep_origem_slot"]

# Mapa event_name (simulate_drivers) -> regime.
EVENT_VOO_EXACT = "Voo Atrasado SBPA"
EVENT_CLIMA_PREFIX = "Estresse Termico"

# Colunas de voo (enriquecidas) que são PÓS-FATO para prever atraso.
COLS_VOOS_BASE = ["qtd_voos_previstos", "qtd_empresas_aereas", "atraso_medio_min", "atraso_max_min"]


# --------------------------------------------------------------------------------------------
# Guardas anti-vazamento
# --------------------------------------------------------------------------------------------
def eh_score_composto(col: str) -> bool:
    """Escores/índices compostos de conforto térmico — mesma régua que rotulou DS_CLIMA."""
    c = col.lower()
    return any(p in c for p in ("utci", "discomfort", "stress", "score", "severidade",
                                "thermal", "conforto", "desconforto"))


def eh_vazamento_atraso(col: str) -> bool:
    """Variáveis pós-fato / definicionais de atraso (vazamento no modelo de atraso)."""
    c = col.lower()
    return any(p in c for p in ("atraso", "status_atraso", "_real", "delay",
                                "minutos_atraso", "sev_", "qtd_voos", "voos_"))


def eh_feature_voo(f: str) -> bool:
    """True se a feature vem da agregação de voos (recebe fillna(0) no serve, como no treino)."""
    return f.startswith("voos_") or f in COLS_VOOS_BASE


# --------------------------------------------------------------------------------------------
# Temporais (UTC) + codificação cíclica
# --------------------------------------------------------------------------------------------
def add_temporais(base: pd.DataFrame, coluna_dt: str) -> pd.DataFrame:
    """hora/mes/dia_semana (UTC) + sin/cos. ``coluna_dt`` deve ser datetime tz-aware (UTC)."""
    dt = base[coluna_dt].dt
    base["hora"] = dt.hour
    base["mes"] = dt.month
    base["dia_semana"] = dt.dayofweek
    for col, period in [("hora", 24), ("mes", 12), ("dia_semana", 7)]:
        base[f"{col}_sin"] = np.sin(2 * np.pi * base[col] / period)
        base[f"{col}_cos"] = np.cos(2 * np.pi * base[col] / period)
    return base


# --------------------------------------------------------------------------------------------
# Agregação de voos (PÓS-FATO) — flights_anac
# --------------------------------------------------------------------------------------------
def agregar_voos(dfv: pd.DataFrame, window_size: str,
                 top_cias: Sequence[str] = TOP_CIAS,
                 top_origens: Sequence[str] = TOP_ORIGENS) -> pd.DataFrame:
    """Features enriquecidas de voos ATRASADOS por janela (chegada real). sev_* por faixas de
    delay_arr_min (15-30 leve / 30-60 moderado / >60 grave), já que não há STATUS_ATRASO."""
    dfv = dfv[dfv["is_delayed"].astype(bool)].copy()
    dfv["actual_arr"] = pd.to_datetime(dfv["actual_arr"], utc=True, errors="coerce")
    dfv = dfv.dropna(subset=["actual_arr"])
    dfv["time_window"] = dfv["actual_arr"].dt.floor(window_size)
    dfv["delay_arr_min"] = pd.to_numeric(dfv["delay_arr_min"], errors="coerce")

    lt = dfv["route_type"].astype(str).str.strip().str.upper()
    dfv["linha_N"] = (lt == "N").astype(int)
    dfv["linha_I"] = (lt == "I").astype(int)
    cia = dfv["airline_icao"].astype(str).str.upper()
    for c in top_cias:
        dfv[f"cia_{c}"] = (cia == c).astype(int)
    org = dfv["origin_icao"].astype(str).str.upper()
    for o in top_origens:
        dfv[f"orig_{o}"] = (org == o).astype(int)
    dm = dfv["delay_arr_min"]
    dfv["sev_leve"] = ((dm > 15) & (dm <= 30)).astype(int)
    dfv["sev_moderado"] = ((dm > 30) & (dm <= 60)).astype(int)
    dfv["sev_grave"] = (dm > 60).astype(int)

    agg: dict = {
        "qtd_voos_previstos": ("time_window", "size"),
        "qtd_empresas_aereas": ("airline_icao", "nunique"),
        "atraso_medio_min": ("delay_arr_min", "mean"),
        "atraso_max_min": ("delay_arr_min", "max"),
    }
    ind_cols = [c for c in dfv.columns if c.startswith(("linha_", "cia_", "orig_", "sev_"))]
    for c in ind_cols:
        agg[f"voos_{c}"] = (c, "sum")
    return dfv.groupby("time_window").agg(**agg).reset_index()


def perfil_voos_marginal(dfv: pd.DataFrame, calendario: pd.DatetimeIndex, window_size: str,
                         top_cias: Sequence[str] = TOP_CIAS,
                         top_origens: Sequence[str] = TOP_ORIGENS) -> pd.DataFrame:
    """Perfil MARGINAL de voos atrasados por (hora UTC, dia_semana UTC): reindexado sobre TODAS
    as janelas do ``calendario`` (janelas sem atraso entram como 0), reproduzindo a distribuição
    com zeros que o treino viu. Usado pelo narrador para repor features de voo numa data futura.
    Retorna DataFrame indexado por (hora, dia_semana)."""
    voos_win = agregar_voos(dfv, window_size, top_cias, top_origens)
    cols_voos = [c for c in voos_win.columns if c != "time_window"]
    cal = pd.DataFrame({"time_window": pd.Index(calendario)})
    marg = cal.merge(voos_win, on="time_window", how="left")
    for c in cols_voos:
        marg[c] = marg[c].fillna(0.0)
    marg["hora"] = marg["time_window"].dt.hour
    marg["dia_semana"] = marg["time_window"].dt.dayofweek
    return marg.groupby(["hora", "dia_semana"])[cols_voos].mean()


# --------------------------------------------------------------------------------------------
# Prior prospectivo de atraso (climatologia + reputação) — flights_anac (todos os voos)
# --------------------------------------------------------------------------------------------
def construir_prior(dfv: pd.DataFrame, limiar_atraso_min: float = 15.0, pseudo: int = 20,
                    corte_utc: pd.Timestamp | None = None, topk_ctx: int = 3,
                    min_n_ctx: int = 5) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Prior por (dia_semana, hora) UTC a partir de ``scheduled_arr``. Shrinkage empírico-Bayes
    ``(soma + K*alvo)/(n+K)``; reputação do par encolhe para a da CIA. ``corte_utc`` restringe a
    voos de treino (sem vazamento temporal). Retorna (prior_df, fallback, rep_ctx)."""
    d = dfv.copy()
    d["_prev_utc"] = pd.to_datetime(d["scheduled_arr"], utc=True, errors="coerce")
    d["delay_arr_min"] = pd.to_numeric(d["delay_arr_min"], errors="coerce")
    d = d.dropna(subset=["_prev_utc", "delay_arr_min"])
    if corte_utc is not None:
        corte = pd.Timestamp(corte_utc)
        corte = corte.tz_localize("UTC") if corte.tzinfo is None else corte.tz_convert("UTC")
        d = d[d["_prev_utc"] <= corte].copy()
    d["_atrasado"] = (d["delay_arr_min"] > limiar_atraso_min).astype(int)
    d["_atraso_pos"] = d["delay_arr_min"].clip(lower=0)
    d["dia_semana"] = d["_prev_utc"].dt.dayofweek
    d["hora"] = d["_prev_utc"].dt.hour
    d["_data"] = d["_prev_utc"].dt.date
    d["_cia"] = d["airline_icao"].astype(str).str.upper()
    d["_orig"] = d["origin_icao"].astype(str).str.upper()

    taxa_global = float(d["_atrasado"].mean())
    atraso_medio_global = float(d["_atraso_pos"].mean())

    def _shrink(soma, n, alvo):
        return (soma + pseudo * alvo) / (n + pseudo)

    g = d.groupby(["dia_semana", "hora"])
    clim = g.agg(_soma_atr=("_atrasado", "sum"), _soma_min=("_atraso_pos", "sum"),
                 _n_voos=("_atrasado", "size"), _n_datas=("_data", "nunique")).reset_index()
    nv = clim["_n_voos"]
    clim["taxa_atraso_hist"] = _shrink(clim["_soma_atr"], nv, taxa_global)
    clim["atraso_medio_hist"] = _shrink(clim["_soma_min"], nv, atraso_medio_global)
    clim["chegadas_previstas_hist"] = clim["_n_voos"] / clim["_n_datas"].clip(lower=1)

    rc = d.groupby("_cia")["_atrasado"].agg(["sum", "size"])
    rep_cia = _shrink(rc["sum"], rc["size"], taxa_global).to_dict()
    ro = d.groupby("_orig")["_atrasado"].agg(["sum", "size"])
    rep_orig = _shrink(ro["sum"], ro["size"], taxa_global).to_dict()
    rco = d.groupby(["_cia", "_orig"])["_atrasado"].agg(["sum", "size"]).reset_index()
    rco["_alvo"] = rco["_cia"].map(rep_cia).fillna(taxa_global)
    rco["_rep"] = (rco["sum"] + pseudo * rco["_alvo"]) / (rco["size"] + pseudo)
    rep_par = {(c, o): r for c, o, r in zip(rco["_cia"], rco["_orig"], rco["_rep"], strict=False)}

    d["_rep_cia"] = d["_cia"].map(rep_cia).fillna(taxa_global)
    d["_rep_orig"] = d["_orig"].map(rep_orig).fillna(taxa_global)
    d["_rep_par"] = [rep_par.get((c, o), taxa_global)
                     for c, o in zip(d["_cia"], d["_orig"], strict=False)]
    rep_slot = d.groupby(["dia_semana", "hora"]).agg(
        rep_cia_slot=("_rep_cia", "mean"), rep_cia_origem_slot=("_rep_par", "mean"),
        rep_origem_slot=("_rep_orig", "mean")).reset_index()

    prior = (clim[["dia_semana", "hora", "taxa_atraso_hist", "atraso_medio_hist",
                   "chegadas_previstas_hist"]]
             .merge(rep_slot, on=["dia_semana", "hora"], how="left"))
    fallback = {"taxa_atraso_hist": taxa_global, "atraso_medio_hist": atraso_medio_global,
                "chegadas_previstas_hist": 0.0, "rep_cia_slot": taxa_global,
                "rep_cia_origem_slot": taxa_global, "rep_origem_slot": taxa_global}

    grp = d.groupby(["dia_semana", "hora", "_cia", "_orig"]).size().reset_index(name="_n")
    tot = d.groupby(["dia_semana", "hora"]).size().rename("_tot").reset_index()
    grp = grp.merge(tot, on=["dia_semana", "hora"])
    grp["share"] = grp["_n"] / grp["_tot"]
    grp["rep"] = [rep_par.get((c, o), taxa_global)
                  for c, o in zip(grp["_cia"], grp["_orig"], strict=False)]
    grp = grp[grp["_n"] >= min_n_ctx]
    rep_ctx = (grp.sort_values(["dia_semana", "hora", "rep"], ascending=[True, True, False])
               .groupby(["dia_semana", "hora"]).head(topk_ctx)
               [["dia_semana", "hora", "_cia", "_orig", "rep", "share"]]
               .rename(columns={"_cia": "cia", "_orig": "origem"}).reset_index(drop=True))
    return prior, fallback, rep_ctx


def namespace_narrador(clima_num_cols: Sequence[str]) -> set[str]:
    """Superconjunto de nomes que o narrador reconstrói no serve (checagem de skew)."""
    return set(
        list(clima_num_cols)
        + [f"{c}_lag" for c in COLS_LAG_PROD] + [f"diff_{c}" for c in COLS_LAG_PROD]
        + COLS_VOOS_BASE
        + [f"voos_cia_{c}" for c in TOP_CIAS] + [f"voos_orig_{o}" for o in TOP_ORIGENS]
        + ["voos_linha_N", "voos_linha_I", "voos_sev_leve", "voos_sev_moderado",
           "voos_sev_grave"]
        + COLS_CLIMATOLOGIA
        + ["lat", "lng", "dist_aeroporto_km", "hora", "mes", "dia_semana",
           "hora_sin", "hora_cos", "mes_sin", "mes_cos", "dia_semana_sin", "dia_semana_cos"])
