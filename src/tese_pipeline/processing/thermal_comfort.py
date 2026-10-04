"""Estágio: Conforto térmico — calcula índices com pythermalcomfort 4.x.

Referência da biblioteca:
  Tartarini & Schiavon (2020). "pythermalcomfort: A Python package for thermal
  comfort calculations." SoftwareX, 12, 100578.
  https://pythermalcomfort.readthedocs.io/en/latest/

Índices calculados por linha horária e por ponto da RM POA:

  UTCI   — Universal Thermal Climate Index (ISO 15743; melhor para ambientes externos)
             Entrada: tdb, tr (Tmrt estimada), v (m/s), rh
             Saída: valor °C + categoria de estresse (cold/no_stress/heat)

  AT     — Apparent Temperature, Steadman (1994)
             Entrada: tdb, rh, v (m/s)
             Relevante para percepção popular de conforto/desconforto

  THI    — Temperature Humidity Index (Thom, 1959)
             Entrada: tdb, rh
             Usado em estudos de conforto animal/humano em regiões subtropicais

  DI     — Discomfort Index (Thom, 1959)
             Entrada: tdb, rh
             Escala direta de desconforto

  HI     — Heat Index (Rothfusz, 1990)
             Aplicado apenas quando tdb >= 27 °C e rh >= 40%

  WCI    — Wind Chill Index
             Aplicado apenas quando tdb < 10 °C

Estimativa de Tmrt (temperatura radiante média):
  Método simplificado para ambiente externo sem dados de radiação difusa/direta
  separados (ISO 7933 / VDI 3787):
    - Dia   (srad > 0):   Tmrt = tdb + 0.7 * ln(srad + 1) - 2
    - Noite (srad == 0):  Tmrt = tdb - 2   (emissão do céu)
  Onde srad é shortwave_radiation (W/m²) da Open-Meteo ERA5.
  Para maior precisão com dados de radiação direta/difusa completos, usar
  pythermalcomfort.models.solar_gain().

Saída: data/processed/thermal_comfort.parquet
  Colunas: location, time, tdb, tr, v_ms, rh, utci, utci_stress,
           at, thi, di, hi (NaN se tdb < 27), wci (NaN se tdb >= 10),
           h3_r8 (célula H3 do ponto), lat, lng
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import h3
import numpy as np
import pandas as pd
from pythermalcomfort.models import (
    at,
    discomfort_index,
    heat_index_rothfusz,
    thi,
    utci,
    wci,
)

from tese_pipeline.config import load_config, set_global_seed
from tese_pipeline.logging_conf import configure_logging
from tese_pipeline.provenance import write_provenance

LOG = logging.getLogger(__name__)

# Categorias UTCI (EN ISO 15743 / Bröde et al. 2012)
UTCI_BINS   = [-np.inf, -40, -27, -13, 0, 9, 26, 32, 38, 46, np.inf]
UTCI_LABELS = [
    "extreme_cold", "very_strong_cold", "strong_cold", "moderate_cold",
    "slight_cold", "no_stress",
    "moderate_heat", "strong_heat", "very_strong_heat", "extreme_heat",
]


def estimate_tmrt(tdb: np.ndarray, srad: np.ndarray) -> np.ndarray:
    """Estima temperatura radiante média (Tmrt) a partir da radiação solar global.

    Aproximação para ambiente externo (VDI 3787 / Jendritzky et al.):
      - Dia  (srad > 0): Tmrt = tdb + 0.7 * ln(srad + 1) - 2
      - Noite (srad = 0): Tmrt = tdb - 2  (perda radiativa para o céu)
    """
    srad = np.maximum(srad, 0.0)
    tmrt = np.where(srad > 0, tdb + 0.7 * np.log(srad + 1) - 2.0, tdb - 2.0)
    return tmrt.round(2)


def compute_utci_safe(tdb: float, tr: float, v: float, rh: float) -> float:
    """Calcula UTCI com tratamento de valores fora do domínio (retorna NaN)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = utci(tdb=tdb, tr=tr, v=v, rh=rh, limit_inputs=False, round_output=True)
        val = result.utci if hasattr(result, "utci") else float(result)
        return float(val)
    except Exception:
        return float("nan")


def compute_at_safe(tdb: float, rh: float, v: float) -> float:
    try:
        result = at(tdb=tdb, rh=rh, v=v, round_output=True)
        return float(result.at if hasattr(result, "at") else result)
    except Exception:
        return float("nan")


def compute_thi_safe(tdb: float, rh: float) -> float:
    try:
        result = thi(tdb=tdb, rh=rh, round_output=True)
        return float(result.thi if hasattr(result, "thi") else result)
    except Exception:
        return float("nan")


def compute_di_safe(tdb: float, rh: float) -> float:
    try:
        result = discomfort_index(tdb=tdb, rh=rh)
        return float(result.di if hasattr(result, "di") else result)
    except Exception:
        return float("nan")


def process_location(df_loc: pd.DataFrame) -> pd.DataFrame:
    """Aplica todos os índices de conforto térmico a um DataFrame de uma localidade."""
    df = df_loc.copy()
    tdb_arr  = df["temperature_2m"].to_numpy(dtype=float)
    rh_arr   = df["relative_humidity_2m"].to_numpy(dtype=float)
    v_kmh    = df["wind_speed_10m"].fillna(0).to_numpy(dtype=float)
    v_ms_arr = (v_kmh / 3.6).round(3)          # km/h → m/s
    srad_arr = df.get("shortwave_radiation", pd.Series(0, index=df.index)).fillna(0).to_numpy(dtype=float)

    tmrt_arr = estimate_tmrt(tdb_arr, srad_arr)

    # UTCI (vetorizado linha a linha por simplicidade; ~8760 linhas/ano/local = rápido)
    utci_vals = np.array([
        compute_utci_safe(t, tr, v, r)
        for t, tr, v, r in zip(tdb_arr, tmrt_arr, v_ms_arr, rh_arr)
    ])

    # AT (Apparent Temperature)
    at_vals = np.array([
        compute_at_safe(t, r, v)
        for t, r, v in zip(tdb_arr, rh_arr, v_ms_arr)
    ])

    # THI & DI
    thi_vals = np.array([compute_thi_safe(t, r) for t, r in zip(tdb_arr, rh_arr)])
    di_vals  = np.array([compute_di_safe(t, r)  for t, r in zip(tdb_arr, rh_arr)])

    # Heat Index — só válido para tdb >= 27°C e rh >= 40%
    hi_vals = np.full(len(df), np.nan)
    mask_hi = (tdb_arr >= 27.0) & (rh_arr >= 40.0)
    for i in np.where(mask_hi)[0]:
        try:
            r = heat_index_rothfusz(tdb=tdb_arr[i], rh=rh_arr[i], round_output=True, limit_inputs=False)
            hi_vals[i] = float(r.hi if hasattr(r, "hi") else r)
        except Exception:
            pass

    # Wind Chill — só válido para tdb < 10°C
    wci_vals = np.full(len(df), np.nan)
    mask_wc = tdb_arr < 10.0
    for i in np.where(mask_wc)[0]:
        try:
            r = wci(tdb=tdb_arr[i], v=v_ms_arr[i], round_output=True)
            wci_vals[i] = float(r.wci if hasattr(r, "wci") else r)
        except Exception:
            pass

    # categoria UTCI
    utci_stress = pd.cut(utci_vals, bins=UTCI_BINS, labels=UTCI_LABELS, right=True).astype(str)

    df["tr"]          = tmrt_arr
    df["v_ms"]        = v_ms_arr
    df["utci"]        = utci_vals.round(2)
    df["utci_stress"] = utci_stress
    df["at"]          = at_vals
    df["thi"]         = thi_vals
    df["di"]          = di_vals
    df["hi"]          = hi_vals
    df["wci"]         = wci_vals

    return df


def main() -> None:
    configure_logging(log_file="reports/logs/thermal_comfort.log")
    params = load_config("params.yaml")
    set_global_seed(params["seed"])
    cfg_rm = load_config("config/meteo_rm.yaml")

    meteo_rm = pd.read_parquet("data/raw/meteo_rm.parquet")
    LOG.info("meteo_rm carregado: %d linhas, %d localidades",
             len(meteo_rm), meteo_rm["location"].nunique())

    # H3 resolution para join com features de corridas
    h3_res = params["geospatial"]["h3_resolution"]

    # mapa localidade → H3 + coordenadas
    loc_map = {l["name"]: l for l in cfg_rm["locations"]}

    frames = []
    for loc_name, df_loc in meteo_rm.groupby("location"):
        LOG.info("Processando %s (%d horas)...", loc_name, len(df_loc))
        df_out = process_location(df_loc)

        # adiciona H3 do ponto
        coords = loc_map.get(loc_name)
        if coords:
            h3_cell = h3.latlng_to_cell(coords["latitude"], coords["longitude"], h3_res)
            df_out["h3_r8"] = h3_cell
            df_out["lat"]   = coords["latitude"]
            df_out["lng"]   = coords["longitude"]
        else:
            df_out["h3_r8"] = None
            df_out["lat"]   = df_out.get("latitude", None)
            df_out["lng"]   = df_out.get("longitude", None)

        frames.append(df_out)

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["location", "time"]).reset_index(drop=True)

    # estatísticas de desconforto
    total = len(result)
    for cat in ["strong_heat", "very_strong_heat", "extreme_heat",
                "strong_cold", "very_strong_cold", "extreme_cold"]:
        n = (result["utci_stress"] == cat).sum()
        if n:
            LOG.info("UTCI %s: %d horas (%.1f%%)", cat, n, n/total*100)

    # colunas de saída
    keep_cols = [
        "location", "time", "lat", "lng", "h3_r8",
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation",
        "shortwave_radiation", "weather_code",
        "tr", "v_ms", "utci", "utci_stress", "at", "thi", "di", "hi", "wci",
    ]
    keep_cols = [c for c in keep_cols if c in result.columns]
    result = result[keep_cols]

    out = Path("data/processed/thermal_comfort.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)

    write_provenance(
        out,
        source="pythermalcomfort:utci+at+thi+di+hi+wci",
        params={
            "thermal": {
                "library": "pythermalcomfort",
                "indices": ["utci", "at", "thi", "di", "hi", "wci"],
                "tmrt_method": "simplified_srad_log",
                "utci_stress_bins": dict(zip(UTCI_LABELS, UTCI_BINS[1:])),
                "h3_resolution": h3_res,
                "n_locations": result["location"].nunique(),
                "n_rows": len(result),
            }
        },
    )
    LOG.info(
        "thermal_comfort salvo: %d linhas | %d localidades | índices: utci, at, thi, di, hi, wci",
        len(result), result["location"].nunique(),
    )


if __name__ == "__main__":
    main()
