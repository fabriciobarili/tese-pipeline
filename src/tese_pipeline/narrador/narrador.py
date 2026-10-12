"""Narrador natural: conversa motorista × modelos de demanda (porta de ``05_5_Narrador_Natural``).

Responde "vale a pena ir ao SBPA no dia X às Y?" carregando os modelos pareados (clima/atraso)
exportados por ``treina_pareados``, montando features em UTC CONSISTENTES com o treino (reusa
``narrador.features``), explicando via SHAP e narrando com a API da Claude.

Uso típico (notebook ``Work/07_narrador_natural.ipynb``):
    nar = Narrador(anthropic_key=os.environ["ANTHROPIC_API_KEY"])
    nar.consultar("amanhã às 18h", dist_km=8)
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
import shap

from tese_pipeline.narrador import features as fe

LOG = logging.getLogger(__name__)

TZ_LOCAL = ZoneInfo("America/Sao_Paulo")
DIAS_SEMANA = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]

# Variáveis horárias que a Open-Meteo oferece (whitelist p/ repor features sem NaN constante).
OPENMETEO_DISPONIVEIS = {
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "precipitation", "rain", "surface_pressure", "pressure_msl", "cloud_cover",
    "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m", "weather_code",
    "shortwave_radiation", "visibility", "vapour_pressure_deficit",
}
_CIA_NOMES = {"TAM": "LATAM", "AZU": "Azul", "GLO": "Gol", "TAP": "TAP",
              "ARG": "Aerolíneas Argentinas"}
_ORIG_NOMES = {"SBGR": "Guarulhos (SP)", "SBSP": "Congonhas (SP)", "SBKP": "Viracopos (SP)",
               "SBCF": "Confins (MG)", "SBBR": "Brasília (DF)"}

PRECO_LLM = {"input": 1.00, "output": 5.00, "cache_read": 0.10, "cache_write": 1.25}


def _base_clima(feat: str) -> str:
    if feat.startswith("diff_"):
        return feat[len("diff_"):]
    if feat.endswith("_lag"):
        return feat[:-len("_lag")]
    return feat


def _safe_print(s: str) -> None:
    """print robusto: evita UnicodeEncodeError em consoles não-UTF-8 (ex.: cp1252 no Windows)
    quando a narração do LLM traz emojis/caracteres fora da página de código do terminal."""
    try:
        print(s)
    except UnicodeEncodeError:
        enc = (sys.stdout.encoding or "utf-8")
        print(s.encode(enc, "replace").decode(enc))


class Narrador:
    """Ambiente de conversa do motorista com os modelos de demanda (clima/atraso)."""

    def __init__(self, art_dir: str | Path = "Work/narrador_artefatos",
                 flights_parquet: str = "data/raw/flights_anac.parquet",
                 meteo_parquet: str = "data/raw/meteo.parquet",
                 anthropic_key: str | None = None,
                 auth_token: str | None = None,
                 base_url: str | None = None,
                 model_llm: str = "claude-haiku-4-5",
                 limiar_evento: float = 0.5, limiar_baixa: float = 0.45,
                 horas_scan: tuple[int, ...] = (1, 2),
                 dist_default_km: float = 8.0, vel_media_kmh: float = 30.0,
                 peso_esforco: float = 0.02,
                 log_path: str | Path = "reports/logs/narrador_consultas.jsonl",
                 openmeteo_key: str | None = None):
        art = Path(art_dir)
        self.LIMIAR_EVENTO = limiar_evento
        self.LIMIAR_BAIXA = limiar_baixa
        self.HORAS_SCAN = list(horas_scan)
        self.DIST_DEFAULT = dist_default_km
        self.VEL = vel_media_kmh
        self.PESO_ESFORCO = peso_esforco
        self.model_llm = model_llm
        self.anthropic_key = anthropic_key
        self.auth_token = auth_token
        self.base_url = base_url
        self.openmeteo_key = openmeteo_key
        self.log_path = Path(log_path)
        self.OPENMETEO_URL = ("https://customer-api.open-meteo.com/v1/forecast" if openmeteo_key
                              else "https://api.open-meteo.com/v1/forecast")

        self.modelos = {
            "clima": self._carrega_par(art / "modelo_par_clima.joblib",
                                       art / "modelo_par_clima_meta.json"),
            "atraso": self._carrega_par(art / "modelo_par_atraso.joblib",
                                        art / "modelo_par_atraso_meta.json"),
        }
        self.janela = self.modelos["clima"]["meta"]["janela"]
        self.explainers = {n: shap.TreeExplainer(d["model"]) for n, d in self.modelos.items()}

        # Prior de atraso (climatologia + reputação) exportado pelo treino.
        self.climatologia = self._carrega_climatologia(art / "climatologia_atraso_sbpa.csv")
        self.reputacao_slot = self._carrega_reputacao(art / "reputacao_slot_sbpa.csv")
        self.taxa_global = float(self.modelos["atraso"]["meta"].get("taxa_atraso_global", 0.156))
        self.clim_fallback = self.modelos["atraso"]["meta"].get("climatologia_fallback") or {}

        # Dados de apoio (histórico de voos + clima para fallback/perfil).
        self.df_voos = pd.read_parquet(flights_parquet)
        self.df_voos["actual_arr"] = pd.to_datetime(self.df_voos["actual_arr"], utc=True,
                                                     errors="coerce")
        self.df_clima = pd.read_parquet(meteo_parquet)
        self.df_clima["time"] = pd.to_datetime(self.df_clima["time"], utc=True, errors="coerce")
        self.df_clima = self.df_clima.dropna(subset=["time"]).sort_values("time")

        # Climatologia de clima (média hora×mês) p/ fallback quando a API falhar.
        cc = self.df_clima.copy()
        cc["hora"] = cc["time"].dt.hour
        cc["mes"] = cc["time"].dt.month
        self._clima_cols = list(cc.select_dtypes("number").columns.drop(["hora", "mes"],
                                                                        errors="ignore"))
        self._clima_hm = cc.groupby(["hora", "mes"])[self._clima_cols].mean()

        # Variáveis climáticas efetivamente necessárias pelos modelos ∩ Open-Meteo.
        self.vars_clima = sorted({
            _base_clima(f) for d in self.modelos.values() for f in d["features"]
            if _base_clima(f) in OPENMETEO_DISPONIVEIS})
        self._clima_cache: dict[pd.Timestamp, dict] = {}
        LOG.info("Narrador pronto | janela=%s | vars_clima=%s", self.janela, self.vars_clima)

    # ---- carga ----
    @staticmethod
    def _carrega_par(model_path: Path, meta_path: Path) -> dict:
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        return {"model": joblib.load(model_path), "features": meta["features_lgbm"], "meta": meta}

    @staticmethod
    def _carrega_climatologia(path: Path) -> pd.DataFrame:
        if not Path(path).exists():
            return pd.DataFrame().set_index(pd.MultiIndex.from_arrays([[], []],
                                            names=["dia_semana", "hora"]))
        c = pd.read_csv(path)
        return c.set_index(["dia_semana", "hora"])

    @staticmethod
    def _carrega_reputacao(path: Path) -> dict:
        rep = {}
        if Path(path).exists():
            r = pd.read_csv(path)
            for (ds, hr), g in r.groupby(["dia_semana", "hora"]):
                rep[(int(ds), int(hr))] = g[["cia", "origem", "rep", "share"]].to_dict("records")
        return rep

    # ---- clima (Open-Meteo + fallback) ----
    def _buscar_openmeteo(self, dt_ini, dt_fim) -> bool:
        params = {"latitude": fe.AER_LAT, "longitude": fe.AER_LNG,
                  "hourly": ",".join(self.vars_clima), "timezone": "UTC",
                  "start_date": pd.Timestamp(dt_ini).date().isoformat(),
                  "end_date": pd.Timestamp(dt_fim).date().isoformat(),
                  "wind_speed_unit": "kmh", "temperature_unit": "celsius"}
        if self.openmeteo_key:
            params["apikey"] = self.openmeteo_key
        try:
            r = requests.get(self.OPENMETEO_URL, params=params, timeout=30)
            r.raise_for_status()
            h = r.json()["hourly"]
            tempos = pd.to_datetime(h["time"], utc=True)
            for i, t in enumerate(tempos):
                tw = pd.Timestamp(t).floor("h")
                self._clima_cache[tw] = {v: (float(h[v][i]) if h.get(v) and h[v][i] is not None
                                             else np.nan) for v in self.vars_clima}
            return True
        except Exception as e:  # noqa: BLE001
            LOG.warning("Open-Meteo indisponível (%s); usando fallback histórico/climatologia.", e)
            return False

    def _clima_na_hora(self, ts_utc) -> dict:
        tw = pd.Timestamp(ts_utc).floor("h")
        if tw not in self._clima_cache:
            self._buscar_openmeteo(tw - pd.Timedelta("1h"),
                                   tw + pd.Timedelta(f"{max(self.HORAS_SCAN)}h"))
        if tw in self._clima_cache:
            return self._clima_cache[tw]
        linha = self.df_clima[self.df_clima["time"] == tw]   # fallback 1: histórico exato
        if not linha.empty:
            return {v: float(linha[v].values[0]) for v in self.vars_clima if v in linha.columns}
        chave = (tw.hour, tw.month)                           # fallback 2: climatologia
        if chave in self._clima_hm.index:
            row = self._clima_hm.loc[chave]
            return {v: float(row[v]) for v in self.vars_clima if v in self._clima_hm.columns}
        return {}

    # ---- features de uma janela (UTC) ----
    def _para_utc(self, dt_local) -> pd.Timestamp:
        t = pd.Timestamp(dt_local)
        t = t.tz_localize(TZ_LOCAL) if t.tzinfo is None else t.tz_convert(TZ_LOCAL)
        return t.tz_convert("UTC")

    def _esforco(self, dist_km: float) -> dict:
        km = float(dist_km)
        return {"km": km, "minutos": km / self.VEL * 60.0,
                "tier": "perto" if km <= 5 else ("médio" if km <= 15 else "longe"),
                "limiar_ajustado": min(self.LIMIAR_EVENTO + self.PESO_ESFORCO * km, 0.75)}

    def construir_features(self, dt_local, dist_km=None):
        t_utc = self._para_utc(dt_local).floor("h")
        clima_atual = self._clima_na_hora(t_utc)
        clima_lag = self._clima_na_hora(t_utc - pd.Timedelta("1h"))
        row = {}
        for v in self.vars_clima:
            row[v] = clima_atual.get(v, np.nan)
            row[f"{v}_lag"] = clima_lag.get(v, np.nan)
            row[f"diff_{v}"] = (row[v] - row[f"{v}_lag"]
                                if pd.notna(row[v]) and pd.notna(row[f"{v}_lag"]) else np.nan)
        # temporal (UTC) + cíclicas
        tmp = pd.DataFrame([{"request_dt": t_utc}])
        fe.add_temporais(tmp, "request_dt")
        for c in tmp.columns:
            if c != "request_dt":
                row[c] = tmp[c].iloc[0]
        # climatologia prospectiva (dia_semana, hora)
        chave = (t_utc.dayofweek, t_utc.hour)
        for c in fe.COLS_CLIMATOLOGIA:
            if chave in self.climatologia.index and c in self.climatologia.columns:
                row[c] = float(self.climatologia.loc[chave, c])
            else:
                _default = (self.taxa_global
                            if (c == "taxa_atraso_hist" or c.startswith("rep_")) else 0.0)
                row[c] = float(self.clim_fallback.get(c, _default))
        dist = self.DIST_DEFAULT if dist_km is None else float(dist_km)
        ctx = {"t_utc": t_utc, "dt_local": self._para_utc(dt_local).tz_convert(TZ_LOCAL),
               "clima_disponivel": bool(clima_atual), "dist_km": dist,
               "esforco": self._esforco(dist)}
        return pd.DataFrame([row]), ctx

    def _preparar_X(self, df_full, feats):
        d = df_full.copy()
        for f in feats:
            if f not in d.columns:
                d[f] = 0.0 if fe.eh_feature_voo(f) else np.nan
        X = d[feats].copy()
        for f in feats:
            if fe.eh_feature_voo(f):
                X[f] = X[f].fillna(0.0)
        return X

    def _top_shap(self, nome, X, top_n):
        sv = self.explainers[nome].shap_values(X)
        if isinstance(sv, list):
            vals = sv[1][0]
        elif getattr(sv, "ndim", 2) == 3:
            vals = sv[0, :, 1]
        else:
            vals = sv[0]
        feats = self.modelos[nome]["features"]
        fv = X.values[0]
        ordem = np.argsort(np.abs(vals))[::-1][:top_n]
        return [{"feature": feats[i], "valor": float(fv[i]) if pd.notna(fv[i]) else None,
                 "shap": float(vals[i]), "label": self._label_feature(feats[i], fv[i])}
                for i in ordem]

    def predizer(self, dt_local, dist_km=None, top_n=6):
        df_full, ctx = self.construir_features(dt_local, dist_km=dist_km)
        dims = {}
        for nome, d in self.modelos.items():
            X = self._preparar_X(df_full, d["features"])
            prob = float(d["model"].predict_proba(X)[0][1])
            dims[nome] = {"prob": prob, "top_shap": self._top_shap(nome, X, top_n)}
        p_clima, p_atraso = dims["clima"]["prob"], dims["atraso"]["prob"]
        tem = (p_clima >= self.LIMIAR_EVENTO) or (p_atraso >= self.LIMIAR_EVENTO)
        return {"tem_demanda": tem,
                "dimensao_dominante": (None if not tem else
                                       ("clima" if p_clima >= p_atraso else "atraso")),
                "prob_clima": p_clima, "prob_atraso": p_atraso,
                "clima": dims["clima"], "atraso": dims["atraso"], "contexto": ctx}

    # ---- rótulos legíveis ----
    @staticmethod
    def _fmt(v):
        return "ausente" if (v is None or (isinstance(v, float) and np.isnan(v))) else v

    def _label_feature(self, feat, valor):
        v = self._fmt(valor)
        if isinstance(v, str):
            return f"{feat} ({v})"
        mapa = {
            "temperature_2m": f"temperatura ({v:.1f}°C)",
            "relative_humidity_2m": f"umidade relativa ({v:.0f}%)",
            "wind_speed_10m": f"velocidade do vento ({v:.1f} km/h)",
            "precipitation": f"precipitação ({v:.1f} mm)",
            "weather_code": f"código de tempo ({int(v)})",
            "taxa_atraso_hist": f"taxa histórica de atraso na chegada ({v:.0%})",
            "atraso_medio_hist": f"atraso médio histórico ({v:.0f} min)",
            "chegadas_previstas_hist": f"chegadas típicas no horário ({v:.1f})",
            "rep_cia_slot": f"reputação média das companhias do horário ({v:.0%})",
            "rep_cia_origem_slot": f"reputação média dos pares CIA/origem ({v:.0%})",
            "rep_origem_slot": f"reputação média das origens do horário ({v:.0%})",
            "hora": f"hora do dia UTC ({int(v)}h)", "mes": f"mês ({int(v)})",
            "dia_semana": f"dia da semana UTC ({DIAS_SEMANA[int(v) % 7]})",
        }
        if feat in mapa:
            return mapa[feat]
        if feat.startswith("diff_"):
            return f"variação de {feat[len('diff_'):]} ({v:+.2f})"
        if feat.endswith("_lag"):
            return f"{feat[:-4]} na hora anterior ({v:.2f})"
        if feat.endswith(("_sin", "_cos")):
            return f"{feat} (componente cíclica)"
        return f"{feat} ({v:.3f})"

    def _detalhe_voos_historicos(self, t_utc):
        mask = ((self.df_voos["actual_arr"].dt.hour == t_utc.hour) &
                (self.df_voos["actual_arr"].dt.dayofweek == t_utc.dayofweek) &
                (self.df_voos["is_delayed"].astype("boolean").fillna(False)))
        slot = self.df_voos[mask].dropna(subset=["actual_arr"])
        if slot.empty:
            return None
        n_reg = int(len(slot))
        n_dias = int(slot["actual_arr"].dt.normalize().nunique())
        partes, conc_cia, conc_orig = [], 0.0, 0.0
        if "airline_icao" in slot.columns:
            vc = slot["airline_icao"].astype(str).str.upper().value_counts()
            conc_cia = float(vc.iloc[0] / n_reg) if not vc.empty else 0.0
            partes.append("CIA: " + " / ".join(_CIA_NOMES.get(c, c) for c in vc.head(2).index))
        if "origin_icao" in slot.columns:
            vo = slot["origin_icao"].astype(str).str.upper().value_counts()
            conc_orig = float(vo.iloc[0] / n_reg) if not vo.empty else 0.0
            partes.append("Chegando de: " + " / ".join(_ORIG_NOMES.get(o, o)
                                                        for o in vo.head(2).index))
        partes.append(f"Histórico: {n_reg} chegadas atrasadas em {n_dias} dia(s) distinto(s)")
        conc = max(conc_cia, conc_orig)
        nivel = ("forte" if (n_dias >= 3 and conc >= 0.40)
                 else "plausivel" if (conc >= 0.50 and n_dias >= 1) else "esporadico")
        return {"texto": " | ".join(partes), "nivel": nivel, "n_dias": n_dias,
                "n_registros": n_reg, "conc": conc}

    def _contexto_reputacao(self, t_utc, max_itens=3):
        itens = self.reputacao_slot.get((t_utc.dayofweek, t_utc.hour))
        if not itens:
            return None
        return " | ".join(
            f"{_CIA_NOMES.get(it['cia'], it['cia'])} vindo de "
            f"{_ORIG_NOMES.get(it['origem'], it['origem'])} ({it['rep']:.0%} de atraso histórico, "
            f"{it['share']:.0%} dos voos do horário)" for it in itens[:max_itens])

    def ficha_tecnica(self, res, dimensao=None, top_n=6):
        dim = (dimensao or res["dimensao_dominante"]
               or ("clima" if res["prob_clima"] >= res["prob_atraso"] else "atraso"))
        rot = {"clima": "EVENTO CLIMÁTICO", "atraso": "ATRASO DE VOO"}[dim]
        linhas = [f"  Ficha técnica — dimensão {rot} (prob {res[f'prob_{dim}']:.1%}):"]
        for t in res[dim]["top_shap"][:top_n]:
            efeito = "AUMENTA" if t["shap"] > 0 else "reduz"
            linhas.append(f"    • {t['label']:<46} {efeito} a chance  [SHAP {t['shap']:+.3f}]")
        if dim == "atraso":
            det = self._detalhe_voos_historicos(res["contexto"]["t_utc"])
            if det:
                rotn = {"forte": "RECORRENTE (forte)", "plausivel": "PLAUSÍVEL (concentrado)",
                        "esporadico": "esporádico (fraco)"}[det["nivel"]]
                linhas.append(f"    Padrão de atraso: {rotn} — {det['n_dias']} dia(s), "
                              f"concentração até {det['conc']:.0%}.")
            rep = self._contexto_reputacao(res["contexto"]["t_utc"])
            if rep:
                linhas.append(f"    Piores históricos neste horário: {rep}")
        esf = res["contexto"].get("esforco")
        if esf:
            linhas.append(f"    Esforço: ~{esf['km']:.0f} km / ~{esf['minutos']:.0f} min "
                          f"({esf['tier']}) — barra ajustada p/ {esf['limiar_ajustado']:.0%}.")
        texto = "\n".join(linhas)
        return dim, texto

    def varrer_para_frente(self, dt_local, dist_km=None):
        for h in self.HORAS_SCAN:
            alvo = pd.Timestamp(dt_local) + pd.Timedelta(hours=h)
            r = self.predizer(alvo, dist_km=dist_km)
            if r["tem_demanda"]:
                return {"encontrou": True, "horas_a_frente": h, "dt_local": alvo, "resultado": r}
        return {"encontrou": False, "resultado": None}

    # ---- narração via Claude ----
    def _nivel_prob(self, p):
        if p < self.LIMIAR_BAIXA:
            return "BAIXA"
        if p < self.LIMIAR_EVENTO:
            return "MODERADA"
        return "ALTA" if p < 0.65 else "MUITO ALTA"

    @property
    def system_prompt(self):
        return (
            "Você é um assistente de demanda de corridas para motoristas de aplicativo em Porto "
            "Alegre, focado no Aeroporto Salgado Filho (SBPA). Explique a previsão de forma clara, "
            "direta e prática, em português, em 3 a 5 frases curtas. NÃO use jargão técnico (nada "
            "de 'SHAP', 'modelo', 'probabilidade' cru, 'feature').\n"
            "ENQUADRAMENTO: o motorista busca passageiros que CHEGAM (desembarcam) de voos "
            "atrasados — nunca fale em voos que 'saem' atrasados.\n"
            "SEJA CONSERVADOR, mas não pessimista. Chance abaixo de 45% é BAIXA e não justifica ir "
            "sozinha. Recomende IR quando houver sinal forte (>= 50%) E um padrão de atraso "
            "recorrente ou plausível (concentrado numa CIA/origem). Pondere o ESFORÇO (distância): "
            "quanto mais longe, mais forte o sinal precisa ser. Termine com uma recomendação "
            "objetiva (ir ou não).")

    def _montar_prompt(self, res, cenario):
        ctx = res["contexto"]
        quando = ctx["dt_local"].strftime("%d/%m/%Y às %Hh")
        dia = DIAS_SEMANA[ctx["dt_local"].dayofweek]
        det = self._detalhe_voos_historicos(ctx["t_utc"])
        if det is None:
            bloco_voos = ("\nContexto histórico de voos: sem registros de chegadas atrasadas "
                          "nesse horário/dia.")
            recor = "PADRÃO do atraso: sem histórico — sinal FRACO."
        else:
            bloco_voos = ("\nContexto histórico de chegadas atrasadas nesse horário/dia "
                          "(passageiros desembarcando):\n  " + det["texto"])
            if det["nivel"] == "forte":
                recor = (f"PADRÃO: RECORRENTE ({det['n_dias']} dias, "
                         f"concentração {det['conc']:.0%}) "
                         "— sinal FORTE, pode justificar ir.")
            elif det["nivel"] == "plausivel":
                recor = (f"PADRÃO: PLAUSÍVEL (concentração {det['conc']:.0%} numa CIA/origem, "
                         f"{det['n_dias']} dia(s)) — considerar, não descartar.")
            else:
                recor = (f"PADRÃO: esporádico/fraco ({det['n_dias']} dia(s), {det['conc']:.0%}). "
                         "Não recomende ir só por isso.")
        rep = self._contexto_reputacao(ctx["t_utc"])
        bloco_rep = ("" if not rep else
                     "\nReputação histórica (quem mais atrasa neste horário): " + rep)
        esf = ctx.get("esforco") or {}
        bloco_esf = (f"\nESFORÇO: ~{esf.get('km', 0):.0f} km (~{esf.get('minutos', 0):.0f} min, "
                     f"{esf.get('tier', '?')}); "
                     f"use a barra ~{esf.get('limiar_ajustado', 0.5):.0%}.")

        def _fatores(top):
            return "\n".join(f"  - {t['label']} — "
                             f"{'puxa a favor' if t['shap'] > 0 else 'puxa contra'}" for t in top)

        return (
            f"Consulta do motorista para {quando} ({dia}).\n\n"
            f"Duas dimensões (cada uma contra 'sem pico'):\n"
            f"  - Chance por EVENTO CLIMÁTICO: {res['prob_clima']:.1%} "
            f"(nível {self._nivel_prob(res['prob_clima'])})\n"
            f"  - Chance por ATRASO DE VOO na chegada: {res['prob_atraso']:.1%} "
            f"(nível {self._nivel_prob(res['prob_atraso'])})\n"
            f"Lembrete: níveis BAIXA (<45%) não justificam ir por si sós.\n\n{recor}\n\n"
            f"Fatores climáticos (com valores):\n{_fatores(res['clima']['top_shap'])}\n\n"
            f"Fatores de atraso de voo (com valores):\n{_fatores(res['atraso']['top_shap'])}\n"
            f"{bloco_voos}{bloco_rep}{bloco_esf}\n\nSituação: {cenario}\n\n"
            "Explique ao motorista, citando valores concretos, se vale a pena ir e por quê. "
            "Siga as regras conservadoras; termine com recomendação objetiva.")

    def _tem_llm(self) -> bool:
        return bool(self.anthropic_key or self.auth_token)

    def _client(self):
        """Cria o cliente Anthropic. Suporta API key (x-api-key), token Bearer (auth_token) e
        base_url customizada (gateway). base_url/auth_token None caem no padrão do SDK."""
        import anthropic
        kw = {}
        if self.base_url:
            kw["base_url"] = self.base_url
        if self.auth_token:
            kw["auth_token"] = self.auth_token
        elif self.anthropic_key:
            kw["api_key"] = self.anthropic_key
            # Com API key e sem base_url explícito, fixa o endpoint público para não ser
            # sequestrado por um ANTHROPIC_BASE_URL de ambiente (ex.: proxy local do Claude Code).
            kw.setdefault("base_url", "https://api.anthropic.com")
        return anthropic.Anthropic(**kw)

    def narrar(self, res, cenario):
        if not self._tem_llm():
            raise RuntimeError("sem credencial de LLM — passe anthropic_key OU auth_token "
                               "(+ base_url do gateway) ao Narrador.")
        client = self._client()
        prompt = self._montar_prompt(res, cenario)
        t0 = time.perf_counter()
        resp = client.messages.create(model=self.model_llm, max_tokens=400,
                                       system=self.system_prompt,
                                       messages=[{"role": "user", "content": prompt}])
        texto = resp.content[0].text
        self._log_consulta(res, cenario, prompt, texto, resp, round(time.perf_counter() - t0, 3))
        return texto

    def _log_consulta(self, res, cenario, prompt, texto, resp, tempo):
        try:
            u = resp.usage
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "horario": datetime.datetime.now(TZ_LOCAL).isoformat(),
                    "tempo_resposta_s": tempo, "modelo": self.model_llm, "cenario": cenario,
                    "prob_clima": res["prob_clima"], "prob_atraso": res["prob_atraso"],
                    "prompt": prompt, "resposta": texto,
                    "tokens": {"input": getattr(u, "input_tokens", None),
                               "output": getattr(u, "output_tokens", None)},
                }, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            LOG.warning("log da consulta falhou: %s", e)

    def _extrair_datetime_local(self, pergunta):
        if not self._tem_llm():
            raise RuntimeError("credencial de LLM necessária p/ interpretar texto; "
                               "passe um datetime direto.")
        client = self._client()
        hoje = datetime.datetime.now(TZ_LOCAL).strftime("%d/%m/%Y")
        resp = client.messages.create(
            model=self.model_llm, max_tokens=80,
            system=(f"Hoje é {hoje} (Porto Alegre). Extraia data e hora da pergunta e responda "
                    'APENAS JSON {"data":"DD/MM/YYYY","hora":HH}. Sem data/hora, use a próxima '
                    "hora cheia."),
            messages=[{"role": "user", "content": pergunta}])
        try:
            m = re.search(r"\{.*\}", resp.content[0].text, re.DOTALL)
            dados = json.loads(m.group(0))
            return datetime.datetime.strptime(f"{dados['data']} {int(dados['hora']):02d}:00",
                                              "%d/%m/%Y %H:%M")
        except Exception:
            return (datetime.datetime.now(TZ_LOCAL).replace(minute=0, second=0, microsecond=0,
                    tzinfo=None) + datetime.timedelta(hours=1))

    def consultar(self, pergunta_ou_dt, dist_km=None, imprimir=True):
        if isinstance(pergunta_ou_dt, str):
            dt_local = self._extrair_datetime_local(pergunta_ou_dt)
            pergunta = pergunta_ou_dt
        else:
            dt_local = pergunta_ou_dt
            pergunta = f"consulta para {pd.Timestamp(dt_local):%d/%m/%Y %Hh}"
        res = self.predizer(dt_local, dist_km=dist_km)
        quando = res["contexto"]["dt_local"].strftime("%d/%m/%Y às %Hh")

        if res["tem_demanda"]:
            cen = (f"HÁ demanda esperada em {quando}; dimensão predominante: "
                   f"{'clima' if res['dimensao_dominante'] == 'clima' else 'atraso de voo'}.")
            alvo, veredito, scan = res, "SIM", None
        else:
            scan = self.varrer_para_frente(dt_local, dist_km=dist_km)
            if scan["encontrou"]:
                qs = scan["resultado"]["contexto"]["dt_local"].strftime("%d/%m/%Y às %Hh")
                cen = (f"NÃO há demanda em {quando}, MAS há em {qs} (+{scan['horas_a_frente']}h). "
                       "Oriente o motorista a considerar esse horário.")
                alvo, veredito = scan["resultado"], "SIM (+adiante)"
            else:
                cen = (f"NÃO há demanda em {quando} nem nas próximas {max(self.HORAS_SCAN)}h. "
                       "Recomende não se deslocar por ora.")
                alvo, veredito = res, "NÃO"
        narrativa = self.narrar(alvo, cen) if self._tem_llm() else None
        saida = {"pergunta": pergunta, "dt_local": dt_local, "veredito": veredito,
                 "resultado": res, "narrativa": narrativa, "scan": scan}
        if imprimir:
            _safe_print(f"Motorista : {pergunta}")
            _safe_print(f"Horário   : {quando}  (UTC {res['contexto']['t_utc']:%d/%m %Hh})")
            _safe_print(f"Veredito  : {veredito}  (clima {res['prob_clima']:.1%} | "
                        f"atraso {res['prob_atraso']:.1%})\n")
            _, ficha = self.ficha_tecnica(alvo)
            _safe_print(ficha)
            if narrativa:
                _safe_print("\n--- Resposta ao motorista ---\n" + narrativa + "\n")
            else:
                _safe_print("\n(narração via LLM desativada: sem credencial)\n")
        return saida
