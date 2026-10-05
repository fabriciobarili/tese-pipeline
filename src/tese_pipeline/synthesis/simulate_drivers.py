"""Estágio 4 (alternativo) — Simulação baseada em agentes (motoristas).

Adapta as premissas de ``02_GERACAO_DE_DADOS/distribuicao_nao_simulada_v6_final_4.py``
(simulação do Colab) para a pipeline da tese. Em vez de gerar corridas isoladas
por chegada de voo (ver ``generate_rides.py``), aqui modelamos **motoristas** que
percorrem a malha viária de Porto Alegre e recebem chamados cuja demanda é
modulada por **eventos**: voos atrasados no SBPA (ANAC VRA) e estresse térmico
UTCI (pythermalcomfort). O resultado aproxima o dado "contribuído pelo motorista":
origem (lat/lng), datetime e booleano ``com_passageiro``.

Premissas e parâmetros vivem em ``config/driver_simulation.yaml``.

Fontes de dados:
  - ``ext/sul-261003_vias_h3r13.parquet``   malha viária OSM em H3
  - ``data/raw/flights_anac.parquet``       chegadas SBPA (coluna is_delayed)
  - ``data/processed/thermal_comfort.parquet`` categorias UTCI horárias (SBPA)
  - ``ext/Aeroporto Salgado Filho.json``    células H3 do aeroporto

Saída: ``data/synthetic/driver_trips.parquet``.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import h3
import numpy as np
import pandas as pd

from tese_pipeline.config import load_config, set_global_seed
from tese_pipeline.logging_conf import configure_logging
from tese_pipeline.provenance import write_provenance

LOG = logging.getLogger(__name__)


# ===========================================================================
# Modelos de tempo / probabilidade de chamado
# ===========================================================================
class TimeModel:
    """Converte distâncias em saltos H3 / km para minutos de deslocamento.

    Usa duas velocidades: urbana (trajetos curtos) e interurbana (longos),
    separadas por ``interurban_threshold_km``.
    """

    def __init__(self, mob: dict):
        self.km_per_cell = float(mob["km_per_h3_cell"])
        self.urban_speed = float(mob["urban_speed_kmh"])
        self.interurban_speed = float(mob["interurban_speed_kmh"])
        self.threshold_km = float(mob["interurban_threshold_km"])
        self.speed_sd = float(mob["speed_sd"])
        self.idle_mean = float(mob["idle_mean_minutes"])
        self.idle_max = float(mob["idle_max_minutes"])

    def _speed_for(self, distance_km: float) -> float:
        base = self.interurban_speed if distance_km > self.threshold_km else self.urban_speed
        return max(10.0, random.normalvariate(base, self.speed_sd))

    def travel_time_minutes(self, h3_dist: int) -> float:
        distance_km = max(0.0, float(h3_dist)) * self.km_per_cell
        return (distance_km / self._speed_for(distance_km)) * 60.0

    def travel_time_from_km(self, distance_km: float) -> float:
        distance_km = max(0.0, distance_km)
        return (distance_km / self._speed_for(distance_km)) * 60.0

    def idle_time_minutes(self, weekday: int, work_p: float) -> float:
        base = float(np.random.exponential(self.idle_mean))
        base = min(base, self.idle_max)        # trunca cauda longa
        # dias de mais demanda (work_p alto) => menos ocioso
        return max(1.0, base * (1.05 - 0.55 * work_p))


class CallModel:
    """Probabilidade de aceitar/receber chamado conforme distância H3."""

    def __init__(self, call_cfg: dict):
        self.base_p = float(call_cfg["base_p"])
        self.floor_p = float(call_cfg["floor_p"])
        self.max_dist = int(call_cfg["max_dist"])

    def probability(self, h3_distance: int) -> float:
        if h3_distance <= 0:
            return 1.0
        x = h3_distance / float(self.max_dist)
        if x <= 1.0:
            p = self.base_p * (1.0 - x * x)
            return max(self.floor_p, p)
        # fora do raio: cai, mas não zera
        return max(0.02, self.floor_p / (1.0 + (x - 1.0) * 2.5))


# ===========================================================================
# Amostrador por eventos (alias sampling) com cache por hora
# ===========================================================================
HOUR_MS = 3600 * 1000


def _floor_to_hour_ms(ts_ms: int) -> int:
    return (int(ts_ms) // HOUR_MS) * HOUR_MS


class _AliasSampler:
    """Amostragem O(1) sobre itens ponderados (Walker's alias method)."""

    __slots__ = ("items", "prob", "alias", "n")

    def __init__(self, items: list[str], weights: list[float]):
        self.items = items
        self.n = len(items)
        if self.n == 0:
            self.prob, self.alias = [], []
            return
        w = np.asarray(weights, dtype=np.float64)
        s = w.sum()
        if s <= 0:
            self.prob = [1.0] * self.n
            self.alias = list(range(self.n))
            return
        w = w * self.n / s
        small, large = [], []
        prob = np.zeros(self.n)
        alias = np.zeros(self.n, dtype=np.int64)
        for i, wi in enumerate(w):
            (small if wi < 1.0 else large).append(i)
        while small and large:
            si, li = small.pop(), large.pop()
            prob[si] = w[si]
            alias[si] = li
            w[li] = (w[li] + w[si]) - 1.0
            (small if w[li] < 1.0 else large).append(li)
        for i in large + small:
            prob[i] = 1.0
            alias[i] = i
        self.prob = prob.tolist()
        self.alias = alias.tolist()

    def sample(self) -> str | None:
        if self.n == 0:
            return None
        i = random.randrange(self.n)
        return self.items[i] if random.random() < self.prob[i] else self.items[self.alias[i]]


class DemandModel:
    """Mantém, por hora, as células aquecidas por eventos ativos."""

    def __init__(self, valid_cells: list[str], events: list[dict]):
        self.valid_cells = valid_cells
        self.M = len(valid_cells)
        self.by_hour: dict[int, list[dict]] = defaultdict(list)
        for ev in events:
            t = _floor_to_hour_ms(ev["start_ms"])
            end_h = _floor_to_hour_ms(ev["end_ms"])
            while t <= end_h:
                self.by_hour[t].append(ev)
                t += HOUR_MS
        self.cache: dict[int, tuple] = {}

    def _bucket(self, bucket_ms: int) -> tuple:
        if bucket_ms in self.cache:
            return self.cache[bucket_ms]
        active = self.by_hour.get(bucket_ms, [])
        if not active:
            res = (0.0, None, {})
            self.cache[bucket_ms] = res
            return res
        extra: dict[str, float] = defaultdict(float)
        best: dict[str, tuple[str, float]] = {}
        for ev in active:
            w = ev["weight"]
            for cell in ev["cells"]:
                extra[cell] += w
                prev = best.get(cell)
                if prev is None or w > prev[1]:
                    best[cell] = (ev["name"], w)
        cells = list(extra.keys())
        weights = [extra[c] for c in cells]
        total = float(sum(weights))
        sampler = _AliasSampler(cells, weights) if total > 0 else None
        res = (total, sampler, dict(best))
        self.cache[bucket_ms] = (total, sampler, best)
        return self.cache[bucket_ms]

    def choose_cell(self, ts_ms: int) -> tuple[str, str | None]:
        """Sorteia célula de demanda: background global vs. hotspot de evento."""
        total, sampler, best = self._bucket(_floor_to_hour_ms(ts_ms))
        if total <= 0 or sampler is None:
            return random.choice(self.valid_cells), None
        # probabilidade de cair no background uniforme vs. hotspot ponderado
        if random.random() < (self.M / (self.M + total)):
            return random.choice(self.valid_cells), None
        cell = sampler.sample()
        return cell, best.get(cell, (None, 0.0))[0]


# ===========================================================================
# Helpers geoespaciais
# ===========================================================================
def safe_grid_distance(a: str, b: str, clamp: int) -> int:
    """grid_distance tolerante: células distantes retornam o teto `clamp`."""
    if a == b:
        return 0
    try:
        d = h3.grid_distance(a, b)
        return min(int(d), clamp)
    except Exception:
        return clamp


def choose_local(current: str, valid_set: set[str], rng_ks=(5, 10, 20)) -> list[str]:
    """Vizinhança trafegável ao redor de `current` (fallback por anéis)."""
    for k in rng_ks:
        ring = h3.grid_disk(current, k)
        local = [c for c in ring if c in valid_set]
        if len(local) >= 50:
            return local
    return [c for c in h3.grid_disk(current, rng_ks[-1]) if c in valid_set]


# índice espacial grosseiro para amostrar destinos dentro de um raio em km.
# bins de ~5 km (0,045° de latitude); evita varrer 148k células por corrida.
BIN_DEG = 0.045
KM_PER_BIN = 5.0


def build_spatial_buckets(centroids: dict[str, tuple[float, float]]) -> dict:
    buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    for cell, (lat, lng) in centroids.items():
        buckets[(int(round(lat / BIN_DEG)), int(round(lng / BIN_DEG)))].append(cell)
    return dict(buckets)


def sample_within_radius(buckets: dict, lat: float, lng: float,
                         radius_km: float) -> str | None:
    """Sorteia uma célula de via dentro de ~`radius_km` do ponto (lat, lng).

    Escolhe um raio efetivo em U(1, radius_bins) para enviesar a distribuição
    a trajetos mais curtos (mais corridas/dia), depois um bin não-vazio próximo.
    """
    radius_bins = max(1, int(round(radius_km / KM_PER_BIN)))
    r = random.randint(1, radius_bins)
    base_lat = int(round(lat / BIN_DEG))
    base_lng = int(round(lng / BIN_DEG))
    offsets = [(dx, dy)
               for dx in range(-r, r + 1)
               for dy in range(-r, r + 1)]
    random.shuffle(offsets)
    for dx, dy in offsets:
        lst = buckets.get((base_lat + dx, base_lng + dy))
        if lst:
            return random.choice(lst)
    return None


# ===========================================================================
# Carregamento de dados
# ===========================================================================
def load_road_cells(cfg_roads: dict, seed: int) -> tuple[list[str], dict[str, tuple[float, float]]]:
    """Carrega células viárias no nível de resolução da simulação.

    Agrega a fonte (res 13) ao parent na resolução alvo, com amostragem
    determinística para limitar memória. Retorna lista de células + mapa
    célula -> (lat, lng) centróide (geometria real da via).
    """
    res = int(cfg_roads["h3_resolution"])
    df = pd.read_parquet(cfg_roads["source_parquet"],
                         columns=["h3_r13", "lat_center", "lng_center"])
    cap = cfg_roads.get("max_road_cells")
    if cap and len(df) > cap:
        df = df.sample(int(cap), random_state=seed)
    df["cell"] = df["h3_r13"].map(lambda c: h3.cell_to_parent(c, res))
    agg = df.groupby("cell").agg(lat=("lat_center", "mean"),
                                 lng=("lng_center", "mean"))
    cells = agg.index.tolist()
    centroids = {c: (float(r.lat), float(r.lng)) for c, r in agg.iterrows()}
    LOG.info("malha viária: %d células H3 res %d (fonte: %d linhas r13)",
             len(cells), res, len(df))
    return cells, centroids


def load_airport_cells(json_path: str, res: int, valid_set: set[str]) -> list[str]:
    """Células H3 do aeroporto na resolução da simulação, filtradas a vias."""
    areas = json.load(open(json_path, encoding="utf-8"))
    parents: set[str] = set()
    for area in areas:
        for c13 in area.get("cells_13", []):
            parents.add(h3.cell_to_parent(c13, res))
    on_roads = [c for c in parents if c in valid_set]
    # se o aeroporto não intersecta vias (pista não é "road"), usa os parents
    result = on_roads if on_roads else list(parents)
    LOG.info("células do aeroporto: %d (res %d; %d sobre vias)",
             len(result), res, len(on_roads))
    return result


def build_events(cfg_events: dict, res: int, airport_cells: list[str],
                 start: datetime, end: datetime) -> list[dict]:
    """Constrói eventos de demanda a partir de voos atrasados + estresse UTCI."""
    events: list[dict] = []
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    def _in_window(t0: pd.Timestamp) -> bool:
        ms = int(t0.timestamp() * 1000)
        return start_ms <= ms < end_ms

    # --- voos atrasados (ANAC VRA) ---
    # Peso do evento PROPORCIONAL ao volume de passageiros da janela horária.
    # Sem contagem de passageiros na VRA, usamos nº de chegadas na hora como
    # proxy × passengers_per_flight. Cria-se um evento por hora que contenha
    # ao menos uma chegada atrasada.
    fd = cfg_events["flight_delay"]
    if fd.get("enabled", True):
        flights = pd.read_parquet("data/raw/flights_anac.parquet",
                                  columns=["is_delayed", "arr_hour_local"])
        flights = flights.copy()
        flights["arr_hour_local"] = pd.to_datetime(flights["arr_hour_local"])
        flights["arr_hour_local"] = flights["arr_hour_local"].apply(
            lambda t: t.tz_localize(None) if t is not pd.NaT and t.tzinfo else t
        )
        flights = flights.dropna(subset=["arr_hour_local"])
        # agrega por hora: total de chegadas (proxy de passageiros) e atrasadas
        by_hour = flights.groupby("arr_hour_local").agg(
            n_arrivals=("is_delayed", "size"),
            n_delayed=("is_delayed", "sum"),
        ).reset_index()
        pax_per_flight = float(fd["passengers_per_flight"])
        w_per_pax = float(fd["weight_per_passenger"])
        n = 0
        for row in by_hour.itertuples(index=False):
            if row.n_delayed < 1:                 # evento só em horas com atraso
                continue
            t0 = pd.Timestamp(row.arr_hour_local)
            if not _in_window(t0):
                continue
            passengers = float(row.n_arrivals) * pax_per_flight
            weight = passengers * w_per_pax
            t_ms = int(t0.timestamp() * 1000)
            events.append({
                "name": "Voo Atrasado SBPA",
                "start_ms": t_ms, "end_ms": t_ms + HOUR_MS,
                "cells": airport_cells, "weight": weight,
            })
            n += 1
        LOG.info("eventos de voo atrasado: %d (peso ∝ passageiros/janela)", n)

    # --- estresse térmico UTCI (SBPA) ---
    ts = cfg_events["thermal_stress"]
    if ts.get("enabled", True):
        tc = pd.read_parquet("data/processed/thermal_comfort.parquet",
                             columns=["location", "time", "utci_stress"])
        tc = tc[tc["location"] == ts["location"]]
        sev_map = ts["severity"]
        base = float(ts["base_weight"])
        n = 0
        for row in tc.itertuples(index=False):
            sev = sev_map.get(row.utci_stress)
            if not sev:
                continue
            t0 = pd.Timestamp(row.time)
            if not _in_window(t0):
                continue
            t_ms = int(t0.timestamp() * 1000)
            events.append({
                "name": f"Estresse Termico {row.utci_stress}",
                "start_ms": t_ms, "end_ms": t_ms + HOUR_MS,
                "cells": airport_cells, "weight": base * float(sev),
            })
            n += 1
        LOG.info("eventos de estresse térmico: %d", n)

    LOG.info("total de eventos: %d", len(events))
    return events


# ===========================================================================
# Simulação
# ===========================================================================
def make_driver_specs(cfg: dict, cells: list[str], seed: int) -> list[dict]:
    """Gera as especificações dos motoristas (casa, jornada, nota-base, histórico).

    Feito uma única vez no processo principal (seed determinística) para depois
    ser fatiado entre os workers — cada motorista é simulado inteiro por um
    worker, preservando a continuidade da sua nota ao longo dos dias.
    """
    random.seed(seed)
    drv = cfg["drivers"]
    rt = cfg["ratings"]
    specs = []
    for i in range(int(drv["total_drivers"])):
        specs.append({
            "driver_id": i,
            "home": random.choice(cells),
            "daily_min": random.uniform(drv["daily_hours_min"],
                                        drv["daily_hours_max"]) * 60.0,
            "base_rating": random.triangular(rt["min"], rt["max"], rt["mode"]),
            "prior_trips": random.randint(int(rt["prior_trips_min"]),
                                          int(rt["prior_trips_max"])),
        })
    return specs


def simulate_specs(specs: list[dict], cfg: dict, cells: list[str], centroids: dict,
                   valid_set: set[str], events: list[dict], start: datetime,
                   end: datetime, seed: int, sink=None) -> "pd.DataFrame | None":
    """Simula um conjunto de motoristas (todos os dias) e retorna suas corridas.

    Se ``sink`` for fornecido, as corridas de cada dia são despachadas via
    ``sink(df_do_dia)`` e descartadas da memória (RAM constante — essencial no
    run completo com 300M+ linhas); nesse caso retorna ``None``.
    """
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))

    drv_cfg = cfg["drivers"]
    mob = cfg["mobility"]
    trips_cfg = cfg["trips"]
    time_model = TimeModel(mob)
    call_model = CallModel(cfg["call"])
    demand = DemandModel(cells, events)
    buckets = build_spatial_buckets(centroids)

    work_p_map = {int(k): float(v) for k, v in drv_cfg["work_probability"].items()}
    calls_per_step = int(mob["calls_per_idle_step"])
    idle_step = float(mob["idle_step_minutes"])
    call_max = int(cfg["call"]["max_dist"])
    max_trip_km = float(mob["max_trip_km"])
    return_after = int(trips_cfg["return_home_after"])
    return_radius = float(trips_cfg["return_radius_km"])
    target_min = int(trips_cfg["target_per_day_min"])
    target_max = int(trips_cfg["target_per_day_max"])
    shift_min = float(drv_cfg["shift_start_hour_min"])
    shift_max = float(drv_cfg["shift_start_hour_max"])
    shift_frac = float(drv_cfg["shift_window_fraction"])
    night_start = float(drv_cfg["night_shift_start_hour"])
    work_cap_min = float(drv_cfg["daily_hours_cap"]) * 60.0

    rt = cfg["ratings"]
    r_high = float(rt["high_threshold"])
    r_med = float(rt["medium_threshold"])
    pt_min, pt_max = float(rt["per_trip_min"]), float(rt["per_trip_max"])
    pt_sd = float(rt["per_trip_sd"])
    chain_offer = float(rt["chain_offer_prob"])
    chain_accept = float(rt["chain_accept_prob"])
    chain_radius = float(rt["chain_pickup_radius_km"])

    def rating_class(r: float) -> str:
        return "alta" if r >= r_high else "media" if r >= r_med else "regular"

    # estado de reputação por motorista (indexado por driver_id global)
    base_rating = {s["driver_id"]: s["base_rating"] for s in specs}
    prior_trips = {s["driver_id"]: s["prior_trips"] for s in specs}
    rating_sum = {s["driver_id"]: s["base_rating"] * s["prior_trips"] for s in specs}
    rating_cnt = {s["driver_id"]: float(s["prior_trips"]) for s in specs}

    rows: list[dict] = []
    trip_id = 0
    day = start
    while day < end:
        # convenção do config: 0=domingo..6=sábado (weekday() usa 0=segunda)
        dow_sun = (day.weekday() + 1) % 7
        work_p = work_p_map[dow_sun]
        date_str = day.date().isoformat()

        for spec in specs:
            drv_id = spec["driver_id"]
            home = spec["home"]
            base_min = spec["daily_min"]
            if random.random() > work_p:
                continue
            # jornada: piso de 8h, teto de daily_hours_cap (12h)
            work_limit = max(base_min, 480.0) * random.uniform(0.9, 1.1)
            work_limit = min(work_limit, work_cap_min)

            # 85% iniciam na janela diurna [min,max]; 15% turno noturno que
            # inicia às 18h e cruza a meia-noite (cobre a madrugada)
            if random.random() < shift_frac:
                start_h = random.uniform(shift_min, shift_max)
            else:
                start_h = night_start
            current_time = day + timedelta(hours=start_h)
            current = home
            worked = 0.0
            seq = 0
            first_origin_latlng: tuple[float, float] | None = None
            # meta diária de corridas (motorista roda até bater a meta ou o
            # limite de jornada): garante média de 25–35 corridas/dia.
            daily_target = random.randint(target_min, target_max)

            while worked < work_limit and seq < daily_target:
                idle_min = time_model.idle_time_minutes(dow_sun, work_p)
                idle_elapsed = 0.0
                assigned = False
                chained = False
                origin = None
                ev_name = None
                dist_call = 0

                # corrida ENCADEADA: motorista nota alta, ao finalizar uma
                # corrida, recebe chamada perto do destino atual. Pode recusar
                # (chain_accept) se não for uma boa corrida. Gatilho pela nota
                # estável do motorista (classe), não pela média flutuante.
                if (base_rating[drv_id] >= r_high and seq > 0
                        and random.random() < chain_offer
                        and random.random() < chain_accept):
                    c_lat, c_lng = centroids.get(current, (None, None))
                    near = (sample_within_radius(buckets, c_lat, c_lng, chain_radius)
                            if c_lat is not None else None)
                    if near:
                        origin, ev_name = near, "Corrida Encadeada"
                        dist_call = safe_grid_distance(current, origin, call_max)
                        assigned = True
                        chained = True

                while not assigned and idle_elapsed < idle_min:
                    ts_ms = int(current_time.timestamp() * 1000)
                    local = choose_local(current, valid_set)
                    for _ in range(calls_per_step):
                        # demanda: hotspot de evento vs. vizinhança local
                        hot, hot_ev = demand.choose_cell(ts_ms)
                        if hot_ev and hot in valid_set:
                            origin, ev_name = hot, hot_ev
                        elif local:
                            origin, ev_name = random.choice(local), None
                        else:
                            origin, ev_name = hot, hot_ev
                        dist_call = safe_grid_distance(current, origin, call_max)
                        if random.random() < call_model.probability(min(dist_call, call_max)):
                            assigned = True
                            break
                    if assigned:
                        break
                    step = min(idle_step, idle_min - idle_elapsed)
                    idle_elapsed += step
                    current_time += timedelta(minutes=step)

                if not assigned:
                    worked += idle_elapsed
                    continue

                pickup_time = time_model.travel_time_minutes(dist_call)

                o_lat, o_lng = centroids.get(origin, (None, None))
                if first_origin_latlng is None and o_lat is not None:
                    first_origin_latlng = (o_lat, o_lng)

                # destino limitado a raio urbano (`max_trip_km`). Após
                # `return_after` corridas, o motorista converge para o ponto da
                # 1ª corrida do dia (volta para casa): destinos sorteados num
                # raio pequeno em torno dela.
                if seq >= return_after and first_origin_latlng is not None:
                    anchor = first_origin_latlng
                    radius = return_radius
                else:
                    anchor = (o_lat, o_lng)
                    radius = max_trip_km

                dest = None
                if anchor[0] is not None:
                    dest = sample_within_radius(buckets, anchor[0], anchor[1], radius)
                if dest is None:
                    dest = origin
                d_lat, d_lng = centroids.get(dest, (None, None))
                if None not in (o_lat, d_lat):
                    trip_km = round(h3.great_circle_distance(
                        (o_lat, o_lng), (d_lat, d_lng), unit="km"), 3)
                    trip_km = min(trip_km, max_trip_km)   # teto urbano
                else:
                    trip_km = max_trip_km
                trip_time = time_model.travel_time_from_km(trip_km)

                # avaliação da corrida (3–5) diluída no histórico do motorista
                score = min(pt_max, max(pt_min,
                            random.normalvariate(base_rating[drv_id], pt_sd)))
                rating_sum[drv_id] += score
                rating_cnt[drv_id] += 1.0
                cur = rating_sum[drv_id] / rating_cnt[drv_id]

                rows.append({
                    "date": date_str,
                    "driver_id": drv_id,
                    "trip_seq": seq,
                    "request_ts": int(current_time.timestamp()),
                    "datetime": current_time,
                    "weekday": dow_sun,
                    "hour": current_time.hour,
                    "origin_h3": origin,
                    "destination_h3": dest,
                    "origin_lat": o_lat,
                    "origin_lng": o_lng,
                    "dest_lat": d_lat,
                    "dest_lng": d_lng,
                    "h3_distance_to_call": dist_call,
                    "trip_distance_km": trip_km,
                    "pickup_time_min": round(pickup_time, 2),
                    "trip_time_min": round(trip_time, 2),
                    "event_name": ev_name or "",
                    "chained_ride": chained,
                    "trip_score": round(score, 2),
                    "driver_rating": round(cur, 4),
                    "driver_rating_class": rating_class(base_rating[drv_id]),
                    "driver_prior_trips": prior_trips[drv_id],
                    "com_passageiro": True,
                })
                trip_id += 1
                worked += pickup_time + trip_time
                current_time += timedelta(minutes=pickup_time + trip_time)
                current = dest
                seq += 1

        if sink is not None:
            if rows:
                sink(pd.DataFrame(rows))
            rows = []                      # libera a memória do dia
        LOG.info("%s processado | corridas acumuladas: %d", date_str, trip_id)
        day += timedelta(days=1)

    return None if sink is not None else pd.DataFrame(rows)


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Compacta dtypes para carga eficiente em ML (LightGBM/SMOTE)."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["month"] = df["datetime"].dt.month.astype("int8")
    cat_cols = ["origin_h3", "destination_h3", "event_name", "date",
                "driver_rating_class"]
    for c in cat_cols:
        df[c] = df[c].astype("category")
    f32 = ["origin_lat", "origin_lng", "dest_lat", "dest_lng",
           "trip_distance_km", "pickup_time_min", "trip_time_min",
           "trip_score", "driver_rating"]
    for c in f32:
        df[c] = df[c].astype("float32")
    df["driver_id"] = df["driver_id"].astype("int32")
    df["driver_prior_trips"] = df["driver_prior_trips"].astype("int32")
    df["trip_seq"] = df["trip_seq"].astype("int16")
    df["h3_distance_to_call"] = df["h3_distance_to_call"].astype("int16")
    df["weekday"] = df["weekday"].astype("int8")
    df["hour"] = df["hour"].astype("int8")
    df["request_ts"] = df["request_ts"].astype("int64")
    df["com_passageiro"] = df["com_passageiro"].astype("bool")
    df["chained_ride"] = df["chained_ride"].astype("bool")
    return df


USE_DICT = ["origin_h3", "destination_h3", "event_name", "date",
            "driver_rating_class"]


def _resolve_fs(out_dir: str):
    """Retorna (filesystem, root_path). Suporta local e gs://bucket/prefix."""
    if "://" in out_dir:
        from pyarrow.fs import FileSystem
        fs, path = FileSystem.from_uri(out_dir)
        return fs, path
    return None, out_dir


def write_shards(df: pd.DataFrame, out_dir: str, batch_id: int, tag: int = 0) -> None:
    """Grava shards Parquet particionados por mês (zstd + dicionário).

    Vários workers (e vários dias, em streaming) escrevem no mesmo diretório sem
    colidir (basename único por batch+tag; ``overwrite_or_ignore`` não apaga
    shards de outros). Destino pode ser local ou ``gs://`` (GCS).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    fs, root = _resolve_fs(out_dir)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_to_dataset(
        table, root_path=root, filesystem=fs,
        partition_cols=["month"],
        compression="zstd",
        use_dictionary=USE_DICT,
        basename_template=f"batch{batch_id}-{tag}-{{i}}.parquet",
        existing_data_behavior="overwrite_or_ignore",
    )


def _make_sink(out_dir: str, batch_id: int):
    """Devolve um sink que grava cada DataFrame-dia como um shard próprio."""
    state = {"tag": 0}

    def sink(df_day: pd.DataFrame) -> None:
        write_shards(optimize_dtypes(df_day), out_dir, batch_id, state["tag"])
        state["tag"] += 1

    return sink


# --- infraestrutura de multiprocessing -------------------------------------
# Cada worker recebe (uma vez, no initializer) os dados compartilhados grandes
# (malha, eventos) e processa lotes de motoristas, gravando seus próprios shards.
_CTX: dict = {}


def _init_worker(cfg, cells, centroids, valid_set, events, start, end, out_dir, stream):
    _CTX.update(cfg=cfg, cells=cells, centroids=centroids, valid_set=valid_set,
                events=events, start=start, end=end, out_dir=out_dir, stream=stream)


def _run_worker(args) -> tuple[int, int]:
    batch_id, specs, seed = args
    if _CTX["stream"]:
        # streaming: grava dia a dia; precisa contar as corridas via sink
        counter = {"n": 0}
        base_sink = _make_sink(_CTX["out_dir"], batch_id)

        def sink(df_day):
            counter["n"] += len(df_day)
            base_sink(df_day)

        simulate_specs(specs, _CTX["cfg"], _CTX["cells"], _CTX["centroids"],
                       _CTX["valid_set"], _CTX["events"], _CTX["start"],
                       _CTX["end"], seed, sink=sink)
        return batch_id, counter["n"]

    df = simulate_specs(specs, _CTX["cfg"], _CTX["cells"], _CTX["centroids"],
                        _CTX["valid_set"], _CTX["events"], _CTX["start"],
                        _CTX["end"], seed)
    n = len(df)
    if n:
        write_shards(optimize_dtypes(df), _CTX["out_dir"], batch_id)
    return batch_id, n


def _chunk(lst: list, k: int) -> list[list]:
    """Divide `lst` em k lotes aproximadamente iguais."""
    k = max(1, k)
    size = (len(lst) + k - 1) // k
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def main() -> None:
    import os
    from multiprocessing import Pool

    configure_logging(log_file="reports/logs/simulate_drivers.log")
    params = load_config("params.yaml")
    seed = params["seed"]
    set_global_seed(seed)

    cfg = load_config("config/driver_simulation.yaml")
    res = int(cfg["roads"]["h3_resolution"])
    runtime = cfg.get("runtime", {})
    n_workers = int(runtime.get("n_workers", 1))
    if n_workers <= 0:
        n_workers = os.cpu_count() or 1
    out_dir = str(runtime.get("output_dir", "data/synthetic/driver_trips"))
    stream = bool(runtime.get("stream_per_day", False))

    start = datetime.fromisoformat(cfg["period"]["start_date"])
    end = start + timedelta(days=int(cfg["period"]["max_days"]))
    LOG.info("período simulado: %s → %s | workers: %d | saída: %s",
             start.date(), end.date(), n_workers, out_dir)

    cells, centroids = load_road_cells(cfg["roads"], seed)
    valid_set = set(cells)
    airport_cells = load_airport_cells(cfg["events"]["airport_cells_json"],
                                       res, valid_set)
    events = build_events(cfg["events"], res, airport_cells, start, end)

    specs = make_driver_specs(cfg, cells, seed)
    LOG.info("motoristas: %d", len(specs))

    # limpa saída local antes de escrever (GCS: shards sobrescritos por batch)
    is_local = "://" not in out_dir
    if is_local:
        import shutil
        if Path(out_dir).exists():
            shutil.rmtree(out_dir)
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    total = 0
    if n_workers == 1:
        if stream:
            counter = {"n": 0}
            base_sink = _make_sink(out_dir, 0)

            def sink(df_day):
                counter["n"] += len(df_day)
                base_sink(df_day)

            simulate_specs(specs, cfg, cells, centroids, valid_set, events,
                           start, end, seed, sink=sink)
            total = counter["n"]
        else:
            df = simulate_specs(specs, cfg, cells, centroids, valid_set, events,
                                start, end, seed)
            total = len(df)
            if total:
                df = optimize_dtypes(df)
                write_shards(df, out_dir, 0)
                _log_summary(df)
    else:
        batches = _chunk(specs, n_workers)
        # seed derivada por batch garante reprodutibilidade e fluxos distintos
        tasks = [(i, b, seed + 1 + i) for i, b in enumerate(batches)]
        with Pool(processes=n_workers,
                  initializer=_init_worker,
                  initargs=(cfg, cells, centroids, valid_set, events,
                            start, end, out_dir, stream)) as pool:
            for batch_id, n in pool.imap_unordered(_run_worker, tasks):
                total += n
                LOG.info("batch %d concluído: %d corridas", batch_id, n)

    # provenance em manifesto (não dá para hashear um dataset particionado)
    if is_local:
        manifest = Path(out_dir) / "_manifest.json"
        manifest.write_text(json.dumps({
            "n_trips": int(total), "n_events": len(events),
            "n_drivers": len(specs), "partition": "month",
        }, indent=2), encoding="utf-8")
        write_provenance(manifest, source="synthesis:driver_simulation", params={
            "driver_simulation": cfg, "n_events": len(events), "n_trips": int(total),
        })

    LOG.info("driver_trips salvo em %s | total de corridas: %d", out_dir, total)


def _log_summary(df: pd.DataFrame) -> None:
    per_driver_day = (df.groupby(["date", "driver_id"], observed=True).size()
                      .groupby(level=0, observed=True).mean().mean())
    exposed = (df["event_name"].astype(str).str.len() > 0).mean() * 100
    night = ((df["hour"] >= 0) & (df["hour"] < 5)).mean() * 100
    chained = df["chained_ride"].mean() * 100
    LOG.info("corridas: %d | média/motorista/dia: %.1f | evento: %.1f%% | "
             "madrugada(0-5h): %.1f%% | encadeadas: %.1f%%",
             len(df), per_driver_day, exposed, night, chained)


if __name__ == "__main__":
    main()
