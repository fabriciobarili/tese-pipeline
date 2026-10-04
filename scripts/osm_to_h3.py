#!/usr/bin/env python
"""Extrai vias navegáveis do OSM PBF e gera índices H3 na resolução 13.

Uso:
    python scripts/osm_to_h3.py ext/sul-261003.osm.pbf
    python scripts/osm_to_h3.py ext/sul-261003.osm.pbf --out ext/vias_h3r13.parquet
    python scripts/osm_to_h3.py --help

Fonte dos dados: https://download.geofabrik.de/south-america/brazil/sul.html
Acessado: 04/10/2026 | Dados ate: 2026-10-03T20:20:50Z

Saida:
    <nome>_vias_h3r13.parquet  -- celulas H3, com way_id e tipo de via
    <nome>_vias_h3r13.meta.json -- metadados de proveniencia
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import h3
import osmium
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger(__name__)

# RM Porto Alegre: bounds derivados dos poligonos UDH 2010
RM_POA_BBOX = dict(min_lat=-31.5, max_lat=-29.0, min_lng=-52.5, max_lng=-50.5)

# Hierarquia de prioridade (menor indice = maior prioridade)
HIGHWAY_PRIORITY = [
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
    "unclassified", "residential", "living_street", "service",
]
NAVIGABLE_HIGHWAY = set(HIGHWAY_PRIORITY)
HW_RANK = {hw: i for i, hw in enumerate(HIGHWAY_PRIORITY)}

H3_RES = 13


class RoadHandler(osmium.SimpleHandler):
    """Processa ways navegaveis e coleta celulas H3 res 13."""

    def __init__(self, bbox: dict, highways: set, res: int = H3_RES) -> None:
        super().__init__()
        self.bbox = bbox
        self.highways = highways
        self.res = res
        # celula -> (way_id, highway_rank, highway_type, lat_center, lng_center)
        self._cells: dict[str, tuple] = {}
        self._n_ways = 0
        self._n_edges = 0
        self._t0 = time.time()

    # ------------------------------------------------------------------
    def way(self, w) -> None:
        hw = w.tags.get("highway")
        if hw not in self.highways:
            return

        hw_rank = HW_RANK[hw]

        # coleta nos com localizacao valida
        nodes: list[tuple[float, float] | None] = []
        try:
            for n in w.nodes:
                try:
                    nodes.append((n.location.lat, n.location.lon))
                except osmium.InvalidLocationError:
                    nodes.append(None)
        except Exception:
            return

        # verifica se a via toca o bbox
        in_bbox = any(
            nd is not None
            and self.bbox["min_lat"] <= nd[0] <= self.bbox["max_lat"]
            and self.bbox["min_lng"] <= nd[1] <= self.bbox["max_lng"]
            for nd in nodes
        )
        if not in_bbox:
            return

        self._n_ways += 1
        way_id = w.id

        # pares consecutivos de nos validos
        valid = [(i, nd) for i, nd in enumerate(nodes) if nd is not None]
        for j in range(len(valid) - 1):
            _, (lat1, lng1) = valid[j]
            _, (lat2, lng2) = valid[j + 1]

            cell1 = h3.latlng_to_cell(lat1, lng1, self.res)
            cell2 = h3.latlng_to_cell(lat2, lng2, self.res)

            try:
                path = h3.grid_path_cells(cell1, cell2)
            except Exception:
                path = [cell1, cell2]

            self._n_edges += 1
            for cell in path:
                prev = self._cells.get(cell)
                if prev is None or hw_rank < prev[1]:
                    lat_c, lng_c = h3.cell_to_latlng(cell)
                    self._cells[cell] = (way_id, hw_rank, hw, round(lat_c, 7), round(lng_c, 7))

        if self._n_ways % 10_000 == 0:
            elapsed = time.time() - self._t0
            LOG.info(
                "vias=%d edges=%d celulas=%d tempo=%.0fs",
                self._n_ways, self._n_edges, len(self._cells), elapsed,
            )

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {"h3_r13": cell, "way_id": v[0], "highway": v[2],
             "lat_center": v[3], "lng_center": v[4]}
            for cell, v in self._cells.items()
        ]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("h3_r13").reset_index(drop=True)
        return df


def write_meta(pbf_path: Path, out_path: Path, n_cells: int, n_ways: int,
               elapsed: float) -> None:
    meta = {
        "artifact": str(out_path),
        "source_file": str(pbf_path),
        "source_url": "https://download.geofabrik.de/south-america/brazil/sul.html",
        "source_accessed": "2026-10-04",
        "source_data_until": "2026-10-03T20:20:50Z",
        "bbox": RM_POA_BBOX,
        "h3_resolution": H3_RES,
        "highway_types": HIGHWAY_PRIORITY,
        "n_cells": n_cells,
        "n_ways": n_ways,
        "processing_seconds": round(elapsed, 1),
        "generator": "scripts/osm_to_h3.py",
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("metadados salvos: %s", meta_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("pbf", type=Path, help="Caminho para o arquivo .osm.pbf")
    ap.add_argument("--out", type=Path, default=None,
                    help="Caminho de saida .parquet (padrao: <pbf>_vias_h3r13.parquet)")
    ap.add_argument("--res", type=int, default=H3_RES,
                    help=f"Resolucao H3 (padrao: {H3_RES})")
    args = ap.parse_args()

    pbf = args.pbf
    if not pbf.exists():
        LOG.error("Arquivo nao encontrado: %s", pbf)
        sys.exit(1)

    out = args.out or pbf.with_name(pbf.stem + f"_vias_h3r{args.res}.parquet")

    LOG.info("Lendo: %s (%.0f MB)", pbf, pbf.stat().st_size / 1e6)
    LOG.info("Bbox RM Porto Alegre: %s", RM_POA_BBOX)
    LOG.info("Resolucao H3: %d | Tipos de via: %d", args.res, len(NAVIGABLE_HIGHWAY))

    handler = RoadHandler(RM_POA_BBOX, NAVIGABLE_HIGHWAY, args.res)

    t0 = time.time()
    LOG.info("Iniciando parse OSM (passa dupla para carregar localizacoes dos nos)...")
    handler.apply_file(str(pbf), locations=True, idx="flex_mem")
    elapsed = time.time() - t0

    LOG.info(
        "Concluido: vias=%d edges=%d celulas=%d tempo=%.1fs",
        handler._n_ways, handler._n_edges, len(handler._cells), elapsed,
    )

    df = handler.to_dataframe()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    LOG.info("Parquet salvo: %s (%d linhas)", out, len(df))

    write_meta(pbf, out, len(df), handler._n_ways, elapsed)

    # resumo por tipo de via
    if not df.empty:
        LOG.info("\nDistribuicao por tipo de via:")
        for hw, cnt in df["highway"].value_counts().items():
            LOG.info("  %-20s %7d celulas", hw, cnt)


if __name__ == "__main__":
    main()
