#!/usr/bin/env python
"""Converte polígonos de um arquivo KMZ para células H3 em múltiplas resoluções.

Uso:
    python scripts/kmz_to_h3.py ext/Aeroporto\\ Salgado\\ Filho.kmz --res 13 10 8 5
    python scripts/kmz_to_h3.py --help

Saídas (na mesma pasta do KMZ):
    <nome>.json  — dados brutos (células por polígono e resolução)
    <nome>_tabela.md — tabela legível em Markdown

Modo de cobertura:
    res 13 → 'center'  (centróide da célula dentro do polígono)
    res <13 → 'overlap' (célula toca o polígono)
    Use --mode para forçar um modo para todas as resoluções.
"""

from __future__ import annotations

import argparse
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import h3

NS = "http://www.opengis.net/kml/2.2"
AREA_KM2 = {0: 4_357_449, 1: 609_788, 2: 86_745, 3: 12_392, 4: 1_770,
             5: 252.9, 6: 36.1, 7: 5.16, 8: 0.737, 9: 0.105,
             10: 0.015, 11: 0.002, 12: 0.0003, 13: 0.000043}


def parse_kmz(kmz_path: Path) -> list[dict]:
    """Extrai placemarks com polígonos de um KMZ. Coordenadas em (lat, lng)."""
    with zipfile.ZipFile(kmz_path) as z:
        kml_name = next(n for n in z.namelist() if n.endswith(".kml"))
        root = ET.fromstring(z.read(kml_name))
    placemarks = []
    for pm in root.iter(f"{{{NS}}}Placemark"):
        name_el = pm.find(f"{{{NS}}}name")
        name = (name_el.text or "sem nome").strip()
        coords_el = pm.find(f".//{{{NS}}}coordinates")
        if coords_el is None:
            continue
        polygon = [
            (float(tok.split(",")[1]), float(tok.split(",")[0]))
            for tok in coords_el.text.strip().split()
        ]
        placemarks.append({"name": name, "polygon": polygon})
    return placemarks


def centroid(polygon: list[tuple]) -> tuple[float, float]:
    lats = [p[0] for p in polygon]
    lngs = [p[1] for p in polygon]
    return sum(lats) / len(lats), sum(lngs) / len(lngs)


def cells_for(polygon: list[tuple], res: int, mode: str | None) -> list[str]:
    poly = h3.LatLngPoly(polygon)
    m = mode or ("center" if res >= 13 else "overlap")
    return sorted(h3.h3shape_to_cells_experimental(poly, res, m))


def build_results(placemarks: list[dict], resolutions: list[int],
                  mode: str | None) -> list[dict]:
    results = []
    for pm in placemarks:
        cen = centroid(pm["polygon"])
        row: dict = {
            "name": pm["name"],
            "n_vertices": len(pm["polygon"]),
            "centroid_lat": cen[0],
            "centroid_lng": cen[1],
        }
        for res in resolutions:
            cells = cells_for(pm["polygon"], res, mode)
            row[f"cells_{res}"] = cells
            row[f"n_{res}"] = len(cells)
            row[f"centroid_cell_{res}"] = h3.latlng_to_cell(cen[0], cen[1], res)
        results.append(row)
    return results


def write_json(results: list[dict], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def write_markdown(results: list[dict], resolutions: list[int],
                   kmz_path: Path, out_path: Path) -> None:
    mode_label = {res: ("center" if res >= 13 else "overlap") for res in resolutions}
    L: list[str] = []

    L += [
        f"# Tabela de Conversão H3 — {kmz_path.stem}",
        "",
        f"**Fonte:** `{kmz_path}`  ",
        "**Versão H3:** 4.x  ",
        "**Modo:** `center` em res ≥ 13; `overlap` em res < 13  ",
        f"**Gerado por:** `scripts/kmz_to_h3.py`",
        "",
        "---",
        "",
        "## Sumário",
        "",
    ]

    # header
    res_headers = " | ".join(f"Res {r} (n)" for r in resolutions)
    res_sep     = " | ".join("------" for _ in resolutions)
    L.append(f"| Polígono | Vértices | Centróide |  {res_headers} |")
    L.append(f"|----------|----------|-----------|  {res_sep} |")
    for r in results:
        cen = f"`{r['centroid_lat']:.6f}, {r['centroid_lng']:.6f}`"
        ns  = " | ".join(str(r[f"n_{res}"]) for res in resolutions)
        L.append(f"| {r['name']} | {r['n_vertices']} | {cen} | {ns} |")
    # union row
    u_ns = []
    for res in resolutions:
        u = set(c for row in results for c in row[f"cells_{res}"])
        u_ns.append(str(len(u)))
    L.append(f"| **União** | — | — | {' | '.join(u_ns)} |")
    L += ["", "---", "", "## Referência de resoluções", "",
          "| Resolução | Área média (km²) | Uso típico |",
          "|-----------|-----------------|------------|"]
    USES = {13: "precisão máxima (sub-métrica)", 10: "nível de rua",
            8: "zona de demanda ride-sharing", 5: "região metropolitana",
            7: "bairro / sub-cidade", 9: "quarteirão"}
    for res in resolutions:
        L.append(f"| {res} | ~{AREA_KM2.get(res, '?')} | {USES.get(res, '')} |")
    L += ["", "---", ""]

    # por polígono
    for row in results:
        L.append(f"## Polígono: {row['name']}")
        L += ["",
              f"**Vértices:** {row['n_vertices']}  ",
              f"**Centróide:** lat `{row['centroid_lat']:.7f}` lng `{row['centroid_lng']:.7f}`",
              ""]
        for res in resolutions:
            cells = row[f"cells_{res}"]
            L.append(f"### Res {res} — {len(cells)} célula(s) ({mode_label[res]}, "
                     f"~{AREA_KM2.get(res, '?')} km²/célula)")
            L += ["", f"Célula do centróide: `{row[f'centroid_cell_{res}']}`", ""]
            sample = cells if len(cells) <= 50 else cells[:20]
            trailer = cells[-5:] if len(cells) > 50 else []
            L += ["| # | Célula H3 | Lat centro | Lng centro |",
                  "|---|-----------|-----------|-----------|"]
            for i, cell in enumerate(sample, 1):
                lat_c, lng_c = h3.cell_to_latlng(cell)
                L.append(f"| {i} | `{cell}` | {lat_c:.7f} | {lng_c:.7f} |")
            if trailer:
                L.append(f"| … | *… {len(cells) - 25} células omitidas …* | | |")
                for i, cell in enumerate(trailer, len(cells) - 4):
                    lat_c, lng_c = h3.cell_to_latlng(cell)
                    L.append(f"| {i} | `{cell}` | {lat_c:.7f} | {lng_c:.7f} |")
            L.append("")
        L += ["---", ""]

    # união
    L += ["## União — todos os polígonos", "",
          "Células que cobrem **qualquer** área do KMZ.", ""]
    for res in resolutions:
        u = sorted(set(c for row in results for c in row[f"cells_{res}"]))
        airport_lat = sum(r["centroid_lat"] for r in results) / len(results)
        airport_lng = sum(r["centroid_lng"] for r in results) / len(results)
        airport_cell = h3.latlng_to_cell(airport_lat, airport_lng, res)
        L.append(f"### Res {res} — {len(u)} célula(s) ({mode_label[res]})")
        L += ["", f"Célula do centróide geral: `{airport_cell}`", ""]
        sample = u if len(u) <= 50 else u[:20]
        L += ["| # | Célula H3 | Lat centro | Lng centro |",
              "|---|-----------|-----------|-----------|"]
        for i, cell in enumerate(sample, 1):
            lat_c, lng_c = h3.cell_to_latlng(cell)
            L.append(f"| {i} | `{cell}` | {lat_c:.7f} | {lng_c:.7f} |")
        if len(u) > 50:
            L.append(f"> {len(u)} células — lista completa em `{out_path.with_suffix('.json').name}`")
        L.append("")

    out_path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kmz", type=Path, help="Caminho para o arquivo .kmz")
    parser.add_argument("--res", type=int, nargs="+", default=[13, 10, 8, 5],
                        help="Resoluções H3 (padrão: 13 10 8 5)")
    parser.add_argument("--mode", choices=["center", "full", "overlap", "bbox_overlap"],
                        default=None,
                        help="Forçar modo de cobertura para todas as resoluções")
    args = parser.parse_args()

    kmz = args.kmz
    base = kmz.with_suffix("")
    json_out = base.with_name(base.name + ".json")
    md_out   = base.with_name(base.name + "_tabela.md")

    print(f"Lendo: {kmz}")
    placemarks = parse_kmz(kmz)
    print(f"  {len(placemarks)} polígono(s) encontrado(s)")

    results = build_results(placemarks, sorted(args.res, reverse=True), args.mode)
    write_json(results, json_out)
    write_markdown(results, sorted(args.res, reverse=True), kmz, md_out)

    print(f"JSON salvo: {json_out}")
    print(f"Markdown salvo: {md_out}")
    for r in results:
        print(f"\n  {r['name']}")
        for res in sorted(args.res, reverse=True):
            print(f"    res {res:2d}: {r[f'n_{res}']:6d} células")


if __name__ == "__main__":
    main()
