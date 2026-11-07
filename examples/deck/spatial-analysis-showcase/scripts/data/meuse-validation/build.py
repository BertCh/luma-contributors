#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Build the reproducible Meuse zinc interpolation inputs from the R sp data exports.

Run with a temporary environment containing pyproj, e.g.:
  python -m venv /tmp/meuse-builder-venv
  /tmp/meuse-builder-venv/bin/pip install pyproj
  /tmp/meuse-builder-venv/bin/python build.py
"""

import hashlib
import json
import math
import pathlib
import sys
import urllib.request

from pyproj import Transformer

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "public" / "data" / "meuse-validation"
BASE = "https://edzer.r-universe.dev/sp/data/"
SOURCES = {
    "samples": BASE + "meuse/json",
    "grid": BASE + "meuse.grid/json",
    "river": BASE + "meuse.riv/json",
}
EXPECTED_SOURCE_SHA256 = {
    "samples": "aea92cadbf125b4bb7dc60d81b88bf1a1f15bf66eea22e27074d9093a65e8bd3",
    "grid": "71057969ba536a79fc4af11733de75b0d83c074bfd758d7565f98c8d298e1c52",
    "river": "f07fd4efecd86750854245490e39b0c20564ec804990c6b3a8d5572cf042daec",
}
TO_WGS84 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)


def fetch_json(url, expected_sha256):
    request = urllib.request.Request(url, headers={"User-Agent": "luma-contributors dataset builder"})
    with urllib.request.urlopen(request, timeout=60) as response:
        source_bytes = response.read()
    actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"Source changed: {url} has SHA-256 {actual_sha256}; expected {expected_sha256}")
    return json.loads(source_bytes), actual_sha256


def write_column(name, values, components=1):
    import array

    payload = array.array("f", values)
    if payload.itemsize != 4:
        raise RuntimeError("Expected 32-bit float array")
    if sys.byteorder != "little":
        payload.byteswap()
    path = OUT / f"{name}.bin"
    with path.open("wb") as stream:
        payload.tofile(stream)
    return {"file": path.name, "dtype": "float32", "components": components, "length": len(values) // components}


def write_category(name, rows, source_name):
    import array

    categories = sorted({row.get(source_name, "missing") for row in rows})
    indexes = {value: index for index, value in enumerate(categories)}
    payload = array.array("B", [indexes[row.get(source_name, "missing")] for row in rows])
    path = OUT / f"{name}.bin"
    with path.open("wb") as stream:
        payload.tofile(stream)
    return {"file": path.name, "dtype": "uint8", "components": 1, "length": len(rows), "categories": categories}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    samples, sample_hash = fetch_json(SOURCES["samples"], EXPECTED_SOURCE_SHA256["samples"])
    grid, grid_hash = fetch_json(SOURCES["grid"], EXPECTED_SOURCE_SHA256["grid"])
    river, river_hash = fetch_json(SOURCES["river"], EXPECTED_SOURCE_SHA256["river"])
    if len(samples) != 155 or len(grid) != 3103 or len(river) != 176:
        raise RuntimeError(f"Unexpected source counts: {len(samples)}, {len(grid)}, {len(river)}")

    columns = {}
    display_positions = []
    for label, rows in (("sample", samples), ("grid", grid)):
        positions_rd = [coordinate for row in rows for coordinate in (row["x"], row["y"])]
        positions_geo = []
        for index in range(0, len(positions_rd), 2):
            longitude, latitude = TO_WGS84.transform(positions_rd[index], positions_rd[index + 1])
            positions_geo.extend((longitude, latitude))
        display_positions.extend(positions_geo)
        columns[f"{label}Position"] = write_column(f"{label}Position", positions_geo, 2)
        columns[f"{label}PositionRD"] = write_column(f"{label}PositionRD", positions_rd, 2)
        for name in (("zinc", "copper", "lead", "cadmium", "dist") if label == "sample" else ("zinc", "dist")):
            if name in rows[0]:
                columns[f"{label}{name.title()}"] = write_column(
                    f"{label}{name.title()}", [float(row[name]) for row in rows]
                )
        if label == "sample":
            for name in ("zinc", "copper", "lead", "cadmium"):
                columns[f"log{name.title()}"] = write_column(f"log{name.title()}", [math.log(row[name]) for row in rows])
            for source_name, column_name in (("elev", "sampleElevation"), ("om", "sampleOrganicMatter"), ("dist.m", "sampleDistanceMeters")):
                columns[column_name] = write_column(column_name, [float(row.get(source_name, float("nan"))) for row in rows])
            for source_name, column_name in (("soil", "sampleSoil"), ("lime", "sampleLime"), ("ffreq", "sampleFloodFrequency"), ("landuse", "sampleLandUse")):
                columns[column_name] = write_category(column_name, rows, source_name)

    river_geo = []
    for x, y in river:
        river_geo.append(TO_WGS84.transform(x, y))
    features = [{"type": "Feature", "properties": {"name": "Meuse river outline"}, "geometry": {"type": "LineString", "coordinates": river_geo}}]
    (OUT / "river.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":")))

    bbox = [min(display_positions[0::2]), min(display_positions[1::2]), max(display_positions[0::2]), max(display_positions[1::2])]
    manifest = {
        "id": "meuse-validation", "version": 1, "kind": "points", "count": 155,
        "bbox": bbox, "crs": "EPSG:4326", "columns": columns,
        "geometry": {"type": "lines", "file": "river.geojson"},
        "properties": {
            "description": "155 measured topsoil zinc samples and the 3,103-cell 40 m prediction support grid, in the original EPSG:28992 frame and WGS84 for display.",
            "source": "R package sp 2.2-4, dataset meuse / meuse.grid / meuse.riv; package commit eae4821e83b8b9eb5fc46c9f05bb9523747c8598.",
            "sourceUrls": SOURCES,
            "sourceSHA256": {"samples": sample_hash, "grid": grid_hash, "river": river_hash},
            "sourceCrs": "EPSG:28992", "displayCrs": "EPSG:4326",
            "response": "log(zinc mg/kg); the exponentiated log-scale prediction is a median, not original-scale expected zinc.",
            "sampleCaution": "Each topsoil observation is a composite sample over a 15 m by 15 m area represented by one point. The published cadmium series replaces recorded zeros with 0.2 mg/kg; do not interpret those entries as measured exact concentrations.",
            "gridCaution": "The original documentation says the grid soil/flooding classes have uncertain provenance; only its x/y coordinates and dist covariate are used here.",
            "license": "GPL-2.0-or-later (R package sp data)."
        }
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    hashes = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "sha256.json":
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"Wrote {len(samples)} samples, {len(grid)} grid cells and {len(river)} river vertices to {OUT}")


if __name__ == "__main__":
    main()
