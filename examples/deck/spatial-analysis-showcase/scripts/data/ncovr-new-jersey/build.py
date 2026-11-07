#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Build the 21-unit New Jersey extract of GeoDa's NCOVR common-support data.

Run with Python 3: python3 build.py --archive /path/to/ncovr.zip --codebook /path/to/codebook.pdf
The builder uses only the Python standard library and preserves source geometries/values.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import zipfile


OUTPUT_DIRECTORY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../public/data/ncovr-new-jersey")
)
EXPECTED_ARCHIVE_SHA256 = "e8cb04e6da634c6cd21808bd8cfe4dad6e295b22e8d40cc628e666887719cfe9"
EXPECTED_CODEBOOK_SHA256 = "229aac156c4005e70c2b43d15c69c73c0aacf4884a6af55f57487dbffe6b3de7"
FIELDS = {
    "unemployment1970": ("UE70", "percent", "Civilian labor-force unemployment rate, 1970 census year"),
    "unemployment1980": ("UE80", "percent", "Civilian labor-force unemployment rate, 1980 census year"),
    "unemployment1990": ("UE90", "percent", "Civilian labor-force unemployment rate, 1990 census year"),
    "familyPoverty1969": ("FP69", "percent", "Families below poverty, 1969/1970 census vintage"),
    "familyPoverty1979": ("FP79", "percent", "Families below poverty, 1979/1980 census vintage"),
    "familyPoverty1989": ("FP89", "percent", "Families below poverty, 1989/1990 census vintage"),
    "logDensity1970": ("DNL70", "natural-log persons per square mile", "Source log population density, 1970"),
    "logDensity1980": ("DNL80", "natural-log persons per square mile", "Source log population density, 1980"),
    "logDensity1990": ("DNL90", "natural-log persons per square mile", "Source log population density, 1990"),
    "population1970": ("PO70", "persons", "Decennial county population, 1970"),
    "population1980": ("PO80", "persons", "Decennial county population, 1980"),
    "population1990": ("PO90", "persons", "Decennial county population, 1990"),
    "logPopulation1970": ("POL70", "natural-log persons", "Source log population, 1970"),
    "logPopulation1980": ("POL80", "natural-log persons", "Source log population, 1980"),
    "logPopulation1990": ("POL90", "natural-log persons", "Source log population, 1990"),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_column(name, values, columns):
    path = os.path.join(OUTPUT_DIRECTORY, name + ".bin")
    with open(path, "wb") as output:
        for value in values:
            output.write(__import__("struct").pack("<I" if name == "fips" else "<f", value))
    columns[name] = {
        "file": name + ".bin",
        "dtype": "uint32" if name == "fips" else "float32",
        "components": 1,
        "length": len(values),
        **({"description": "Source NCOVR FIPSNO county unit identifier"} if name == "fips" else {
            "unit": FIELDS[name][1], "description": FIELDS[name][2]
        }),
    }


def write_integer_column(name, values, columns):
    path = os.path.join(OUTPUT_DIRECTORY, name + ".bin")
    with open(path, "wb") as output:
        for value in values:
            output.write(__import__("struct").pack("<I", value))
    columns[name] = {"file": name + ".bin", "dtype": "uint32", "components": 1, "length": len(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, help="Original downloaded ncovr.zip")
    parser.add_argument("--codebook", required=True, help="Original NCOVR codebook PDF")
    parser.add_argument(
        "--allow-unverified-source", action="store_true",
        help="Allow different source hashes and record the archive/codebook as unverified"
    )
    args = parser.parse_args()
    archive_hash = sha256(args.archive)
    codebook_hash = sha256(args.codebook)
    verified_source = archive_hash == EXPECTED_ARCHIVE_SHA256 and codebook_hash == EXPECTED_CODEBOOK_SHA256
    if not verified_source and not args.allow_unverified_source:
        raise ValueError("Archive/codebook SHA-256 differs from the pinned NCOVR inputs; use --allow-unverified-source to explicitly rebuild from another source")
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        csv_name = next(name for name in archive.namelist() if name.endswith("/NAT.csv"))
        geojson_name = next(name for name in archive.namelist() if name.endswith("/NAT.geojson"))
        source_rows = list(csv.DictReader(archive.open(csv_name).read().decode("utf-8-sig").splitlines()))
        source_geometry = json.loads(archive.read(geojson_name))

    selected = sorted((row for row in source_rows if row["STATE_FIPS"] == "34"), key=lambda row: int(row["FIPSNO"]))
    if len(selected) != 21:
        raise ValueError(f"Expected 21 New Jersey common-support units, found {len(selected)}")
    selected_ids = {int(row["FIPSNO"]) for row in selected}
    geometries = [
        feature for feature in source_geometry["features"]
        if int(feature["properties"]["FIPSNO"]) in selected_ids
    ]
    geometry_by_fips = {int(feature["properties"]["FIPSNO"]): feature for feature in geometries}
    if set(geometry_by_fips) != selected_ids:
        raise ValueError("Geometry identifiers do not match all 21 selected data rows")
    geometries = [geometry_by_fips[int(row["FIPSNO"])] for row in selected]

    # The source geometry coordinates are already longitude/latitude; retain the historical
    # common-support shapes and let the scene project them into a local metric analysis frame.
    geometry_path = os.path.join(OUTPUT_DIRECTORY, "counties.geojson")
    with open(geometry_path, "w", encoding="utf-8") as output:
        json.dump({"type": "FeatureCollection", "features": geometries}, output, separators=(",", ":"))

    columns = {}
    write_column("fips", [int(row["FIPSNO"]) for row in selected], columns)
    for name, (source_field, _, _) in FIELDS.items():
        values = [float(row[source_field]) for row in selected]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite source values in {source_field}")
        write_column(name, values, columns)

    # Build exact shared-vertex queen and shared-edge rook graphs from the supplied source shapes.
    # Geometry coordinates are source float values; no modern boundary overlay or nearest-neighbor
    # approximation is introduced. Both relations are symmetric, sorted CSR lists.
    point_owners = {}
    edge_owners = {}
    for row_index, feature in enumerate(geometries):
        rings = []
        def collect(value):
            if isinstance(value, list) and len(value) >= 2 and all(isinstance(part, (int, float)) for part in value[:2]):
                rings.append(value[:2])
            elif isinstance(value, list):
                for child in value:
                    collect(child)
        collect(feature["geometry"]["coordinates"])
        points = {tuple(point) for point in rings}
        for point in points:
            point_owners.setdefault(point, set()).add(row_index)
        # Re-traverse individual rings so edges never cross between polygon parts or holes.
        def collect_rings(value):
            if not value:
                return
            if isinstance(value[0], list) and len(value[0]) >= 2 and all(isinstance(part, (int, float)) for part in value[0][:2]):
                yield value
            else:
                for child in value:
                    yield from collect_rings(child)
        for ring in collect_rings(feature["geometry"]["coordinates"]):
            for point_index, point in enumerate(ring):
                next_point = ring[(point_index + 1) % len(ring)]
                if tuple(point) == tuple(next_point):
                    continue
                edge = tuple(sorted((tuple(point), tuple(next_point))))
                edge_owners.setdefault(edge, set()).add(row_index)

    def adjacency(owner_groups, prefix):
        neighbors = [set() for _ in geometries]
        for owners in owner_groups.values():
            for left in owners:
                neighbors[left].update(right for right in owners if right != left)
        offsets = [0]
        flattened = []
        for row_neighbors in neighbors:
            flattened.extend(sorted(row_neighbors))
            offsets.append(len(flattened))
        write_integer_column(prefix + "Offsets", offsets, columns)
        write_integer_column(prefix + "Neighbors", flattened, columns)
    adjacency(point_owners, "queen")
    adjacency(edge_owners, "rook")

    names = [{"fips": int(row["FIPSNO"]), "name": row["NAME"]} for row in selected]
    names_path = os.path.join(OUTPUT_DIRECTORY, "names.json")
    with open(names_path, "w", encoding="utf-8") as output:
        json.dump(names, output, separators=(",", ":"))

    coordinates = []
    def visit(value):
        if isinstance(value, list) and len(value) >= 2 and all(isinstance(part, (int, float)) for part in value[:2]):
            coordinates.append(value[:2])
        elif isinstance(value, list):
            for child in value:
                visit(child)
    for feature in geometries:
        visit(feature["geometry"]["coordinates"])
    bbox = [min(p[0] for p in coordinates), min(p[1] for p in coordinates), max(p[0] for p in coordinates), max(p[1] for p in coordinates)]
    artifact_hashes = {
        filename: sha256(os.path.join(OUTPUT_DIRECTORY, filename))
        for filename in sorted(os.listdir(OUTPUT_DIRECTORY))
        if filename != "manifest.json" and os.path.isfile(os.path.join(OUTPUT_DIRECTORY, filename))
    }
    manifest = {
        "id": "ncovr-new-jersey", "version": 1, "kind": "polygons", "count": 21,
        "bbox": [round(value, 6) for value in bbox], "crs": "EPSG:4326", "columns": columns,
        "geometry": {"type": "polygons", "file": "counties.geojson"},
        "properties": {
            "rowOrder": "ascending NCOVR FIPSNO; all columns and geometry features use this order",
            "units": "21 New Jersey county units represented in the NCOVR Horan-Hargis archive historical common-support template; the template label does not imply changed present county identities",
            "analysisProjection": "Spherical azimuthal-equidistant planar-meter frame centered at [-74.55, 40.05], used consistently for local analysis and inverse-projected map display; source GeoJSON remains longitude/latitude",
            "sourceVariables": "UE = percent civilian labor force unemployed; FP = percent of families below the period poverty threshold; DNL = log population density; PO = decennial population counts; POL = log population",
            "limitations": "The three periods are 1970, 1980 and 1990 source observations, not an annual series. FP uses source census-vintage thresholds; FP59/1960 is excluded because its $3,000 threshold differs. Nominal income fields are not included.",
            "sourceTerms": "GeoDa describes the data as provided as-is with no warranties. No additional license is asserted here.",
            "derivedFilesSha256": artifact_hashes,
            "sources": {
                "data": "https://geodacenter.github.io/data-and-lab/ncovr/",
                "archive": "https://geodacenter.github.io/data-and-lab/data/ncovr.zip",
                "archiveSha256": archive_hash,
                "codebook": "NCOVR codebook.pdf in the archive",
                "suppliedCodebookSha256": codebook_hash,
                "sourceHashVerification": "verified against pinned archive and codebook" if verified_source else "unverified CLI override",
                "attribution": "Messner, Anselin, Hawkins, Deane, Tolnay and Baller (2000), An Atlas of the Spatial Patterning of County-Level Homicide, 1960-1990; archive prepared by Luc Anselin, last updated 2003-10-04"
            }
        }
    }
    with open(os.path.join(OUTPUT_DIRECTORY, "manifest.json"), "w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")
    print(f"Wrote {len(selected)} county units and {len(columns)} columns to {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
