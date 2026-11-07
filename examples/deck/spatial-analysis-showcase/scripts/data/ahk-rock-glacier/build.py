#!/usr/bin/env python3
# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Prepare a bounded, real two-epoch rock-glacier sample; never invent surface motion."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

import laspy
import numpy as np
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "public/data/ahk-rock-glacier"
ARCHIVE = Path("/tmp/py4dgeo-data-2026-01-29.tar.gz")
ARCHIVE_URL = (
    "https://github.com/3dgeo-heidelberg/py4dgeo-test-data/releases/download/"
    "2026-01-29/data.tar.gz"
)
ARCHIVE_SHA256 = "f4c2edb52a6e2c40206ca07782c4b70e450742fb60aa48478674e776e6a33651"
SOURCE_DOI = "https://doi.org/10.11588/data/XHYB10"
NATIVE_ORIGIN = np.array([652920.0, 5189155.0, 2485.0])
VOXEL_ORIGIN = np.array([652900.0, 5189100.0, 2470.0])
VOXEL_SIZE = 0.6
CORE_STRIDE = 8
FILES = {
    "reference": "ahk_2017_652900_5189100_gnd_subarea.laz",
    "comparison": "ahk_2018A_652900_5189100_gnd_subarea.laz",
    "cores": "ahk_cp_652900_5189100_subarea.laz",
}


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    if not ARCHIVE.exists():
        urllib.request.urlretrieve(ARCHIVE_URL, ARCHIVE)
    payload = ARCHIVE.read_bytes()
    if digest(payload) != ARCHIVE_SHA256:
        raise RuntimeError("Source archive checksum differs from the reviewed release")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        # Read only explicitly named members; never extract an archive to the workspace.
        extracted = {
            name: archive.extractfile(name).read()
            for name in [*FILES.values(), "tfM.csv", "redPoint.csv", "LICENSE.md"]
        }
    transform = np.loadtxt(io.BytesIO(extracted["tfM.csv"]), delimiter=",")
    reduction_origin = np.loadtxt(io.BytesIO(extracted["redPoint.csv"]), delimiter=",")
    # The provider calls the native frame UTM32N. EPSG:32632 is used for map placement only;
    # all point-cloud computations retain the supplied native metre coordinates.
    projection = Transformer.from_crs(32632, 4326, always_xy=True)
    columns = {}
    metadata = {}
    all_geographic = []
    retained_clouds = {}

    def write_column(name: str, values: np.ndarray, dtype: str, unit: str = "") -> None:
        array = np.asarray(values, dtype=np.dtype(dtype).newbyteorder("<"))
        filename = f"{name}.bin"
        (OUTPUT / filename).write_bytes(array.tobytes())
        columns[name] = {
            "file": filename,
            "dtype": dtype,
            "length": len(array),
            **({"components": array.shape[1]} if array.ndim == 2 else {}),
            **({"unit": unit} if unit else {}),
        }

    for epoch in ["reference", "comparison", "cores"]:
        cloud = laspy.read(io.BytesIO(extracted[FILES[epoch]]))
        native = cloud.xyz
        if epoch == "comparison":
            # Exactly the provider's documented reduction-origin affine transformation.
            native = (native - reduction_origin) @ transform[:, :3].T + transform[:, 3] + reduction_origin
        if epoch == "cores":
            retained = np.arange(0, len(native), CORE_STRIDE, dtype=np.uint32)
            normals = np.column_stack([cloud[f"normals{axis}"] for axis in range(3)])
            write_column("normal", normals[retained], "float32", "unit vector in native frame")
        else:
            cells = np.floor((native - VOXEL_ORIGIN) / VOXEL_SIZE).astype(np.int64)
            retained = np.sort(np.unique(cells, axis=0, return_index=True)[1]).astype(np.uint32)
        native = native[retained]
        retained_clouds[epoch] = native - NATIVE_ORIGIN
        if not np.isfinite(native).all():
            raise RuntimeError("Nonfinite retained source point")
        longitude, latitude = projection.transform(native[:, 0], native[:, 1])
        geographic = np.column_stack([longitude, latitude])
        all_geographic.append(geographic)
        position_name = "position" if epoch == "cores" else f"{epoch}Position"
        write_column(position_name, geographic, "float64", "longitude/latitude degrees")
        write_column(f"{epoch}XYZ", native - NATIVE_ORIGIN, "float32", "native UTM32N metres relative to nativeOrigin")
        write_column(f"{epoch}SourceRow", retained, "uint32", "original LAZ row")
        metadata[epoch] = {
            "sourceFile": FILES[epoch],
            "sourceSha256": digest(extracted[FILES[epoch]]),
            "sourceCount": cloud.header.point_count,
            "retainedCount": len(retained),
            "nativeBounds": [native.min(axis=0).tolist(), native.max(axis=0).tolist()],
        }
        if epoch == "cores":
            retained_normals = normals[retained]
            metadata[epoch]["invalidNormalCount"] = int(
                (~np.isfinite(retained_normals).all(axis=1)).sum()
            )
            metadata[epoch]["normalPolicy"] = (
                "Preserve provider normals, including nonfinite values; reject invalid normals "
                "during change analysis rather than synthesizing an orientation."
            )

    # A second study compares cell-mean elevations, keeping observation support explicit.
    # The north-up one-metre grid is fixed across epochs; missing support is never interpolated.
    grid_width, grid_height = 38, 37
    grid_left, grid_top = -18.0, 18.0
    cell_size = 1.0
    minimum_returns = 3
    for epoch in ["reference", "comparison"]:
        cloud = retained_clouds[epoch]
        column = np.floor((cloud[:, 0] - grid_left) / cell_size).astype(int)
        row = np.floor((grid_top - cloud[:, 1]) / cell_size).astype(int)
        if not ((column >= 0) & (column < grid_width) & (row >= 0) & (row < grid_height)).all():
            raise RuntimeError("Source row outside declared DEM grid")
        indices = row * grid_width + column
        counts = np.bincount(indices, minlength=grid_width * grid_height)
        sums = np.bincount(indices, weights=cloud[:, 2], minlength=grid_width * grid_height)
        means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts >= minimum_returns)
        write_column(f"dem{epoch.title()}", means, "float32", "native elevation metres relative to nativeOrigin Z; NaN if fewer than three returns")
        write_column(f"dem{epoch.title()}Count", counts, "uint32", "retained original returns in cell")
    grid_columns, grid_rows = np.meshgrid(np.arange(grid_width), np.arange(grid_height))
    center_x = NATIVE_ORIGIN[0] + grid_left + (grid_columns.ravel() + 0.5) * cell_size
    center_y = NATIVE_ORIGIN[1] + grid_top - (grid_rows.ravel() + 0.5) * cell_size
    center_longitude, center_latitude = projection.transform(center_x, center_y)
    write_column("demPosition", np.column_stack([center_longitude, center_latitude]), "float64", "longitude/latitude degrees at DEM cell centres")
    west, south = projection.transform(NATIVE_ORIGIN[0] + grid_left, NATIVE_ORIGIN[1] + grid_top - grid_height)
    east, north = projection.transform(NATIVE_ORIGIN[0] + grid_left + grid_width, NATIVE_ORIGIN[1] + grid_top)
    dem_grid = {
        "width": grid_width, "height": grid_height,
        "affine": [cell_size, 0, grid_left, 0, -cell_size, grid_top],
        "pixelInterpretation": "area", "cellSizeMeters": cell_size,
        "bounds": [west, south, east, north], "minimumReturns": minimum_returns,
        "algorithm": "Arithmetic mean of retained original returns per fixed north-up 1 m native-XY cell; fewer than three returns are NaN. No interpolation or hole filling.",
        "uncertainty": "No observed stable-control mask or calibrated difference variance is supplied. Any chosen control region and measurement covariance are explicit scenario assumptions.",
    }

    combined = np.vstack(all_geographic)
    bounds = [float(combined[:, 0].min()), float(combined[:, 1].min()), float(combined[:, 0].max()), float(combined[:, 1].max())]
    origin_longitude, origin_latitude = projection.transform(*NATIVE_ORIGIN[:2])
    properties = {
        "source": SOURCE_DOI,
        "sourceArchive": ARCHIVE_URL,
        "sourceArchiveSha256": ARCHIVE_SHA256,
        "license": "CC BY 4.0",
        "attribution": "Winiwarter, Anders, Zahs, Hämmerle and Höfle (2021), heiDATA, doi:10.11588/data/XHYB10; sample distributed by 3DGeo Heidelberg py4dgeo-test-data",
        "epochs": ["2017", "2018A"],
        "location": "Äußeres Hochebenkar rock glacier, Obergurgl, Tirol, Austria",
        "nativeOrigin": NATIVE_ORIGIN.tolist(),
        "origin": [origin_longitude, origin_latitude],
        "nativeFrame": "Provider UTM32N horizontal coordinates and supplied elevation, metres; vertical datum not specified in sample",
        "displayProjection": "EPSG:32632 assumed solely for longitude/latitude map placement; computational XYZ remains in native metres",
        "sampling": "First original LAZ row in each occupied 0.6 m 3D voxel; source row order restored. Every eighth provided core point and its provided normal retained.",
        "voxelSizeMeters": VOXEL_SIZE,
        "voxelOrigin": VOXEL_ORIGIN.tolist(),
        "alignment": "Comparison epoch transformed once by supplied tfM.csv around supplied redPoint.csv, following py4dgeo M3C2-EP documentation. No additional local ICP alignment applied during preparation.",
        "comparisonAffineTransform": transform.tolist(),
        "comparisonReductionOrigin": reduction_origin.tolist(),
        "uncertainty": "Sampling changes roughness and support. This subset is not the published M3C2-EP result and does not carry the original full 12x12 registration covariance or per-scanner measurement propagation.",
        "sources": metadata,
        "demGrid": dem_grid,
        "checksums": {spec["file"]: digest((OUTPUT / spec["file"]).read_bytes()) for spec in columns.values()},
    }
    manifest = {
        "id": "ahk-rock-glacier", "version": 1, "kind": "points",
        "count": metadata["cores"]["retainedCount"], "bbox": bounds,
        "crs": "EPSG:4326 display positions; native UTM32N computational XYZ",
        "columns": columns, "properties": properties,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUTPUT / "SOURCE-LICENSE.md").write_bytes(extracted["LICENSE.md"])
    (OUTPUT / "metadata.json").write_text(json.dumps({
        **properties, "bbox": bounds,
        "outputSha256": {spec["file"]: digest((OUTPUT / spec["file"]).read_bytes()) for spec in columns.values()},
        "binaryBytes": sum((OUTPUT / spec["file"]).stat().st_size for spec in columns.values()),
    }, indent=2) + "\n")
    print(json.dumps({"bbox": bounds, "origin": properties["origin"], "counts": {key: value["retainedCount"] for key, value in metadata.items()}}))


if __name__ == "__main__":
    main()
