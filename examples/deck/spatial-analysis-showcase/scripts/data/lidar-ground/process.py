#!/usr/bin/env python3
# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Build the bounded Seattle lidar sample used by the lidar-ground story.

Requires Python 3.11+, laspy, lazrs, numpy and pyproj. Outputs little-endian float64 longitude/latitude
to preserve the 0.5 m sample spacing, NAVD88 elevation in metres, and uint32 LAS classification.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import urllib.request

import laspy
import numpy as np
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "public/data/lidar-ground"
SOURCE_PATH = Path("/tmp/USGS_LPC_WA_KingCounty_2021_B21_King_2282.laz")
SOURCE_URL = (
    "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/"
    "WA_KingCounty_2021_B21/WA_KingCo_1_2021/LAZ/"
    "USGS_LPC_WA_KingCounty_2021_B21_King_2282.laz"
)
SOURCE_ITEM = "https://www.sciencebase.gov/catalog/item/619ce206d34eb622f694418b"
SOURCE_METADATA = (
    "https://thor-f5.er.usgs.gov/ngtoc/metadata/waf/elevation/lidar_point_cloud/laz/"
    "WA_KingCo_1_2021/USGS_LPC_WA_KingCounty_2021_B21_King_2282.xml"
)
SOURCE_SHA256 = "cfceeda5525b335ba523613b9837c650d72d7cc40b8c32d22c18ff7344124ab3"
SOURCE_POINT_COUNT = 6_373_072
SOURCE_CRS = "EPSG:6597 + EPSG:6360"
OUTPUT_CRS = "EPSG:4326 positions; NAVD88 elevation in metres"
FOOT_TO_METRE = 0.3048006096012192
CENTER_FOOT = (1_268_750.0, 221_000.0)
HALF_WIDTH_FOOT = 295.275590551
SAMPLE_SPACING_METRE = 0.5
SEED = 20210424


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_source() -> None:
    SOURCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SOURCE_PATH.exists() and sha256_file(SOURCE_PATH) == SOURCE_SHA256:
        return
    temporary_path = SOURCE_PATH.with_suffix(SOURCE_PATH.suffix + ".part")
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "luma.gl lidar story"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response, temporary_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if sha256_file(temporary_path) != SOURCE_SHA256:
            raise RuntimeError("Downloaded source checksum does not match the pinned USGS tile")
        os.replace(temporary_path, SOURCE_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    download_source()
    source_sha256 = sha256_file(SOURCE_PATH)
    if SOURCE_SHA256 and source_sha256 != SOURCE_SHA256:
        raise RuntimeError(f"Source checksum mismatch: {source_sha256}")

    header = laspy.open(SOURCE_PATH).header
    source_crs = header.parse_crs()
    if header.point_count != SOURCE_POINT_COUNT or source_crs is None:
        raise RuntimeError("The downloaded tile does not match the recorded USGS product metadata")

    center_easting, center_northing = CENTER_FOOT
    half_width = HALF_WIDTH_FOOT
    low_easting, high_easting = center_easting - half_width, center_easting + half_width
    low_northing, high_northing = center_northing - half_width, center_northing + half_width
    width = int(round((2 * half_width * FOOT_TO_METRE) / SAMPLE_SPACING_METRE))
    rng = np.random.default_rng(SEED)
    chosen_priority = np.full(width * width, np.inf, dtype=np.float32)
    chosen_points = np.zeros((width * width, 4), dtype=np.float64)
    chosen_classes = np.zeros(width * width, dtype=np.uint32)
    crop_count = 0

    with laspy.open(SOURCE_PATH) as reader:
        for points in reader.chunk_iterator(500_000):
            easting = np.asarray(points.x)
            northing = np.asarray(points.y)
            inside = (
                (easting >= low_easting)
                & (easting < high_easting)
                & (northing >= low_northing)
                & (northing < high_northing)
            )
            if not inside.any():
                continue
            crop_count += int(inside.sum())
            east = easting[inside]
            north = northing[inside]
            elevation = np.asarray(points.z)[inside] * FOOT_TO_METRE
            classes = np.asarray(points.classification)[inside].astype(np.uint32)
            east_offset = (east - center_easting) * FOOT_TO_METRE
            north_offset = (north - center_northing) * FOOT_TO_METRE
            columns = np.floor((east_offset + half_width * FOOT_TO_METRE) / SAMPLE_SPACING_METRE).astype(np.int32)
            rows = np.floor((north_offset + half_width * FOOT_TO_METRE) / SAMPLE_SPACING_METRE).astype(np.int32)
            cells = rows * width + columns
            priorities = rng.random(len(cells), dtype=np.float32)
            for index in range(len(cells)):
                cell = cells[index]
                if priorities[index] < chosen_priority[cell]:
                    chosen_priority[cell] = priorities[index]
                    chosen_points[cell] = (east_offset[index], north_offset[index], elevation[index], 0.0)
                    chosen_classes[cell] = classes[index]

    occupied = np.isfinite(chosen_priority)
    local_points = chosen_points[occupied, :3]
    selected_classes = chosen_classes[occupied]
    to_lonlat = Transformer.from_crs(source_crs, 4326, always_xy=True)
    longitude, latitude = to_lonlat.transform(
        center_easting + local_points[:, 0] / FOOT_TO_METRE,
        center_northing + local_points[:, 1] / FOOT_TO_METRE,
    )
    positions = np.column_stack((longitude, latitude)).astype("<f8")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    longitude, latitude = to_lonlat.transform(center_easting, center_northing)
    (OUTPUT / "position.bin").write_bytes(positions.tobytes())
    (OUTPUT / "elevation.bin").write_bytes(local_points[:, 2].astype("<f4").tobytes())
    (OUTPUT / "classification.bin").write_bytes(selected_classes.astype("<u4").tobytes())
    output_hashes = {
        name: sha256_file(OUTPUT / name)
        for name in ("position.bin", "elevation.bin", "classification.bin")
    }
    class_ids, class_counts = np.unique(selected_classes, return_counts=True)
    bounds = [
        float(positions[:, 0].min()),
        float(positions[:, 1].min()),
        float(positions[:, 0].max()),
        float(positions[:, 1].max()),
    ]
    properties = {
        "source": SOURCE_URL,
        "sourceItem": SOURCE_ITEM,
        "sourceMetadata": SOURCE_METADATA,
        "sourceSha256": source_sha256,
        "collectionDates": ["2021-04-01", "2021-04-24"],
        "license": "U.S. Government work; USGS 3DEP products are public domain",
        "horizontalCrs": "EPSG:6597 (NAD83(2011) / Washington North, US survey feet)",
        "verticalCrs": "EPSG:6360 (NAVD88 height, Geoid18, US survey feet)",
        "verticalOutput": "NAVD88 metres",
        "origin": [longitude, latitude],
        "cropSizeMeters": [2 * half_width * FOOT_TO_METRE] * 2,
        "sampling": "one seeded random original return per occupied 0.5 m horizontal cell; no fabricated or interpolated points",
        "seed": SEED,
    }
    manifest = {
        "id": "lidar-ground",
        "version": 1,
        "kind": "points",
        "count": int(occupied.sum()),
        "bbox": bounds,
        "crs": "EPSG:4326",
        "columns": {
            "position": {"file": "position.bin", "dtype": "float64", "components": 2, "length": int(occupied.sum())},
            "elevation": {"file": "elevation.bin", "dtype": "float32", "length": int(occupied.sum()), "unit": "m NAVD88"},
            "classification": {"file": "classification.bin", "dtype": "uint32", "length": int(occupied.sum()), "unit": "LAS classification code"},
        },
        "properties": properties,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    metadata = {
        "title": "USGS 3DEP King County 2021 tile 2282, clipped sample",
        "source": SOURCE_URL,
        "sourceItem": SOURCE_ITEM,
        "sourceMetadata": SOURCE_METADATA,
        "sourceSha256": source_sha256,
        "sourcePointCount": header.point_count,
        "sourceBoundsFeet": {
            "easting": [float(header.mins[0]), float(header.maxs[0])],
            "northing": [float(header.mins[1]), float(header.maxs[1])],
            "elevationNAVD88Geoid18": [float(header.mins[2]), float(header.maxs[2])],
        },
        "collectionDates": ["2021-04-01", "2021-04-24"],
        "license": "U.S. Government work; USGS 3DEP products are public domain",
        "horizontalCrs": "EPSG:6597 (NAD83(2011) / Washington North, US survey feet)",
        "verticalCrs": "EPSG:6360 (NAVD88 height, Geoid18, US survey feet)",
        "outputCoordinates": OUTPUT_CRS,
        "origin": [longitude, latitude],
        "boundsWgs84": bounds,
        "cropCenterSourceCoordinates": [center_easting, center_northing],
        "cropSizeMeters": [2 * half_width * FOOT_TO_METRE] * 2,
        "sampleSpacingMeters": SAMPLE_SPACING_METRE,
        "sampling": "one seeded random original return per occupied 0.5 m horizontal cell; no fabricated or interpolated points",
        "seed": SEED,
        "croppedPointCount": crop_count,
        "samplePointCount": int(occupied.sum()),
        "finiteXYZCount": int(np.isfinite(chosen_points[occupied, :3]).all(axis=1).sum()),
        "sampleBoundsLocalMetres": {
            "east": [float(chosen_points[occupied, 0].min()), float(chosen_points[occupied, 0].max())],
            "north": [float(chosen_points[occupied, 1].min()), float(chosen_points[occupied, 1].max())],
            "elevationNAVD88": [float(chosen_points[occupied, 2].min()), float(chosen_points[occupied, 2].max())],
        },
        "classCounts": {str(int(class_id)): int(count) for class_id, count in zip(class_ids, class_counts)},
        "columns": {
            "position": "position.bin (longitude, latitude float64 pairs)",
            "elevation": "elevation.bin (NAVD88 metres float32)",
            "classification": "classification.bin (LAS class uint32)",
        },
        "manifest": "manifest.json",
        "finiteXYZCount": int(np.isfinite(local_points).all(axis=1).sum()),
        "outputSha256": output_hashes,
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
