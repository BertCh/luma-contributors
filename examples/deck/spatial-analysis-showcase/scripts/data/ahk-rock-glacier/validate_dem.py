#!/usr/bin/env python3
# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Independent float64 median-bias/covariance reference for the sampled epoch cell means."""

import json
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "public/data/ahk-rock-glacier"
grid = json.loads((DATA / "manifest.json").read_text())["properties"]["demGrid"]
reference = np.fromfile(DATA / "demReference.bin", dtype="<f4").astype(float)
comparison = np.fromfile(DATA / "demComparison.bin", dtype="<f4").astype(float)
difference = comparison - reference
valid = np.isfinite(difference)
columns, rows = np.meshgrid(np.arange(grid["width"]), np.arange(grid["height"]))
columns, rows = columns.ravel(), rows.ravel()
locations = np.column_stack([columns, rows]).astype(float)
distances = np.linalg.norm(locations[:, None, :] - locations[None, :, :], axis=2)
cases = []
for region in ["west", "east", "north", "all"]:
    region_mask = {
        "west": columns < grid["width"] / 2,
        "east": columns >= grid["width"] / 2,
        "north": rows < grid["height"] / 2,
        "all": np.ones_like(valid),
    }[region]
    stable = valid & region_mask
    bias = np.median(difference[stable])
    deviation = np.median(np.abs(difference[stable] - bias)) * 1.4826022
    corrected = difference[valid] - bias
    assumptions = [(0, 0.15), (2, 0.15), (5, 0.15)]
    if region == "west":
        assumptions.append((2, 0.3))
    for correlation_length, assumed_difference_deviation in assumptions:
        correlation = np.eye(len(valid)) if correlation_length == 0 else np.exp(-distances / correlation_length)
        stable_covariance = correlation[np.ix_(stable, stable)].sum()
        bias_error = 1.2533141 * deviation * np.sqrt(stable_covariance) / stable.sum()
        measurement_covariance = assumed_difference_deviation ** 2 * correlation[np.ix_(valid, valid)].sum()
        volume_deviation = np.sqrt(measurement_covariance + valid.sum() ** 2 * bias_error ** 2)
        corrected_deviation = np.sqrt(assumed_difference_deviation ** 2 + bias_error ** 2)
        cases.append({
            "region": region, "correlationLengthMeters": correlation_length,
            "assumedDifferenceDeviationMeters": assumed_difference_deviation,
            "stableCount": int(stable.sum()), "validCells": int(valid.sum()),
            "medianBiasMeters": float(bias), "residualNmadMeters": float(deviation),
            "biasStandardErrorMeters": float(bias_error),
            "correctedVolumeCubicMeters": float(corrected.sum()),
            "volumeStandardDeviationCubicMeters": float(volume_deviation),
            "validAreaSquareMeters": int(valid.sum()),
            "detectableCells": int((np.abs(corrected) > 1.96 * corrected_deviation).sum()),
        })
print(json.dumps({
    "method": "Float64 NumPy median/NMAD and explicit exponential covariance matrix on shipped float32 cell means; one square metre cells, minimum three observed returns per epoch. Control regions and 0.15/0.30 m raw-difference SD are assumptions.",
    "rawVolumeCubicMeters": float(difference[valid].sum()), "cases": cases
}, indent=2, allow_nan=False))
