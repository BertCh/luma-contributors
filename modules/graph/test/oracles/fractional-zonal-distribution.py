# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""RS01 independent reference, reproducible with Python 3.12 and:

pip install exactextract==0.3.0 numpy==2.3.4 shapely==2.1.2

Direct exactextract.exact_extract calls supply coverage-weighted histogram quantiles,
mean, population variance and count. Shapely builds a multipolygon of disjoint cell
strips of exactly representable widths; it does not calculate the statistics.
NumPy cov(aweights=coverage, ddof=1) supplies reliability sample variance. NumPy
sum/dot supply Kish effective support as an explicitly composed reference. Planar
unit cells, positive fractions, no observation-weight raster; nodata is NaN.
"""
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource
from shapely.geometry import box, mapping, Polygon
from shapely.ops import unary_union

VERSIONS = {"exactextract": "0.3.0", "numpy": "2.3.4", "shapely": "2.1.2"}
for package, expected in VERSIONS.items():
    assert version(package) == expected, (package, version(package), expected)

PROBABILITIES = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
CASES = [
    ("unequal-coverage-repeated-values", [-4, 0, 2, 2, 9, 20], [1, 0.5, 0.25, 0.75, 0.125, 0.0625]),
    ("one-observed-cell", [7, 11], [0.25, 0]),
    ("one-observed-cell-nondyadic-weight", [7, 11], [0.09, 0]),
    ("equal-value-distinct-cell-support", [3, 3, 3], [1, 0.5, 0.25]),
    ("nodata-retains-covered-area", [None, 2, None, 8], [1, 0.5, 0.25, 0.125]),
    ("large-offset-small-spread", [1048576, 1048576.25, 1048576.5], [1, 0.5, 0.25]),
    ("all-nodata", [None, None], [0.5, 1]),
    ("empty-zone", [2, 8], [0, 0]),
    ("unit-weights-negative-positive", [-8, -2, 0, 4, 11], [1, 1, 1, 1, 1]),
    ("tiny-positive-support", [-2, 5, 10], [1 / 4096, 1 / 8192, 1 / 16384]),
]


def finite_or_null(value):
    return float(value) if value is not None and np.isfinite(value) else None


def reference_case(name, source_values, fractions, geometry_override=None, width=None):
    values = np.array([np.nan if value is None else value for value in source_values], dtype=np.float32)
    weights = np.array(fractions, dtype=np.float32).astype(np.float64)
    polygons = [box(index, 0, index + fraction, 1) for index, fraction in enumerate(fractions) if fraction > 0]
    geometry = geometry_override if geometry_override is not None else (unary_union(polygons) if polygons else box(len(values) + 1, 0, len(values) + 2, 1))
    width = width or len(values)
    height = len(values) // width
    raster = NumPyRasterSource(values.reshape(height, width), 0, 0, width, height, nodata=np.nan)
    operations = ["mean", "variance", "count"] + [f"quantile(q={probability})" for probability in PROBABILITIES]
    feature = {"type": "Feature", "geometry": mapping(geometry), "properties": {}}
    result = exact_extract(raster, [feature], operations)[0]["properties"]
    quantiles = [finite_or_null(result[f"quantile_{int(100 * probability)}"]) for probability in PROBABILITIES]
    accepted = np.isfinite(values) & (weights > 0)
    valid_weights = weights[accepted]
    weight_sum = np.sum(valid_weights)
    squared_weight_sum = np.dot(valid_weights, valid_weights)
    sample_variance = np.cov(values[accepted].astype(np.float64), aweights=valid_weights, ddof=1) if np.count_nonzero(accepted) > 1 else np.nan
    effective_support = weight_sum**2 / squared_weight_sum if squared_weight_sum else 0
    return {
        "name": name, "values": source_values, "fractions": fractions, "probabilities": PROBABILITIES,
        "expected": {
            "means": finite_or_null(result["mean"]), "populationVariances": finite_or_null(result["variance"]),
            "sampleVariances": finite_or_null(sample_variance), "effectiveSampleSizes": float(effective_support),
            "weightSums": float(result["count"]), "squaredWeightSums": float(squared_weight_sum),
            "coveredArea": float(np.sum(weights)), "validArea": float(weight_sum), "quantiles": quantiles,
            "validity": int(weight_sum > 0),
            "status": (4 if weight_sum == 0 else 0) | (8 if np.count_nonzero(accepted) < 2 else 0),
        },
    }


hole_positions = [0.25, 0.25, 2.75, 0.25, 2.75, 2.75, 0.25, 2.75, 1, 1, 1, 2, 2, 2, 2, 1]
hole_polygon = Polygon([(0.25, 0.25), (2.75, 0.25), (2.75, 2.75), (0.25, 2.75)], holes=[[(1, 1), (1, 2), (2, 2), (2, 1)]])
hole_case = reference_case("partial-polygon-with-hole", list(range(1, 10)), [0.5625, 0.75, 0.5625, 0.75, 0, 0.75, 0.5625, 0.75, 0.5625], hole_polygon, 3)
hole_case["positions"] = hole_positions
hole_case["width"] = 3
hole_case["height"] = 3
fixture = {
    "versions": {**VERSIONS, "python": platform.python_version()},
    "reference": {
        "direct": "exactextract.exact_extract: quantile, variance, mean, count",
        "sampleVariance": "numpy.cov(aweights=coverage, ddof=1)",
        "effectiveSupport": "composed numpy.sum(weights)**2 / numpy.dot(weights,weights)",
        "quantileConvention": "exactextract 0.3.0: coverage histogram by distinct value; interpolated weighted plotting positions",
        "coordinates": "planar unit area cells; no implicit resampling; NaN nodata ignored",
    },
    "cases": [reference_case(*entry) for entry in CASES],
    "coverageIntegration": hole_case,
}
Path(__file__).with_name("fixtures").joinpath("fractional-zonal-distribution.json").write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n")
