# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference fixtures for GPURasterStretch (math correctness review 2026-11-10, raster-imagery-12).

- Equalization: OpenCV `cv2.equalizeHist` on an 8-bit raster. Its rule,
  `(cdf[v] - cdfFirst) / (total - cdfFirst)`, is the one the published `GPURasterContrast` and
  `GPURasterStretch` use. (scikit-image `exposure.equalize_hist` is importable but follows another
  rule, the inclusive CDF interpolated at bin centres without `cdfFirst`, so it is not a reference.)
- Percentile bounds: `numpy.percentile(values, q, method='lower')`, the order statistic at
  `floor((N - 1) * q)`. The kernel reports the centre of the histogram bin that holds it.
- Curve: NumPy evaluation of the linear stretch, the display gamma `t ** (1 / gamma)` and the
  rio-color sigmoidal contrast, written from the rio-color source formula. rio-color and rasterio
  are not installed, so they were not run.

Output: fixtures/raster-stretch-references.json. Recorded with the pinned NumPy 2.5.4 on CPython
3.14 and OpenCV 4.12.0 (`opencv-python-headless==4.12.0.88`, which requirements.txt does not list:
install it with `--no-deps`, its metadata asks for an older NumPy). The first run, with NumPy
2.0.2 on the system Python 3.9.6, gave the same values to the last digit.
"""
import json
import pathlib
import platform

import cv2
import numpy as np

OUTPUT = pathlib.Path(__file__).resolve().parent / 'fixtures' / 'raster-stretch-references.json'


def sigmoidal(values, contrast, bias):
    """rio-color `sigmoidal(arr, contrast, bias)`: normalised so 0 maps to 0 and 1 to 1."""
    numerator = 1 / (1 + np.exp(contrast * (bias - values))) - 1 / (1 + np.exp(contrast * bias))
    denominator = 1 / (1 + np.exp(contrast * (bias - 1))) - 1 / (1 + np.exp(contrast * bias))
    return numerator / denominator


def main():
    random = np.random.default_rng(20261110)

    # Skewed 8-bit raster with both ends present, so the automatic domain is [0, 255] and 256
    # bins hold one value each.
    width, height = 24, 16
    image = np.clip(np.round(random.gamma(2.0, 28.0, size=(height, width))), 0, 255).astype(np.uint8)
    image[0, 0] = 0
    image[0, 1] = 255
    equalized = cv2.equalizeHist(image)

    # Float raster for the percentile bounds and the curve.
    float_width, float_height = 30, 20
    samples = (random.lognormal(mean=1.0, sigma=0.6, size=float_width * float_height) * 12.5 - 4.0).astype(np.float32)
    percentile_cases = []
    for low, high in ((2, 98), (10, 90), (25, 75), (50, 99.5), (0, 100)):
        bounds = np.percentile(samples.astype(np.float64), [low, high], method='lower')
        percentile_cases.append({'percentiles': [low, high], 'lower': [float(bounds[0]), float(bounds[1])]})

    minimum = float(samples.min())
    maximum = float(samples.max())
    unit = (samples.astype(np.float64) - minimum) / (maximum - minimum)
    gamma, contrast, midpoint = 2.2, 6.0, 0.4
    curve = sigmoidal(unit ** (1 / gamma), contrast, midpoint)

    fixture = {
        'source': 'cv2.equalizeHist, numpy.percentile(method="lower"), NumPy evaluation of the rio-color sigmoidal formula (rio-color not run)',
        'versions': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'opencv': cv2.__version__,
        },
        'equalize': {
            'width': width,
            'height': height,
            'values': [int(v) for v in image.ravel()],
            'expectedOpenCV': [int(v) for v in equalized.ravel()],
        },
        'float': {
            'width': float_width,
            'height': float_height,
            'values': [float(v) for v in samples],
            'percentiles': percentile_cases,
            'curve': {
                'gamma': gamma,
                'sigmoidContrast': contrast,
                'sigmoidMidpoint': midpoint,
                'expected': [float(v) for v in curve],
            },
        },
    }
    OUTPUT.write_text(json.dumps(fixture))
    print('wrote', OUTPUT, OUTPUT.stat().st_size)


if __name__ == '__main__':
    main()
