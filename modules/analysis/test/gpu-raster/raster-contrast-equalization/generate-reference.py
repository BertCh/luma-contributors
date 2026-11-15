# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Generate small CART-03 oracles with /usr/bin/python3 (NumPy/SciPy/scikit-image).

Ordinary CLAHE cases use skimage.exposure._adapthist._clahe for the raw map and
exposure.equalize_adapthist for the independently rescaled public-API comparison.
The GPU omits the final integer interpolation truncation (<= 1/16383 error) and
public API's global input/output rescale. Fixed-domain inputs avoid ambiguity.
Masked cases use skimage clip_histogram/map_histogram and SciPy interpolation;
scikit-image's public CLAHE has no mask. Matching uses public match_histograms
on quantized, valid samples. No implementation output is used as an oracle.
"""
import json
from pathlib import Path

import numpy
import scipy
from scipy.ndimage import map_coordinates
import skimage
from skimage import exposure
from skimage.exposure._adapthist import _clahe, clip_histogram, map_histogram


def quantize_clahe(values, bins):
    return numpy.rint(numpy.clip(values, 0, 1) * 16383).astype(numpy.uint16) // (1 + 16384 // bins)


def masked_clahe(values, validity, tile_height, tile_width, bins, clip_limit):
    height, width = values.shape
    tile_rows = (height + tile_height - 1) // tile_height
    tile_columns = (width + tile_width - 1) // tile_width
    padding = ((0, tile_rows * tile_height - height), (0, tile_columns * tile_width - width))
    quantized = quantize_clahe(numpy.nan_to_num(values), bins)
    padded = numpy.pad(quantized, padding, mode='reflect')
    mask = numpy.pad(validity, padding, mode='reflect')
    mappings = numpy.zeros((tile_rows, tile_columns, bins), dtype=float)
    occupied = numpy.zeros((tile_rows, tile_columns), dtype=float)
    for row in range(tile_rows):
        for column in range(tile_columns):
            region = (slice(row * tile_height, (row + 1) * tile_height),
                      slice(column * tile_width, (column + 1) * tile_width))
            samples = padded[region][mask[region]]
            if not samples.size:
                continue
            occupied[row, column] = 1
            histogram = numpy.bincount(samples, minlength=bins)
            limit = max(1, int(clip_limit * samples.size)) if clip_limit > 0 else samples.size
            clipped = clip_histogram(histogram, limit)
            mappings[row, column] = map_histogram(clipped, 0, 16383, samples.size) / 16383
    rows, columns = numpy.indices((height, width))
    coordinates = [(rows + tile_height // 2) / tile_height - 1,
                   (columns + tile_width // 2) / tile_width - 1]
    weights = map_coordinates(occupied, coordinates, order=1, mode='nearest')
    result = numpy.zeros((height, width))
    for bin_index in range(bins):
        interpolated = map_coordinates(mappings[:, :, bin_index], coordinates, order=1, mode='nearest')
        selected = quantized == bin_index
        result[selected] = interpolated[selected] / numpy.maximum(weights[selected], 1e-30)
    result[~validity | (weights == 0)] = numpy.nan
    return result


def serialize(values):
    return [float(value) if numpy.isfinite(value) else None for value in numpy.asarray(values).ravel()]


cases = []
random = numpy.random.default_rng(3103)
for name, height, width, tile_height, tile_width, bins, clip_limit in [
    ('odd-reflected', 5, 7, 3, 4, 16, 0.25),
    ('odd-reflected-unclipped', 5, 7, 3, 4, 16, 0.0),
    ('odd-reflected-low-clip', 5, 7, 3, 4, 16, 0.01),
    ('full-tiles', 8, 8, 4, 4, 256, 0.2),
    ('saturated-redistribution', 8, 8, 8, 8, 3, 21/64),
    ('deep-redistribution', 8, 8, 8, 8, 3, 22/64),
    ('two-bins', 4, 8, 2, 4, 2, 0.75),
    ('one-column', 7, 1, 3, 4, 16, 1.0),
    ('one-cell', 1, 1, 1, 1, 256, 0.01),
    ('constant', 3, 5, 4, 3, 16, 0.2),
]:
    values = random.integers(0, 16, (height, width)).astype(float) / 15
    values.flat[0], values.flat[-1] = 0, 1
    if name.startswith('odd-reflected'):
        values = ((numpy.arange(height * width).reshape(height, width) * 7) % 16) / 15
        values.flat[0], values.flat[-1] = 0, 1
    if name in ('saturated-redistribution', 'deep-redistribution'):
        values.flat[:46], values.flat[46:] = 0, 1
    if name in ('constant', 'one-cell'):
        values[:] = 0.5
    integers = numpy.rint(values * 16383).astype(numpy.uint16)
    expected = _clahe(integers, [tile_height, tile_width], clip_limit, bins).astype(float) / 16383
    public = exposure.equalize_adapthist(values, kernel_size=(tile_height, tile_width),
                                        clip_limit=clip_limit, nbins=bins)
    cases.append(dict(name=name, mode='clahe', height=height, width=width,
                      tileHeight=tile_height, tileWidth=tile_width, binCount=bins,
                      clipLimit=clip_limit, values=serialize(values), expected=serialize(expected),
                      equalizeAdapthist=serialize(public)))

for name in ['masked-empty-tiles', 'all-invalid']:
    values = numpy.arange(48).reshape(6, 8).astype(float) / 47
    validity = numpy.zeros_like(values, dtype=bool)
    if name == 'masked-empty-tiles':
        validity[0:3, 0:4] = True
        validity[4:, 5:] = True
        validity[1, 2] = False
        values[2, 2] = numpy.nan
        validity &= numpy.isfinite(values)
    expected = masked_clahe(values, validity, 3, 4, 16, 0.25)
    cases.append(dict(name=name, mode='clahe', height=6, width=8, tileHeight=3, tileWidth=4,
                      binCount=16, clipLimit=0.25, values=serialize(values),
                      mask=validity.ravel().astype(int).tolist(), expected=serialize(expected)))

for name, source, reference, source_mask, reference_mask, bins in [
    ('matching-ties-gaps', [0, 0, 0.2, 0.2, 0.6, 1, 1, 1], [0, 0.4, 0.4, 0.8, 1], None, None, 16),
    ('matching-half-bin-ties', [0, 1/30, 0.1, 0.5, 29/30, 1], [0, 0.2, 0.8, 1], None, None, 16),
    ('matching-masks', [0, 0.4, 0.4, 0.6, 0.8, 1], [0, 0.2, 0.2, 1], [1, 0, 1, 1, 0, 1], [0, 1, 1, 0], 16),
    ('matching-empty-reference', [0, 0.2, 0.8, 1], [0, 1], None, [0, 0], 256),
    ('matching-empty-source', [0, 1], [0, 0.2, 0.8, 1], [0, 0], None, 256),
    ('matching-constant', [0.5, 0.5, 0.5], [0.2, 0.2, 0.2, 0.2], None, None, 2),
]:
    values = numpy.asarray(source, dtype=numpy.float32)
    reference_values = numpy.asarray(reference, dtype=numpy.float32)
    source_validity = numpy.ones(values.size, bool) if source_mask is None else numpy.asarray(source_mask, bool)
    reference_validity = numpy.ones(reference_values.size, bool) if reference_mask is None else numpy.asarray(reference_mask, bool)
    quantized = numpy.floor(values * numpy.float32(bins - 1) + numpy.float32(0.5)).astype(int) / (bins - 1)
    quantized_reference = numpy.floor(reference_values * numpy.float32(bins - 1) + numpy.float32(0.5)).astype(int) / (bins - 1)
    expected = numpy.full(values.shape, numpy.nan)
    if source_validity.any() and reference_validity.any():
        expected[source_validity] = exposure.match_histograms(quantized[source_validity], quantized_reference[reference_validity])
    cases.append(dict(name=name, mode='match-histograms', width=values.size, height=1,
                      binCount=bins, clipLimit=0.01, values=serialize(values),
                      reference=serialize(reference_values), mask=source_validity.astype(int).tolist(),
                      referenceMask=reference_validity.astype(int).tolist(), expected=serialize(expected)))

# Store only counts and two oracle outputs, never the large synthetic rasters.
# uint8 inputs make public match_histograms use its bincount route at these declared limits.
source_histogram = [16777215, 1]
reference_histogram = [12582911, 1]
large_source = numpy.zeros(sum(source_histogram), dtype=numpy.uint8)
large_reference = numpy.zeros(sum(reference_histogram), dtype=numpy.uint8)
large_source[-1] = 1
large_reference[-1] = 1
large_matched = exposure.match_histograms(large_source, large_reference)
large_histogram_matching = dict(sourceHistogram=source_histogram, referenceHistogram=reference_histogram,
                                expected=[float(large_matched[0]), float(large_matched[-1])])

output = dict(oracle=dict(skimage=skimage.__version__, numpy=numpy.__version__, scipy=scipy.__version__), cases=cases, largeHistogramMatching=large_histogram_matching)
Path(__file__).with_name('reference.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
print(f'Wrote {len(cases)} cases')
