# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent NumPy FFT oracle. Run with /usr/bin/python3 from the repository root."""
import json
from pathlib import Path
import numpy

random = numpy.random.default_rng(7407)
width, height = 16, 8
values = random.normal(size=(height, width)).astype(numpy.float32)
frequency_columns, frequency_rows = numpy.meshgrid(numpy.fft.fftfreq(width), numpy.fft.fftfreq(height))
frequency_radius = numpy.hypot(frequency_columns, frequency_rows)
settings = dict(cutoff=0.25, lowerCutoff=0.125, upperCutoff=0.375,
                sigma=0.14, notchCenter=[0.25, 0.125], notchRadius=0.07)


def filter_case(name, filter_name, changes=None):
    current = dict(settings, **(changes or {}))
    if filter_name == 'low-pass':
        response = frequency_radius <= current['cutoff']
    elif filter_name == 'high-pass':
        response = frequency_radius > current['cutoff']
    elif filter_name == 'band-pass':
        response = ((frequency_radius >= current['lowerCutoff']) &
                    (frequency_radius <= current['upperCutoff']))
    elif filter_name == 'gaussian':
        response = numpy.exp(-0.5 * (frequency_radius / current['sigma']) ** 2)
    else:
        distances = []
        for sign in [1, -1]:
            distance_columns = numpy.abs(frequency_columns - sign * current['notchCenter'][0])
            distance_rows = numpy.abs(frequency_rows - sign * current['notchCenter'][1])
            distances.append(numpy.hypot(numpy.minimum(distance_columns, 1 - distance_columns),
                                      numpy.minimum(distance_rows, 1 - distance_rows)))
        response = numpy.minimum(*distances) > current['notchRadius']
    # Band calibration is applied before the transform, matching the raster-band contract.
    calibrated = (values * numpy.float32(1.5) + numpy.float32(-2)).astype(numpy.float32)
    expected = numpy.fft.ifft2(numpy.fft.fft2(calibrated) * response).real.astype(numpy.float32)
    return dict(name=name, filter=filter_name, settings=current, expected=expected.ravel().tolist())


filters = [filter_case(name, name) for name in ['low-pass', 'high-pass', 'band-pass', 'gaussian', 'notch']]
filters += [filter_case('dc-only', 'low-pass', dict(cutoff=0)),
            filter_case('remove-dc', 'high-pass', dict(cutoff=0)),
            filter_case('all-pass', 'low-pass', dict(cutoff=float(numpy.sqrt(0.5)))),
            filter_case('empty-pass', 'high-pass', dict(cutoff=float(numpy.sqrt(0.5)))),
            filter_case('single-ring', 'band-pass', dict(lowerCutoff=0.25, upperCutoff=0.25)),
            filter_case('nyquist-notch', 'notch', dict(notchCenter=[-0.5, 0], notchRadius=0)),
            filter_case('notch-all', 'notch', dict(notchCenter=[0, 0], notchRadius=float(numpy.sqrt(0.5))))]


def periodic_case(name, reference, shift, expected_status):
    rows, columns = reference.shape
    frequency_columns, frequency_rows = numpy.meshgrid(numpy.fft.fftfreq(columns), numpy.fft.fftfreq(rows))
    moving = numpy.fft.ifft2(numpy.fft.fft2(reference) * numpy.exp(-2j * numpy.pi *
        (frequency_columns * shift[0] + frequency_rows * shift[1]))).real.astype(numpy.float32)
    product = numpy.fft.fft2(reference) * numpy.conj(numpy.fft.fft2(moving))
    magnitude = numpy.abs(product)
    scale = numpy.max(magnitude.ravel()[1:])
    eligible = magnitude > max(1e-20, scale * 1e-6)
    eligible[0, 0] = False
    cross_power = numpy.zeros_like(product)
    cross_power[eligible] = product[eligible] / magnitude[eligible]
    correlation = numpy.fft.ifft2(cross_power).real
    selected = numpy.argmax(correlation)
    peak_row, peak_column = divmod(int(selected), columns)
    row_distance = numpy.abs(numpy.arange(rows) - peak_row)
    column_distance = numpy.abs(numpy.arange(columns) - peak_column)
    nonadjacent = ((numpy.minimum(row_distance, rows - row_distance)[:, None] > 1) |
                   (numpy.minimum(column_distance, columns - column_distance)[None, :] > 1))
    maximum = correlation.ravel()[selected]
    secondary = numpy.max(correlation[nonadjacent]) if nonadjacent.any() else 0
    ratio = float(numpy.clip(secondary / maximum, 0, 1)) if maximum > 0 else None
    return dict(name=name, width=columns, height=rows, reference=reference.ravel().tolist(),
                moving=moving.ravel().tolist(), status=expected_status, secondPeakRatio=ratio,
                shift=[-shift[0], -shift[1]])

# Suppress Nyquist axes before a fractional Fourier shift so both images remain real.
spectrum = numpy.fft.fft2(random.normal(size=(8, 16)))
spectrum[4, :] = 0
spectrum[:, 8] = 0
reference = numpy.fft.ifft2(spectrum).real.astype(numpy.float32)
registration = [periodic_case('fractional', reference, [1.25, -0.5], 0),
                periodic_case('repeated', numpy.tile(random.normal(size=(8, 4)), (1, 4)).astype(numpy.float32), [1, 2], 2),
                periodic_case('flat', numpy.ones((4, 4), dtype=numpy.float32), [0, 0], 1),
                periodic_case('no-nonadjacent-peak', numpy.array([[0, 1], [3, -2]], dtype=numpy.float32), [0, 0], 0)]


def masked_case(name, reference, moving, maximum_shift):
    rows, columns = reference.shape
    scores = []
    shifts = []
    for row_shift in range(-maximum_shift[1], maximum_shift[1] + 1):
        for column_shift in range(-maximum_shift[0], maximum_shift[0] + 1):
            first, second = [], []
            for row in range(rows):
                for column in range(columns):
                    moving_row, moving_column = row - row_shift, column - column_shift
                    if 0 <= moving_row < rows and 0 <= moving_column < columns:
                        first.append(reference[row, column])
                        second.append(moving[moving_row, moving_column])
            first, second = numpy.asarray(first, dtype=numpy.float64), numpy.asarray(second, dtype=numpy.float64)
            first, second = first - first.mean(), second - second.mean()
            score = float(numpy.dot(first, second) / numpy.sqrt(numpy.dot(first, first) * numpy.dot(second, second)))
            scores.append(score)
            shifts.append((column_shift, row_shift))
    selected = int(numpy.argmax(scores))
    best = scores[selected]
    secondary = [score for score, shift in zip(scores, shifts)
                 if max(abs(shift[0] - shifts[selected][0]), abs(shift[1] - shifts[selected][1])) > 1]
    ratio = max(0, max(secondary, default=0)) / best
    return dict(name=name, width=columns, height=rows, reference=reference.ravel().tolist(),
                moving=moving.ravel().tolist(), maximumShift=maximum_shift, shift=shifts[selected],
                secondPeakRatio=ratio, status=2 if ratio >= 1 - 1e-4 else 0)

masked_reference = random.normal(size=(8, 8)).astype(numpy.float32)
masked = [masked_case('masked-unique', masked_reference, masked_reference, [2, 2]),
          masked_case('masked-single-candidate', masked_reference, masked_reference, [0, 0]),
          masked_case('masked-repeated', numpy.tile(numpy.array([0, 1, 0, -1], dtype=numpy.float32), (8, 2)),
                      numpy.tile(numpy.array([0, 1, 0, -1], dtype=numpy.float32), (8, 2)), [2, 2])]
calibrated = (values * numpy.float32(1.5) + numpy.float32(-2)).astype(numpy.float32).ravel()
by_name = {case['name']: numpy.asarray(case['expected']) for case in filters}
numpy.testing.assert_allclose(by_name['low-pass'] + by_name['high-pass'], calibrated, atol=1e-6)
numpy.testing.assert_allclose(by_name['all-pass'], calibrated, atol=1e-6)
numpy.testing.assert_allclose(by_name['dc-only'], calibrated.mean(dtype=numpy.float64), atol=1e-6)
numpy.testing.assert_allclose(by_name['empty-pass'], 0, atol=1e-6)
numpy.testing.assert_allclose(by_name['notch-all'], 0, atol=1e-6)
fixture = dict(oracle='NumPy ' + numpy.__version__, width=width, height=height,
               input=values.ravel().tolist(), filters=filters, registration=registration, masked=masked)
output = Path(__file__).parent / 'fixtures/raster-spectral-filter.json'
output.write_text(json.dumps(fixture, indent=2, allow_nan=False) + '\n')
print(f'Wrote {output}: {len(filters)} filters, {len(registration)} periodic and {len(masked)} masked registration cases')
