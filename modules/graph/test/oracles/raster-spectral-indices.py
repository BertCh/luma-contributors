# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""RS09 oracle: python3.12, spyndex==0.12.0, numpy==2.2.6.

Run in an isolated environment; this generator never imports luma.gl or mirrors WGSL.
Spyndex's bundled (online=False) formula catalog supplies the independent index formulas.
Masks/calibration are an explicit adapter around its library calls. The Gao calculation
uses the algebraically identical NDMI formula with caller-supplied shortwave samples;
this is formula-composition evidence, not proof of identical sensor spectral responses.
"""

import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import spyndex

assert importlib.metadata.version('spyndex') == '0.12.0'
assert np.__version__ == '2.2.6'

INDEX_NAMES = ['ndwi-mcfeeters', 'ndwi-gao', 'nbr', 'evi', 'savi']
LIBRARY_NAMES = {'ndwi-mcfeeters': 'NDWI', 'ndwi-gao': 'NDMI', 'nbr': 'NBR', 'evi': 'EVI', 'savi': 'SAVI'}
REQUIRED_BANDS = {
    'ndwi-mcfeeters': ['green', 'nearInfrared'],
    'ndwi-gao': ['nearInfrared', 'shortwaveInfrared'],
    'nbr': ['nearInfrared', 'shortwaveInfrared2'],
    'evi': ['nearInfrared', 'red', 'blue'],
    'savi': ['nearInfrared', 'red'],
}
SYMBOLS = {'green': 'G', 'nearInfrared': 'N', 'shortwaveInfrared': 'S1', 'shortwaveInfrared2': 'S2', 'red': 'R', 'blue': 'B'}


def evaluate(index, bands, epsilon=0, soil_adjustment=0.5, clamp=None):
    parameters = {'g': 2.5, 'C1': 6.0, 'C2': 7.5, 'L': 1.0 if index == 'evi' else soil_adjustment}
    validity = np.ones(len(next(iter(bands.values()))['values']), dtype=bool)
    for name, band in bands.items():
        source = np.array(band['values'], dtype=np.float64)
        # Float inputs are stored in float32 buffers; the independent formula uses float64.
        if band['format'] == 'float32':
            source = source.astype(np.float32).astype(np.float64)
        sample = source * band.get('scale', 1) + band.get('offset', 0)
        parameters[SYMBOLS[name]] = sample
        validity &= np.isfinite(sample)
        if 'noDataValue' in band:
            validity &= source != band['noDataValue']
        if 'validity' in band:
            validity &= np.array(band['validity']) != 0
    if index == 'ndwi-mcfeeters':
        denominator = parameters['G'] + parameters['N']
    elif index == 'ndwi-gao':
        denominator = parameters['N'] + parameters['S1']
    elif index == 'nbr':
        denominator = parameters['N'] + parameters['S2']
    elif index == 'evi':
        denominator = parameters['N'] + 6 * parameters['R'] - 7.5 * parameters['B'] + 1
    else:
        denominator = parameters['N'] + parameters['R'] + soil_adjustment
    with np.errstate(divide='ignore', invalid='ignore'):
        values = spyndex.computeIndex(index=LIBRARY_NAMES[index], params=parameters, online=False)
    validity &= np.isfinite(denominator) & (np.abs(denominator) > epsilon) & np.isfinite(values)
    if clamp:
        values = np.clip(values, *clamp)
    return {'values': [float(value) if valid else None for value, valid in zip(values, validity)], 'validity': validity.astype(int).tolist()}


def make_case(name, index, bands, epsilon=0, soil_adjustment=0.5, clamp=None, updated_bands=None):
    count = len(next(iter(bands.values()))['values'])
    result = {'name': name, 'index': index, 'width': count // 2, 'height': 2, 'bands': bands, 'epsilon': epsilon, 'soilAdjustment': soil_adjustment}
    if clamp:
        result['clamp'] = clamp
    result['expected'] = evaluate(index, bands, epsilon, soil_adjustment, clamp)
    if updated_bands:
        result['updatedBands'] = updated_bands
        result['updatedExpected'] = evaluate(index, updated_bands, epsilon, soil_adjustment, clamp)
    return result


cases = []
raw_values = [18639, 24000, 35000, 10000, 0, 12000, 20000, 16000, 40000, 8000, 28000, 32000]
for index in INDEX_NAMES:
    bands = {}
    updated_bands = {}
    for ordinal, name in enumerate(REQUIRED_BANDS[index]):
        values = [value + ordinal * 400 if value else 0 for value in raw_values]
        validity = [1] * len(values)
        validity[7 + ordinal] = 0
        validity[1] = 7  # any nonzero quality mask is valid
        bands[name] = {'format': 'uint32', 'values': values, 'validity': validity, 'noDataValue': 0, 'scale': 0.0000275, 'offset': -0.2}
        updated_values = [value + 1500 if value else 0 for value in values]
        updated_validity = validity.copy()
        updated_validity[7 + ordinal] = 1
        updated_validity[2] = 0
        updated_bands[name] = {**bands[name], 'values': updated_values, 'validity': updated_validity}
    cases.append(make_case(f'{index}-landsat-calibration-and-updates', index, bands, epsilon=0.0001, updated_bands=updated_bands))

for index in INDEX_NAMES:
    if index == 'evi':
        samples = {'nearInfrared': [0.5, 0.625, 0.75, 0.375, 0.5, 0.5], 'red': [0.375, 0.375, 0.375, 0.375, 0.25, 0.25], 'blue': [0.5, 0.5, 0.5, 0.5, 0.5, 0.375]}
    elif index == 'savi':
        samples = {'nearInfrared': [0.5] * 6, 'red': [-1, -0.875, -0.75, -1.125, -1.25, -0.5]}
    else:
        first, second = REQUIRED_BANDS[index]
        samples = {first: [0.5] * 6, second: [-0.5, -0.375, -0.25, -0.625, -0.75, 0.5]}
    bands = {name: {'format': 'float32', 'values': values} for name, values in samples.items()}
    cases.append(make_case(f'{index}-denominator-boundaries', index, bands, epsilon=0.125))
    if index == 'evi':
        cases.append(make_case('evi-explicit-clamp', index, bands, epsilon=0.125, clamp=[-1, 1]))

savi_bands = {'nearInfrared': {'format': 'sint32', 'values': [8, 6, 4, 2, -2147483648, 12], 'scale': 0.0625, 'offset': 0.125, 'noDataValue': -2147483648}, 'red': {'format': 'uint32', 'values': [1, 2, 3, 4, 5, 6], 'scale': 0.03125, 'offset': 0.0625}}
for soil_adjustment in [0, 1]:
    cases.append(make_case(f'savi-soil-adjustment-{soil_adjustment}', 'savi', savi_bands, soil_adjustment=soil_adjustment))

fixture = {
    'generator': 'test/oracles/raster-spectral-indices.py',
    'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'spyndex': importlib.metadata.version('spyndex')},
    'reference': {'kind': 'direct spyndex.computeIndex calls with explicit calibration/mask adapter', 'catalog': 'bundled in spyndex 0.12.0; online=False', 'formulas': {name: spyndex.indices[name].formula for name in LIBRARY_NAMES.values()}, 'gaoMapping': 'NDMI supplies identical two-band algebra on caller-selected shortwave reflectance; not spectral-response equivalence or direct OTB execution'},
    'conventions': {'units': 'dimensionless reflectance and dimensionless index; no output integer scaling', 'invalid': 'canonical NaN and validity zero; JSON represents invalid expected values as null', 'denominator': 'abs(denominator) > epsilon, including equality rejection; signed denominators permitted', 'calibration': 'raw * scale + offset; raw nodata and per-band masks retained', 'absoluteTolerance': 0.00001, 'relativeTolerance': 0.00001, 'toleranceReason': 'float32 calibration and multi-pass arithmetic versus independent float64 evaluation; valid stress denominators bounded away from zero'},
    'cases': cases,
}
path = Path(__file__).parent / 'fixtures' / 'raster-spectral-indices.json'
path.write_text(json.dumps(fixture, indent=2, allow_nan=False) + '\n')
print(f'Wrote {path}: {len(cases)} independent cases')
