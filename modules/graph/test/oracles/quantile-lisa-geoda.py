# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Run with pygeoda==0.1.3; writes the actual CPU reference, not a translated oracle."""
import json
import pathlib
import tempfile

import pygeoda

assert pygeoda.__version__ == '0.1.3'

cases = [
    ('negative values and type-five quarter break', [-8, -6, -4, -2, 0, 2, 4, 6], 4, 1),
    ('interior ties go to higher class', [0, 0, 1, 1, 1, 1, 2, 2], 4, 2),
    ('tied class includes lower boundary', [0, 0, 1, 1, 1, 1, 2, 2], 4, 3),
    ('constant first class is empty', [5] * 8, 4, 1),
    ('constant final class contains every row', [5] * 8, 4, 4),
]
result = {
    'package': 'pygeoda==0.1.3',
    'sourceSemantics': 'GeoDaCenter/libgeoda@71238cb58979032048081434ce6e36a53e254280',
    'seed': 314159,
    'permutations': 9999,
    'significanceLevel': 0.1,
    'cases': [],
}
with tempfile.TemporaryDirectory() as directory:
    weights_path = pathlib.Path(directory) / 'cycle.gal'
    weights_path.write_text('0 8 luma id\n' + ''.join(
        f'{row + 1} 2\n{(row + 1) % 8 + 1} {(row + 7) % 8 + 1}\n'
        for row in range(8)))
    weights = pygeoda.read_gal(str(weights_path), [str(row + 1) for row in range(8)])
    for name, values, quantile_count, selected_quantile in cases:
        lisa = pygeoda.local_quantilelisa(
            weights, values, quantile_count, selected_quantile,
            permutations=result['permutations'], seed=result['seed'], cpu_threads=1,
            significance_cutoff=result['significanceLevel'])
        result['cases'].append({
            'name': name, 'observed': list(lisa.lisa_values()),
            'pseudoPValues': [value if value == value else None for value in lisa.lisa_pvalues()],
            'classes': list(lisa.lisa_clusters()),
        })
path = pathlib.Path(__file__).with_name('fixtures') / 'quantile-lisa-geoda.json'
path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
