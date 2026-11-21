// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {expect, it} from 'vitest';

it('PTS-144 counts distinct coincident rows in shell zero and excludes identity pairs', () => {
  // Execute the generator's calculation without running its unrelated mgwr/spreg fits.
  const script = `
import ast, json, sys
from pathlib import Path
import math
source = ast.parse(Path(sys.argv[1]).read_text())
function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'calculate_marked_statistics')
exec(compile(ast.Module(body=[function], type_ignores=[]), sys.argv[1], 'exec'))
# The central fixture has disjoint type masks and no coincident rows, so none of
# its pairs should change. Check exact counts and binary64-rounding-level statistics.
central = json.loads(Path(sys.argv[2]).read_text())
central_positions = list(zip(central['positions'][::2], central['positions'][1::2]))
central_marks = central['marked']['marks']
intensities = central['marked']['intensities']
point_types = central['marked']['types']
category_pairs = [[first == 0 and second == 1 for second in point_types] for first in point_types]
weights = [[25 / ((5-abs(first[0]-second[0])) * (5-abs(first[1]-second[1]))) /
            intensities[row] / intensities[other] for other, second in enumerate(central_positions)]
           for row, first in enumerate(central_positions)]
focus_marks = [mark for mark, kind in zip(central_marks, point_types) if kind == 0]
neighbor_marks = [mark for mark, kind in zip(central_marks, point_types) if kind == 1]
statistics, counts, _ = calculate_marked_statistics(
    [[math.dist(first, second) for second in central_positions] for first in central_positions],
    category_pairs, weights, [[first*second for second in central_marks] for first in central_marks],
    sum(focus_marks)/len(focus_marks) * sum(neighbor_marks)/len(neighbor_marks), 3, 25)
assert all(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-14, abs_tol=1e-14)
           for actual, expected in zip(statistics, central['marked']['statistics']))
assert len(statistics) == len(central['marked']['statistics'])
assert counts == central['marked']['counts']
positions = [[1, 1], [1, 1], [2.5, 1]]
marks = [1, 2, 3]
values, counts, shells = calculate_marked_statistics(
    [[math.dist(first, second) for second in positions] for first in positions],
    [[True]*3 for _ in positions], [[1]*3 for _ in positions],
    [[first*second for second in marks] for first in marks], (sum(marks)/len(marks))**2, 2, 25)
print(json.dumps(dict(counts=counts, shells=shells, correlations=values[3::4]), allow_nan=False))
`;
  const run = spawnSync(
    process.env['ORACLE_TEST_PYTHON'] ?? 'python3',
    [
      '-I',
      '-S',
      '-c',
      script,
      fileURLToPath(new URL('./geospatial-expansion-statistics.py', import.meta.url)),
      fileURLToPath(new URL('./fixtures/geospatial-expansion-statistics.json', import.meta.url))
    ],
    {encoding: 'utf8'}
  );
  expect(run.status, run.stderr).toBe(0);
  // Two directed (0,1)/(1,0) pairs have distance 0 and product 2. Four directed
  // pairs at distance 1.5 have products 3,6,3,6. Mean mark squared is 4:
  // shell correlations are (2+2)/2/4=.5 and (3+6+3+6)/4/4=1.125.
  expect(JSON.parse(run.stdout)).toEqual({
    counts: [2, 6],
    shells: [2, 4],
    correlations: [0.5, 1.125]
  });
});

it('B5 PTS-144 preserves cumulative estimates and nulls unavailable empty-shell correlations', () => {
  const script = `
import ast, json, sys
from pathlib import Path
source = ast.parse(Path(sys.argv[1]).read_text())
function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'calculate_marked_statistics')
exec(compile(ast.Module(body=[function], type_ignores=[]), sys.argv[1], 'exec'))
values, counts, shells = calculate_marked_statistics(
    [[0, 2], [2, 0]], [[True, True], [True, True]], [[1, 1], [1, 1]],
    [[1, 3], [3, 9]], 4, 2, 25)
print(json.dumps(dict(values=values, counts=counts, shells=shells), allow_nan=False))
`;
  const run = spawnSync(
    process.env['ORACLE_TEST_PYTHON'] ?? 'python3',
    [
      '-I',
      '-S',
      '-c',
      script,
      fileURLToPath(new URL('./geospatial-expansion-statistics.py', import.meta.url))
    ],
    {encoding: 'utf8'}
  );
  expect(run.status, run.stderr).toBe(0);
  // Marks 1,3 have mean squared 4. Radius 1 has no pairs; radius 2 has
  // two directed pairs of weight 1 and product 3: K=2/25, L=sqrt(K/pi),
  // g=2/(25*pi*(4-1)), k_mm=(3+3)/2/4=.75. Empty shell g/k_mm are unavailable.
  expect(JSON.parse(run.stdout)).toEqual({
    values: [0, 0, null, null, 0.08, 0.15957691216057307, 0.008488263631567752, 0.75],
    counts: [0, 2],
    shells: [0, 2]
  });
});
