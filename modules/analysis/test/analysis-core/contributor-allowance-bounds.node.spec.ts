// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {expect, it} from 'vitest';
import {getRegressionTileLayout} from '../../src/gpu-spatial-analysis/spatial-regression/regression-row-tiles';
import {getSpaceTimeStoppingBatchSize} from '../../src/gpu-spatial-analysis/space-time-tests/space-time-stopping';
import {getDistanceFieldJumpFloodSteps} from '../../src/raster-analysis/distance-field/distance-field-kernels';
import {
  getDelaunayParallelStepCount,
  getDelaunayStepCount,
  getDelaunayScratchTriangleCapacity
} from '../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation';

type Allowance = {description: string; reason: string};
const baseline = JSON.parse(readFileSync('scripts/contributor-contract-baseline.json', 'utf8'))[
  'single-invocation-data-loop'
] as Record<string, Allowance[]>;
const sourceRoot = 'modules/analysis/src/';
function getAllowance(file: string, label: string): Allowance {
  const allowance = baseline[sourceRoot + file]?.find(entry => entry.description.includes(label));
  if (!allowance) throw new Error(`missing allowance ${file} ${label}`);
  return allowance;
}

it('invalidates the actual allowances on host-cap changes, including powers and aliases (CORE-118)', () => {
  // Execute the checker's own bound resolver over the real source trees, without editing sources
  // or running its CLI. Host-only mutations leave every baselined shader/site identity untouched.
  let source = readFileSync('scripts/check-contributor-contract.mjs', 'utf8');
  source = source.slice(0, source.indexOf('  for (const file of checkedFiles) {'));
  source = source.replace(
    /from '(\.[^']+)'/g,
    (_match, path: string) => `from '${pathToFileURL(resolve('scripts', path)).href}'`
  );
  const checks: (string | number | boolean)[][] = [
    ...[
      'temporal-reduction/temporal-reduction-sorted-range.ts',
      'time-window-filter/time-window-sorted-nodes.ts'
    ].flatMap(file => [
      [
        'dataframe-analysis/' + file,
        'kernel',
        'SORTED_TIME_RANGE_MAXIMUM_SEARCH_STEPS',
        'dataframe-analysis/temporal-reduction/sorted-range-bounds.ts',
        32,
        31
      ],
      [
        'dataframe-analysis/' + file,
        'kernel',
        'SORTED_TIME_RANGE_MAXIMUM_ROW_COUNT',
        'dataframe-analysis/temporal-reduction/sorted-range-bounds.ts',
        '0xffffffff',
        '0x7fffffff'
      ]
    ]),
    ...['gpu-jacquez-test.ts', 'gpu-knox-test.ts', 'space-time-stopping.ts'].map(file => [
      'gpu-spatial-analysis/space-time-tests/' + file,
      file === 'space-time-stopping.ts' ? 'batch-' : 'summary',
      'SPACE_TIME_MAXIMUM_PERMUTATIONS',
      'gpu-spatial-analysis/space-time-tests/space-time-kernels.ts',
      '2 ** 16',
      '2 ** 17'
    ]),
    [
      'dataframe-analysis/column-classification/exact-natural-breaks-nodes.ts',
      'backtrack',
      'MAXIMUM_CLASS_COUNT',
      'dataframe-analysis/column-classification/gpu-class-breaks.ts',
      64,
      65
    ],
    [
      'dataframe-analysis/column-classification/monotone-natural-breaks-nodes.ts',
      'backtrack',
      'MAXIMUM_CLASS_COUNT',
      'dataframe-analysis/column-classification/gpu-class-breaks.ts',
      64,
      65
    ],
    ...[
      'prefix',
      'backtrack',
      'kernel ["\\u0000"]',
      'bulk-switch',
      'bulk-backtrack',
      'bulk-assemble',
      'advance',
      'finalize'
    ].map(label => [
      'dataframe-analysis/column-classification/natural-breaks-nodes.ts',
      label,
      label === 'prefix' ? 'MAXIMUM_NATURAL_BREAKS_BIN_COUNT' : 'MAXIMUM_CLASS_COUNT',
      'dataframe-analysis/column-classification/gpu-class-breaks.ts',
      label === 'prefix' ? 2048 : 64,
      label === 'prefix' ? 2049 : 65
    ]),
    [
      'gpu-spatial-analysis/spatial-validation/gpu-area-of-applicability.ts',
      'mean-distance',
      'GPU_AREA_OF_APPLICABILITY_MAXIMUM_PREDICTOR_COUNT',
      'gpu-spatial-analysis/spatial-validation/gpu-area-of-applicability.ts',
      16,
      17
    ]
  ];
  checks.push(
    [
      'gpu-spatial-analysis/analysis-spacing/analysis-spacing-statistics-nodes.ts',
      'search-parameters',
      'GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH',
      'gpu-spatial-analysis/neighbor-search/neighbor-search-parameters.ts',
      12,
      13
    ],
    [
      'dataframe-analysis/pair-statistics/gpu-point-pattern-indices.ts',
      'quadrat-statistics',
      'GPU_QUADRAT_MAXIMUM_COUNT',
      'dataframe-analysis/pair-statistics/point-pattern-indices-parameters.ts',
      65536,
      65537
    ],
    [
      'dataframe-analysis/pair-statistics/gpu-point-pattern-indices.ts',
      'quadrat-statistics',
      'GAMMA_MAXIMUM_ITERATIONS',
      'dataframe-analysis/pair-statistics/gpu-point-pattern-indices.ts',
      500,
      501
    ],
    [
      'dataframe-analysis/pair-statistics/gpu-point-pattern-indices.ts',
      'quadrat-statistics',
      'GAMMA_CENTERED_SERIES_MAXIMUM_ITERATIONS',
      'dataframe-analysis/pair-statistics/gpu-point-pattern-indices.ts',
      2048,
      2049
    ],
    [
      'gpu-spatial-analysis/spatial-regionalization/ward-mutual-selection.ts',
      'selection-choose',
      'GPU_WARD_MUTUAL_SELECTION_RADIX_BIN_COUNT',
      'gpu-spatial-analysis/spatial-regionalization/ward-mutual-selection.ts',
      256,
      128
    ],
    ...['solve', 'refine'].map(label => [
      'gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares.ts',
      label,
      'GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_MAXIMUM_CENTERED_INSTRUMENT_COLUMN_COUNT',
      'gpu-spatial-analysis/spatial-regression/spatial-two-stage-least-squares-parameters.ts',
      16,
      18
    ]),
    [
      'gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-regression.ts',
      'within-finish',
      'GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_MAXIMUM_PREDICTOR_COUNT',
      'gpu-spatial-analysis/spatial-regression/spatial-two-stage-least-squares-parameters.ts',
      8,
      9
    ],
    [
      'raster-analysis/cost-distance/gpu-cost-distance-path.ts',
      'walk',
      'MAXIMUM_COST_DISTANCE_PATH_REFERENCE_CELLS',
      'raster-analysis/cost-distance/gpu-cost-distance-path.ts',
      256,
      257
    ],
    [
      'raster-analysis/distance-field/distance-field-kernels.ts',
      'jump-flood-steps',
      'GPU_DISTANCE_FIELD_MAXIMUM_DIMENSION',
      'raster-analysis/distance-field/distance-field-parameters.ts',
      32768,
      65536
    ],
    [
      'raster-analysis/distance-field/distance-field-kernels.ts',
      'jump-flood-steps',
      'GPU_DISTANCE_FIELD_MAXIMUM_REFINEMENT_COUNT',
      'raster-analysis/distance-field/distance-field-parameters.ts',
      2,
      3
    ],
    [
      'raster-analysis/raster-stretch/gpu-raster-stretch.ts',
      'finalize',
      'GPU_RASTER_STRETCH_MAXIMUM_RANK_SEARCH_STEPS',
      'raster-analysis/raster-stretch/gpu-raster-stretch.ts',
      16,
      17
    ]
  );
  checks.push(
    ...['solve', 'refine'].flatMap(label => [
      [
        'gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares.ts',
        label,
        'SYMMETRIC_EIGENVALUE_SWEEP_COUNT',
        'gpu-spatial-analysis/spatial-regression/spatial-regression-solve.ts',
        16,
        17,
        false
      ],
      [
        'gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares.ts',
        label,
        'GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_MAXIMUM_PREDICTOR_COUNT',
        'gpu-spatial-analysis/spatial-regression/spatial-two-stage-least-squares-parameters.ts',
        8,
        9,
        false
      ]
    ])
  );
  const probe =
    source +
    `
    const baseline = JSON.parse(fs.readFileSync(BASELINE_PATH, 'utf8'))['single-invocation-data-loop'];
    const results = [];
    for (const [suffix, label, name, hostSuffix, before, after, expectedInvalidation = true] of ${JSON.stringify(checks)}) {
      const file = SOURCE_ROOT + '/' + suffix;
      const hostFile = SOURCE_ROOT + '/' + hostSuffix;
      const entry = baseline[file].find(item => item.description.includes(label));
      if (!entry) throw new Error('missing site ' + suffix + ' ' + label);
      const originalText = fileTexts.get(hostFile);
      const originalTree = sourceTrees.get(hostFile);
      const replacement = originalText.replace(name + ' = ' + before, name + ' = ' + after);
      if (replacement === originalText) throw new Error('mutation missed ' + name);
      const unchanged = getCitedBoundProblems(file, entry.reason);
      const shaderTexts = tree => { const texts = []; const visit = node => { if (typeScript.isTemplateLiteral(node)) texts.push(node.getText()); typeScript.forEachChild(node, visit); }; visit(tree); return texts; };
      const shadersBefore = shaderTexts(originalTree);
      fileTexts.set(hostFile, replacement);
      sourceTrees.set(hostFile, typeScript.createSourceFile(hostFile, replacement, typeScript.ScriptTarget.Latest, true));
      topLevelConstants = undefined;
      const changed = getCitedBoundProblems(file, entry.reason);
      if (JSON.stringify(shadersBefore) !== JSON.stringify(shaderTexts(sourceTrees.get(hostFile)))) throw new Error('mutation changed shader fingerprints');
      results.push({label: suffix + ' ' + label + ' ' + name, unchanged, changed, expectedInvalidation});
      fileTexts.set(hostFile, originalText);
      sourceTrees.set(hostFile, originalTree);
      topLevelConstants = undefined;
    }
    process.stdout.write(JSON.stringify(results));
  }
  runContributorContract();`;
  const result = spawnSync(process.execPath, ['--input-type=module', '-e', probe], {
    encoding: 'utf8',
    timeout: 30000,
    maxBuffer: 1024 * 1024
  });
  expect(result.status, result.stderr).toBe(0);
  const results = JSON.parse(result.stdout) as {
    label: string;
    unchanged: string[];
    changed: string[];
    expectedInvalidation: boolean;
  }[];
  for (const result of results) {
    expect(result.unchanged, result.label).toEqual([]);
    if (result.label.endsWith('SORTED_TIME_RANGE_MAXIMUM_SEARCH_STEPS')) {
      expect(result.changed).toEqual([
        'cites SORTED_TIME_RANGE_MAXIMUM_SEARCH_STEPS = 32, but the source value is 31'
      ]);
    }
    if (result.label.endsWith('SORTED_TIME_RANGE_MAXIMUM_ROW_COUNT')) {
      expect(result.changed).toEqual([
        'cites SORTED_TIME_RANGE_MAXIMUM_ROW_COUNT = 4294967295, but the source value is 2147483647'
      ]);
    }

    if (result.label.includes('space-time-tests/')) {
      expect(result.changed).toEqual([
        'cites SPACE_TIME_MAXIMUM_PERMUTATIONS = 65536, but the source value is 131072'
      ]);
    }

    if (!result.expectedInvalidation) {
      expect(
        result.changed,
        `${result.label}: unrelated host value must not invalidate this route`
      ).toEqual([]);
      continue;
    }
    expect(
      result.changed.length,
      `${result.label}: host cap must invalidate the allowance`
    ).toBeGreaterThan(0);
  }
  expect(
    getAllowance(
      'gpu-spatial-analysis/spatial-validation/gpu-area-of-applicability.ts',
      'threshold'
    ).reason
  ).not.toContain('GPU_AREA_OF_APPLICABILITY_MAXIMUM_PREDICTOR_COUNT');
});

it.each([
  [
    'gpu-spatial-analysis/analysis-spacing/analysis-spacing-statistics-nodes.ts',
    'search-parameters',
    /12.*parameter|parameter.*12/i,
    /GPU_ANALYSIS_MAXIMUM_NEIGHBOR_RANK/
  ],
  [
    'gpu-spatial-analysis/point-process/gpu-point-pattern-envelope.ts',
    '-test',
    /MAXIMUM_SIMULATION_COUNT = 4096/,
    /MAXIMUM_CURVE_ROW_COUNT/
  ],
  [
    'gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression.ts',
    'summarize',
    /4096|4,096/,
    /65536 tiles/
  ]
] as const)('records the correct serial axis for %s %s', (file, label, expected, unrelated) => {
  const {reason} = getAllowance(file, label);
  expect(reason).toMatch(expected);
  expect(reason).not.toMatch(unrelated);
});

it.each([
  [
    'gpu-network/network-centrality/network-centrality-passes.ts',
    'createCentralityConvergedNode',
    'GPU_MAXIMUM_COMMAND_NODE_COUNT = 32768'
  ],
  ['gpu-spatial-analysis/space-time-tests/space-time-stopping.ts', 'batch-', '512'],
  ['raster-analysis/distance-field/distance-field-kernels.ts', 'jump-flood-steps', '17']
])('records route or layout bounds for %s %s', (file, label, citation) => {
  expect(getAllowance(file, label).reason).toContain(citation);
});

it('pins the stopping, jump-flood and continuation route formulas (B6, S2)', () => {
  // 65537/256 rounds up to 257 batches; doubling 64 to 512 is required to stay <=256.
  expect(getSpaceTimeStoppingBatchSize(65537)).toBe(512);
  // 32768 = 2^15, so the halving jump list has 15 entries plus two refinement entries.
  expect(getDistanceFieldJumpFloodSteps({width: 32768, height: 32768}, 2)).toHaveLength(17);
  expect(getDelaunayParallelStepCount(2048)).toBeLessThan(getDelaunayStepCount(2048));
  expect(getDelaunayParallelStepCount(2049)).toBe(getDelaunayStepCount(2049));
  // Capacity = 2 * sites + max(groups,1) + 7; groups are independent of the site cutoff.
  expect(getDelaunayScratchTriangleCapacity('fixture', 2048)).toBe(4104);
  expect(getDelaunayScratchTriangleCapacity('fixture', 2048, 1000000)).toBe(1004103);
  // At 2048 sites one group gives 4104 triangles, two groups give 4105. Both axes gate serial work.
  expect(getDelaunayParallelStepCount(2048, 1)).toBeLessThan(getDelaunayStepCount(2048));
  expect(getDelaunayParallelStepCount(2048, 2)).toBe(getDelaunayStepCount(2048));
  expect(getDelaunayParallelStepCount(2048, 1000000)).toBe(getDelaunayStepCount(2048));
});

it.each([
  ['dataframe-analysis/temporal-reduction/temporal-reduction-sorted-range.ts', true],
  ['dataframe-analysis/time-window-filter/time-window-sorted-nodes.ts', false]
])('cites supplied sorted-search bounds at %s (review B2)', (file, hasWordDivision) => {
  const reason = getAllowance(file, 'kernel').reason;
  expect(reason).toContain('SORTED_TIME_RANGE_MAXIMUM_SEARCH_STEPS = 32');
  expect(reason).toContain('SORTED_TIME_RANGE_MAXIMUM_ROW_COUNT = 4294967295');
  if (hasWordDivision) expect(reason).toMatch(/64.*division|division.*64/);
  expect(reason).not.toContain('No named host bound');
});

it.each([
  'gpu-spatial-analysis/trajectory-restriction/trajectory-restriction-kernels.ts',
  'gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-segments.ts'
])('retires the removed serial allowance at %s (review B3)', file => {
  expect(baseline[sourceRoot + file]).toBeUndefined();
});

it.each([
  [
    'dataframe-analysis/pair-statistics/gpu-point-pattern-indices.ts',
    'quadrat-statistics',
    [
      'GPU_QUADRAT_MAXIMUM_COUNT = 65536',
      'GAMMA_MAXIMUM_ITERATIONS = 500',
      'GAMMA_CENTERED_SERIES_MAXIMUM_ITERATIONS = 2048'
    ]
  ],
  [
    'gpu-spatial-analysis/analysis-spacing/analysis-spacing-statistics-nodes.ts',
    'search-parameters',
    ['GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH = 12']
  ],
  [
    'gpu-spatial-analysis/spatial-regionalization/ward-mutual-selection.ts',
    'selection-choose',
    ['GPU_WARD_MUTUAL_SELECTION_RADIX_BIN_COUNT = 256']
  ],
  [
    'gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-regression.ts',
    'within-finish',
    [
      'GPU_PANEL_REGRESSION_MAXIMUM_TABLE_ROW_COUNT = 10',
      'GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE = 4',
      'GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH = 16'
    ]
  ],
  ...['solve', 'refine'].map(label => [
    'gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares.ts',
    label,
    ['GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_MAXIMUM_CENTERED_INSTRUMENT_COLUMN_COUNT = 16']
  ]),
  [
    'raster-analysis/cost-distance/gpu-cost-distance-path.ts',
    'walk',
    ['MAXIMUM_COST_DISTANCE_PATH_REFERENCE_CELLS = 256']
  ],
  [
    'raster-analysis/distance-field/distance-field-kernels.ts',
    'jump-flood-steps',
    [
      'GPU_DISTANCE_FIELD_MAXIMUM_DIMENSION = 32768',
      'GPU_DISTANCE_FIELD_MAXIMUM_REFINEMENT_COUNT = 2'
    ]
  ],
  [
    'raster-analysis/raster-stretch/gpu-raster-stretch.ts',
    'finalize',
    [
      'GPU_RASTER_STRETCH_MAXIMUM_RANK_SEARCH_STEPS = 16',
      'GPU_RASTER_STRETCH_MAXIMUM_HISTOGRAM_BIN_COUNT = 65536'
    ]
  ]
] as [
  string,
  string,
  string[]
][])('CORE-118 cites the surviving serial axes for %s %s', (file, label, citations) => {
  const reason = getAllowance(file, label).reason;
  for (const citation of citations) expect(reason).toContain(citation);
});

it('review B1 records nine log-gamma shifts for an eagerly evaluated empty-quadrat p-value', () => {
  // Four centers of a 2x2 grid over [0,4]^2 lie inside the [0.5,3.5]^2 hole.
  const centers = [
    [1, 1],
    [3, 1],
    [1, 3],
    [3, 3]
  ];
  const activeCount = centers.filter(
    ([horizontal, vertical]) =>
      !(horizontal > 0.5 && horizontal < 3.5 && vertical > 0.5 && vertical < 3.5)
  ).length;
  expect(activeCount).toBe(0);
  // WGSL select eagerly evaluates the p-value operand: shape=(m-1)/2=-0.5.
  // Exact half-integer sequence -0.5,0.5,...,7.5,8.5 takes nine unit increments.
  let shape = Math.fround((activeCount - 1) / 2);
  let shiftCount = 0;
  while (shape < 8) {
    shape = Math.fround(shape + 1);
    shiftCount++;
  }
  expect(shiftCount).toBe(9);
  expect(shape).toBe(8.5);
  expect(
    getAllowance(
      'dataframe-analysis/pair-statistics/gpu-point-pattern-indices.ts',
      'quadrat-statistics'
    ).reason
  ).toContain(`LOG_GAMMA_MAXIMUM_SHIFT_COUNT = ${shiftCount} shifts`);
});

it('review B3 distinguishes the bounded default pivot layout from explicit likelihood widths', () => {
  // ceil(n / max(64,ceil(n/4096))) gives these counts independently of the helper.
  for (const [rows, expectedTiles] of [
    [262144, 4096],
    [262145, 4033],
    [16777216, 4096]
  ]) {
    expect(getRegressionTileLayout(rows).tileCount).toBe(expectedTiles);
  }
  expect(getRegressionTileLayout(262145, 64).tileCount).toBe(4097);
});

it.each([
  'solve',
  'refine'
])('review B4 excludes inactive diagnostic and descriptive bounds from %s', label => {
  const reason = getAllowance(
    'gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares.ts',
    label
  ).reason;
  expect(reason).not.toContain('SYMMETRIC_EIGENVALUE_SWEEP_COUNT');
  expect(reason).not.toContain('GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_MAXIMUM_PREDICTOR_COUNT');
  expect(reason).toContain('validated instrumentOrder in {1, 2}');
  expect(reason).toContain('Q = k + 1 <= 9');
});
