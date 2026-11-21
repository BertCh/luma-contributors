// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {createGWRAuditScene} from './gwr-mathematical-audit-fixtures';
import {computeGeographicallyWeightedRegressionOnCPU} from './geographically-weighted-regression-oracle';
import {
  getGPUGeographicallyWeightedRegressionParameterValues,
  type GPUGeographicallyWeightedRegressionSettings
} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {GPUGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression';
import {
  getGPUCommandNodeKernelSources,
  createNullWebGPUDevice
} from '../../utils/gpu-contributor-test-utils';
import {createGWRReviewHarness} from './gwr-review-test-utils';
import {
  getStudentTWGSL,
  getStudentTTwoSidedTail
} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-student-t';
import {GET_NAN_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {
  checkTranslatedSingularity,
  checkCountCompleteness,
  checkNonzeroTies,
  checkZeroBudgetStages,
  checkConstantStatistics,
  checkEmptyFirstConstantReduction,
  GWR_REVIEW_TAIL_CASES
} from './gwr-review-cases';
import {
  computeNonstationarityOnCPU,
  getLocalEstimateStandardDeviations
} from './geographically-weighted-regression-nonstationarity-oracle';
import {readFileSync} from 'node:fs';
import {
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_ROW_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_INDEXED_ROW_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_DENSE_INDEXED_ROW_COUNT
} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {GWR_NONSTATIONARITY_AUDIT_CASES} from './gwr-nonstationarity-audit-fixtures';
import {GPULocalCollinearity} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-local-collinearity';

for (const [name, check] of [
  [
    'R3 parallel variance with an empty first tile',
    (harness: ReturnType<typeof createGWRReviewHarness>) =>
      checkEmptyFirstConstantReduction(
        harness,
        async nodes => harness.execute(nodes),
        async view => harness.get(view)
      )
  ],
  ...[0, 1000].map(
    translation =>
      [
        `R1 translated singularity ${translation}`,
        (harness: ReturnType<typeof createGWRReviewHarness>) =>
          checkTranslatedSingularity(
            harness,
            async nodes => harness.execute(nodes),
            translation,
            async view => harness.get(view)
          )
      ] as const
  ),
  ...[false, true, 'chunk'].flatMap(chunked =>
    [
      [40, true],
      [32, false]
    ].map(
      ([capacity, includeCount]) =>
        [
          `R2 count completeness chunked=${chunked}, capacity=${capacity}`,
          (harness: ReturnType<typeof createGWRReviewHarness>) =>
            checkCountCompleteness(
              harness,
              async nodes => harness.execute(nodes),
              chunked as boolean | 'chunk',
              async view => harness.get(view),
              capacity as number,
              includeCount as boolean
            )
        ] as const
    )
  ),
  [
    'R4 nonzero identical spreads',
    (harness: ReturnType<typeof createGWRReviewHarness>) =>
      checkNonzeroTies(
        harness,
        async nodes => harness.execute(nodes),
        async view => harness.get(view)
      )
  ],
  [
    'R7 zero-budget merge and finish clear prior valid draws',
    (harness: ReturnType<typeof createGWRReviewHarness>) =>
      checkZeroBudgetStages(
        harness,
        async nodes => harness.execute(nodes),
        async view => harness.get(view)
      )
  ],
  [
    'R3 ten float32(0.1) responses have zero tile variance',
    (harness: ReturnType<typeof createGWRReviewHarness>) =>
      checkConstantStatistics(
        harness,
        async nodes => harness.execute(nodes),
        async view => harness.get(view)
      )
  ]
] as const) {
  it(name, async () => {
    const device = createNullWebGPUDevice();
    const harness = createGWRReviewHarness(device);
    try {
      await check(harness);
    } finally {
      harness.destroy();
      device.destroy();
    }
  });
}

it('R6 compacts the intact masked scene before randomization', () => {
  const actual = computeNonstationarityOnCPU(
    createGWRAuditScene(),
    {kernel: 'gaussian', bandwidths: [3]},
    5,
    12345
  );
  const fixture = GWR_NONSTATIONARITY_AUDIT_CASES[0];
  for (const [index, value] of actual.standardDeviations.flat().entries()) {
    expect(Number.isFinite(value)).toBe(true);
    expect(Math.abs(value - fixture.standardDeviations[index])).toBeLessThan(1e-10);
  }
  expect(actual.pseudoPValues).toEqual([1 / 6, 1, 1 / 6]);
});
it('R6 holds the observed winner on a two-bandwidth ladder', () => {
  const scene = {
    positions: Float32Array.from(Array.from({length: 16}, (_, row) => [row, 0]).flat()),
    predictors: Float32Array.from(Array.from({length: 16}, (_, row) => row % 2)),
    response: Float32Array.from(
      Array.from({length: 16}, (_, row) => Math.floor(row / 4) ** 2 + (row % 2))
    ),
    predictorCount: 1
  };
  const actual = computeNonstationarityOnCPU(
    scene,
    {kernel: 'gaussian', bandwidths: [2, 8]},
    5,
    12345
  );
  expect(actual.pseudoPValues[1]).toBe(1);
  expect(Math.abs(actual.standardDeviations[1][1] - 1.775439)).toBeLessThan(1e-6);
  const expected = [
    3.1470446556653178, 0.34080305790515947, 1.523568732110783, 1.7754390033496723,
    0.716827029150525, 1.0760594125475909, 1.6086275324574435, 1.9592146811315811,
    2.3698087285352947, 1.7805397233389555, 1.5655990068287946, 2.522167926550591
  ];
  for (const [index, value] of actual.standardDeviations.flat().entries()) {
    expect(Number.isFinite(value)).toBe(true);
    expect(Math.abs(value - expected[index])).toBeLessThan(1e-10);
  }
});
it('R6 population spread retains small deviations from a large anchor', () => {
  expect(getLocalEstimateStandardDeviations([100000000, 100000001], 1)).toEqual([0.5]);
});

it('R10 refuses cross-dispatch output aliasing with contributor ID', () => {
  const device = createNullWebGPUDevice();
  const harness = createGWRReviewHarness(device);
  const {port} = harness;
  try {
    const condition = port('float32', Array(8).fill(0));
    expect(
      () =>
        new GPULocalCollinearity({
          id: 'review-alias',
          predictorCount: 1,
          predictors: port('float32', Array(8).fill(1)),
          weights: {
            offsets: port('uint32', Array(9).fill(0)),
            neighbors: port('uint32', [0]),
            weights: port('float32', [1])
          },
          output: {
            varianceInflationFactors: port('float32', Array(8).fill(0)),
            conditionNumbers: condition,
            designConditionNumbers: condition,
            varianceDecomposition: port('float32', Array(16).fill(0)),
            status: port('uint32', Array(8).fill(0))
          }
        })
    ).toThrow(/^review-alias.*overlap/);
  } finally {
    harness.destroy();
    device.destroy();
  }
});

it('R3 constant nonzero response has undefined local/global R2', () => {
  for (const value of [3, 0.1]) {
    const scene = createGWRAuditScene();
    scene.response.fill(value);
    const actual = computeGeographicallyWeightedRegressionOnCPU({
      ...scene,
      settings: {kernel: 'gaussian', bandwidths: [3]}
    });
    expect(actual.localR2.every(Number.isNaN)).toBe(true);
    expect(actual.rSquared).toBeNaN();
  }
});
it('R5 default adaptive bounds collapse to the feasible eight-neighbor endpoint', () => {
  const actual = computeGeographicallyWeightedRegressionOnCPU({
    ...createGWRAuditScene(),
    settings: {bandwidthMode: 'adaptive', bandwidths: [0, 0]},
    bandwidthSearch: 'golden-section',
    maximumBandwidthCount: 2,
    maximumSearchIterations: 1
  });
  expect(actual.selectedValue).toBe(8);
  expect(actual.hasValidCandidate).toBe(true);
  expect(Math.abs(actual.aicc - 339.5339004011)).toBeLessThan(1e-8);
});
it('R8 invalid adaptive bandwidth counts zero fitted observations', () => {
  const actual = computeGeographicallyWeightedRegressionOnCPU({
    ...createGWRAuditScene(),
    settings: {bandwidthMode: 'adaptive', bandwidths: [9]},
    maximumBandwidthCount: 1
  });
  expect(actual.candidates[0].observationCount).toBe(0);
  expect(actual.candidates[0].includedRowCount).toBe(8);
  expect(actual.candidates[0].aicc).toBeNaN();
});
it('R9 captures the nested great-circle radius before caller mutation', () => {
  const device = createNullWebGPUDevice();
  const harness = createGWRReviewHarness(device);
  const {port, graph} = harness;
  const context = {
    coordinateSpace: 'longitude-latitude' as const,
    metric: 'great-circle' as const,
    units: 'meters' as const,
    sphereRadius: 6371000
  };
  try {
    const contributor = new GPUGeographicallyWeightedRegression({
      positions: port('float32x2', Array(16).fill(0)),
      predictors: port('float32', Array(8).fill(1)),
      predictorCount: 1,
      response: port('float32', Array(8).fill(1)),
      parameters: port('float32', [0, 0, 1, 0, 3]),
      maximumBandwidthCount: 1,
      spatialContext: context,
      output: {coefficients: port('float32', Array(16).fill(0))}
    });
    context.sphereRadius = 1;
    expect(contributor.spatialContext.sphereRadius).toBe(6371000);
    expect(
      getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph))
        .map(item => item.source)
        .join('\n')
    ).toContain('6371000.0');
  } finally {
    harness.destroy();
    device.destroy();
  }
});
it('R11 stable direct t tail at the df cap and NaN propagation', () => {
  // scipy.stats.t.sf(1.96,200000)*2 = 0.049997176742712555.
  const values = new Float32Array(2);
  new WgslExec(
    new WgslParser().parse(`${GET_NAN_WGSL}\n${getStudentTWGSL()}
@group(0) @binding(0) var<storage,read_write> result:array<f32>;
@compute @workgroup_size(1,1,1) fn main(){result[0]=gwrStudentTwoSidedTail(1.96,200000.0);result[1]=gwrStudentTwoSidedTail(getNaN(),7.0);}`)
  ).dispatchWorkgroups('main', [1, 1, 1], {0: {0: values}});
  expect(Number.isFinite(values[0])).toBe(true);
  expect(Math.abs(values[0] - 0.049997176742712555)).toBeLessThan(2e-6);
  expect(values[1]).toBeNaN();
  expect(getStudentTTwoSidedTail(NaN, 7)).toBeNaN();
});
it('R12 inherited names are rejected without changing a target word', () => {
  for (const setting of [{kernel: 'toString'}, {bandwidthMode: 'constructor'}]) {
    const target = new Float32Array(5).fill(123);
    expect(() =>
      getGPUGeographicallyWeightedRegressionParameterValues(
        {...setting, bandwidths: [3]} as GPUGeographicallyWeightedRegressionSettings,
        1,
        target
      )
    ).toThrow(/recognized/);
    expect(Array.from(target)).toEqual([123, 123, 123, 123, 123]);
  }
});

it('R11 large-df tail range matches independent SciPy literals', () => {
  const values = new Float32Array(GWR_REVIEW_TAIL_CASES.length);
  const body = GWR_REVIEW_TAIL_CASES.map(
    ([value, degrees], index) =>
      `result[${index}u]=gwrStudentTwoSidedTail(${value.toFixed(2)},${degrees}.0);`
  ).join('\n');
  new WgslExec(
    new WgslParser().parse(
      `${GET_NAN_WGSL}\n${getStudentTWGSL()}\n@group(0) @binding(0) var<storage,read_write> result:array<f32>; @compute @workgroup_size(1,1,1) fn main(){${body}}`
    )
  ).dispatchWorkgroups('main', [1, 1, 1], {0: {0: values}});
  for (const [index, [, , expected]] of GWR_REVIEW_TAIL_CASES.entries()) {
    expect(Number.isFinite(values[index])).toBe(true);
    expect(Math.abs(values[index] / expected - 1)).toBeLessThan(3e-5);
  }
});

it('R13 guide states the named scan, indexed and exact Gaussian limits', () => {
  const guide = readFileSync(
    new URL(
      '../../../../../docs/api/gpu-spatial-analysis/operators-regression.md',
      import.meta.url
    ),
    'utf8'
  );
  for (const [name, limit] of Object.entries({
    GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_ROW_COUNT,
    GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_INDEXED_ROW_COUNT,
    GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_DENSE_INDEXED_ROW_COUNT
  })) {
    expect(guide).toContain(name);
    expect(guide).toContain(limit.toLocaleString('en-US'));
  }
  const prose = guide.replace(/\s+/g, ' ');
  expect(prose).toContain(
    `scan limit is ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_ROW_COUNT.toLocaleString('en-US')} rows`
  );
  expect(prose).toContain(
    `indexed execution supports up to ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_INDEXED_ROW_COUNT.toLocaleString('en-US')} rows subject to device`
  );
  expect(prose).toContain(
    `Explicit exact Gaussian support is limited to ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_DENSE_INDEXED_ROW_COUNT.toLocaleString('en-US')} rows`
  );
});
