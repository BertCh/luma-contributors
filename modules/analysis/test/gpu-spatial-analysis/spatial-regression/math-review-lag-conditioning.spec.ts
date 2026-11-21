// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it, expect} from 'vitest';
import type {GPUCommandGraph, GPUCommandNodeProducer, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {GPUOrdinaryLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares';
import {GPUSpatialErrorGM} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm';
import {GPUSpatialLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood';
import {GPUSpatialTwoStageLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';
import {GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-parameters';
import {GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-error-gm-parameters';
import {
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_AT_BOUNDARY,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_LIKELIHOOD,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PSEUDO_R_SQUARED,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PSEUDO_R_SQUARED_SPATIAL,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SIGMA_SQUARED,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_VARIANCE_DEFINED
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-parameters';
import {
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_ANSELIN_KELEJIAN,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_MORAN_I,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_SIGMA_SQUARED
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-two-stage-least-squares-parameters';
import {createRegressionTestContext} from './math-review-regression-test-utils';
import fixture from '../../oracles/fixtures/statistics-regression-lag-conditioning.json';

/**
 * The spatial lag estimators on inputs that stress their float32 conditioning: the review fixes
 * of the closing pass (2026-11-11), frc-regression-01, -02, -04 and -06 of
 * `dev-docs/research/math-correctness-review-2026-11-10/fix-review-closing-regression.md`.
 *
 * Reference numbers: `test/oracles/fixtures/statistics-regression-lag-conditioning.json`, written
 * by `test/oracles/statistics-regression-lag-conditioning.py` (its docstring has the recipe). In
 * short, float64 on the float32 inputs with spreg 1.9.1, libpysal 4.15.0 and NumPy 2.5.4:
 * - lag maximum likelihood: the maximum of the concentrated likelihood (centered least squares,
 *   eigenvalues of `W`, grid and golden section), the estimator of `spreg.ML_Lag(method='full')`,
 *   whose own search the generator holds to within 5e-7 in `rho` and 0.05 log-likelihood units;
 *   standard errors are spreg's;
 * - two-stage least squares: the centered float64 solve, which the generator holds to
 *   `spreg.GM_Lag(w_lags=1, spat_diag=True)` wherever a shift of the response is an intercept
 *   shift (every scene but the islands and the mixed row sums).
 *
 * The scenes are rebuilt here from the parameters stored in the fixture, with a 32-bit linear
 * congruential stream and IEEE double arithmetic in the generator's order (`responseSum` checks
 * that the inputs are the same bits).
 *
 * Before these fixes, on the test GPU:
 * - `high-fit-*` (frc-regression-01): the `rho` search differenced
 *   `e0'e0 - 2 rho e0'e1 + rho^2 e1'e1` in float32, where the minimum is 1e-9 to 1e-7 of
 *   `e0'e0`: 17 of the 24 scenes were status 3 or a wrong fit with status 0 (`sigma2` up to
 *   41 times the reference);
 * - `islands-lag`: `sigma2` 2,801 times the reference with status 0;
 * - `lag-*` (frc-regression-02): the lag of the raw response was rounded to a float32 step of
 *   10,000: `sigma2` up to 4.2% off and the Anselin-Kelejian statistic up to 45%.
 */
const ROWS = fixture.rowCount;
const ISLAND_ROWS = new Set(fixture.islandRows);

type Scene = (typeof fixture.scenes)[number];
type WeightRows = {neighbors: number[]; weights: number[]}[];

/** Uniform stream in [0, 1): a 32-bit linear congruential state, top 24 bits. */
function createStream(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state >>> 8) / 16777216;
  };
}

function getNoise(seed: number): number[] {
  const getUniform = createStream(seed);
  const noise: number[] = [];
  for (let row = 0; row < ROWS; row++) {
    const total = getUniform() + getUniform() + getUniform() + getUniform();
    noise.push((total - 2) * 1.7320508075688772);
  }
  return noise;
}

/** `uniform` on [-2, 2) as float32; `grid` is 4096 plus a multiple of 2^-9 in [-2, 2). */
function getPredictor(kind: string): number[] {
  if (kind === 'uniform') {
    const getUniform = createStream(20261111);
    return Array.from({length: ROWS}, () => Math.fround(4 * getUniform() - 2));
  }
  const getUniform = createStream(424242);
  return Array.from(
    {length: ROWS},
    () => fixture.gridPredictorCenter + Math.floor(getUniform() * 2048 - 1024) / 512
  );
}

/** The generator's weights: neighbours ascending, float32 values. */
function getWeightRows(kind: string): WeightRows {
  return Array.from({length: ROWS}, (_, row) => {
    let neighbors = [(row + ROWS - 1) % ROWS, (row + 1) % ROWS];
    if (kind === 'six') {
      neighbors = [-3, -2, -1, 1, 2, 3].map(step => (row + ROWS + step) % ROWS);
    } else if ((kind === 'mixed' || kind === 'unequal') && row % 3 === 0) {
      neighbors.push((row + 2) % ROWS);
    }
    if (kind === 'islands') {
      neighbors = ISLAND_ROWS.has(row)
        ? []
        : neighbors.filter(neighbor => !ISLAND_ROWS.has(neighbor));
    }
    neighbors.sort((first, second) => first - second);
    const weight =
      kind === 'ring'
        ? 0.5
        : kind === 'binary' || kind === 'unequal'
          ? 1
          : Math.fround(1 / neighbors.length);
    return {neighbors, weights: neighbors.map(() => weight)};
  });
}

/** Float64 lag, row by row in slot order. */
function getLag(rows: WeightRows, values: ArrayLike<number>): number[] {
  return rows.map(({neighbors, weights}) => {
    let total = 0;
    for (let slot = 0; slot < neighbors.length; slot++)
      total += weights[slot] * values[neighbors[slot]];
    return total;
  });
}

/** `(I - rho W)^-1 (constant + sum slope (x - offset) + deviation e)` by the power series. */
function createResponse(scene: Scene, rows: WeightRows, predictors: number[][]): Float32Array {
  const noise = getNoise(scene.noiseSeed);
  const reduced = Array.from({length: ROWS}, (_, row) => {
    let value = scene.constant;
    scene.slopes.forEach((slope, column) => {
      value += slope * (predictors[column][row] - scene.predictorOffsets[column]);
    });
    return value + scene.noiseDeviation * noise[row];
  });
  const total = [...reduced];
  let term = [...reduced];
  if (scene.parameter !== 0) {
    for (let power = 1; power < fixture.seriesTermCount; power++) {
      term = getLag(rows, term).map(value => scene.parameter * value);
      for (let row = 0; row < ROWS; row++) total[row] += term[row];
    }
  }
  return Float32Array.from(total, value =>
    scene.roundToGrid ? Math.floor(value * 512 + 0.5) / 512 : value
  );
}

function createInputs(scene: Scene) {
  const rows = getWeightRows(scene.weights);
  const predictors = scene.predictors.map(getPredictor);
  const response = createResponse(scene, rows, predictors);
  expect(Array.from(response).reduce((sum, value) => sum + value, 0)).toBe(scene.responseSum);
  return {rows, predictors, response};
}

type Estimator =
  | 'lag-maximum-likelihood'
  | 'two-stage'
  | 'ordinary-least-squares'
  | 'kelejian-prucha'
  | 'homoskedastic';

/**
 * Workspace slot of the searched `rho` in `GPUSpatialLagMaximumLikelihood` (its `W_PARAMETER`
 * constant; `spatial-maximum-likelihood-nodes.node.spec.ts` pins the value in the kernel text).
 */
const LAG_PARAMETER_SLOT = 2;

it('keeps row-zero lag ties in the centered column when other rows differ', async () => {
  const context = await createRegressionTestContext();
  const {graph, input, output} = context;
  try {
    const rowCount = 8;
    const lag = output('float32', rowCount);
    recordWGSLKernelNodes(graph);
    const model = new GPUSpatialLagMaximumLikelihood({
      id: 'varying-lag',
      maximumPower: 4,
      predictorCount: 1,
      predictors: input('float32', [0, 1, 2, 3, 4, 5, 6, 7]),
      response: input(
        'float32',
        [1, -1, 0, 0, 2, -2, 1, -1].map(offset => 10_000 + offset / 1024)
      ),
      weights: {
        offsets: input('uint32', [0, 2, 4, 6, 8, 10, 12, 14, 16]),
        neighbors: input('uint32', [3, 6, 4, 7, 0, 4, 1, 4, 2, 5, 1, 7, 0, 2, 1, 3]),
        weights: input('float32', new Float32Array(16).fill(0.5))
      },
      output: {
        table: output('float32', 12).view,
        summary: output('float32', GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH).view,
        status: {code: output('uint32', 1).view}
      }
    });
    const nodes = model.getCommandNodes(graph);
    const shiftId = 'varying-lag-lagged-response-shift';
    const shiftIndex = nodes.findIndex(node => node.id === shiftId);
    expect(shiftIndex).toBeGreaterThan(0);
    const centeredLag = getWGSLKernelNodeRecord(graph, shiftId)?.bindings.find(
      binding => binding.name === 'lagged'
    )?.view as GraphDataView<'float32'> | undefined;
    expect(centeredLag).toBeDefined();
    graph.add(nodes.slice(0, shiftIndex + 1));
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'read-centered-lag',
        operation: 'math-review-lag-conditioning',
        invocationCount: rowCount,
        bindings: [
          {name: 'source', view: centeredLag!, type: 'f32', access: 'read'},
          {name: 'publishedLag', view: lag.view, type: 'f32', access: 'read_write'}
        ],
        body: 'publishedLag[publishedLagOffset + index] = source[sourceOffset + index];'
      })
    );
    context.run();
    // The mean is 10,000 and each row averages two neighbors. Row zero and row one both
    // have raw lag 10,000 + 1/2048, a tie that would round to 10,000 before centering.
    expect(await readFloat32(lag.buffer, rowCount)).toEqual(
      [1, 1, 3, 1, -2, -2, 1, -1].map(sum => sum / 2048)
    );
  } finally {
    context.destroy();
  }
});

/**
 * The lag maximum likelihood nodes with one more node right after the `rho` search, which adds
 * `parameterOffset` to the stored `rho`. Every later node (coefficients, residuals, variance,
 * the first-order check of the finish kernel) then runs on the displaced value, as it would
 * after a search that ended in the wrong place. Test only: the class has no such input.
 */
function createDisplacedSearch(
  estimator: GPUSpatialLagMaximumLikelihood,
  parameterOffset: number
): GPUCommandNodeProducer {
  return {
    getCommandNodes(graph: GPUCommandGraph) {
      const nodes = [...estimator.getCommandNodes(graph)];
      const searchIndex = nodes.findIndex(node => node.id === `${estimator.id}-rho-search`);
      expect(searchIndex).toBeGreaterThan(0);
      // The search writes one buffer: the workspace.
      const written = (nodes[searchIndex].resources ?? []).filter(
        resource => 'buffer' in resource && resource.usage === 'storage-read-write'
      );
      expect(written.length).toBe(1);
      const workspace = (written[0] as {buffer: GraphDataView<'float32'>}).buffer;
      nodes.splice(
        searchIndex + 1,
        0,
        createWGSLKernelNode(graph, {
          id: `${estimator.id}-test-displace-parameter`,
          operation: 'math-review-lag-conditioning',
          variant: 'displace-parameter',
          invocationCount: 1,
          bindings: [{name: 'workspace', view: workspace, type: 'f32', access: 'read_write'}],
          declarations: '',
          body: `workspace[workspaceOffset + ${LAG_PARAMETER_SLOT}u] = workspace[workspaceOffset + ${LAG_PARAMETER_SLOT}u] + ${getWGSLFloatLiteral(parameterOffset)};`
        })
      );
      return nodes;
    }
  };
}

/**
 * Runs one estimator; `columns[column][row]` are the predictors. `parameterOffset` (lag maximum
 * likelihood only) displaces the searched `rho`, see `createDisplacedSearch`.
 */
async function runEstimator(
  estimator: Estimator,
  rows: WeightRows,
  columns: ArrayLike<number>[],
  response: ArrayLike<number>,
  parameterOffset?: number
) {
  const rowCount = response.length;
  const context = await createRegressionTestContext();
  const {graph, input, output} = context;
  try {
    const predictorCount = columns.length;
    const offsets = [0];
    for (const {neighbors} of rows) offsets.push(offsets[offsets.length - 1] + neighbors.length);
    const weights = {
      offsets: input('uint32', offsets),
      neighbors: input(
        'uint32',
        rows.flatMap(({neighbors}) => neighbors)
      ),
      weights: input(
        'float32',
        rows.flatMap(row => row.weights)
      )
    };
    const predictors = input(
      'float32',
      Array.from({length: rowCount * predictorCount}, (_, index) => {
        const column = index % predictorCount;
        return columns[column][(index - column) / predictorCount];
      })
    );
    const status = output('uint32', 1);
    const residuals = output('float32', rowCount);
    if (estimator === 'ordinary-least-squares') {
      const coefficients = output('float32', predictorCount + 1);
      const standardErrors = output('float32', predictorCount + 1);
      graph.add(
        new GPUOrdinaryLeastSquares({
          predictors,
          predictorCount,
          response: input('float32', response),
          output: {
            coefficients: coefficients.view,
            standardErrors: standardErrors.view,
            tStatistics: output('float32', predictorCount + 1).view,
            summary: output('float32', GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH).view,
            status: {code: status.view},
            residuals: residuals.view
          }
        })
      );
      context.run();
      return {
        status: (await readUint32(status.buffer, 1))[0],
        coefficients: Array.from(await readFloat32(coefficients.buffer, predictorCount + 1)),
        standardErrors: Array.from(await readFloat32(standardErrors.buffer, predictorCount + 1)),
        summary: [] as number[],
        residuals: Array.from(await readFloat32(residuals.buffer, rowCount))
      };
    }
    const tableLength = (predictorCount + 2) * 4;
    const table = output('float32', tableLength);
    const summaryLength =
      estimator === 'lag-maximum-likelihood'
        ? GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH
        : estimator === 'two-stage'
          ? GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH
          : GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH;
    const summary = output('float32', summaryLength);
    const common = {
      weights,
      predictors,
      predictorCount,
      response: input('float32', response),
      output: {
        table: table.view,
        summary: summary.view,
        status: {code: status.view},
        residuals: residuals.view
      }
    };
    if (estimator === 'lag-maximum-likelihood') {
      const lagEstimator = new GPUSpatialLagMaximumLikelihood(common);
      graph.add(
        parameterOffset === undefined
          ? lagEstimator
          : createDisplacedSearch(lagEstimator, parameterOffset)
      );
    } else if (estimator === 'two-stage') graph.add(new GPUSpatialTwoStageLeastSquares(common));
    else graph.add(new GPUSpatialErrorGM({...common, estimator}));
    context.run();
    const tableValues = await readFloat32(table.buffer, tableLength);
    return {
      status: (await readUint32(status.buffer, 1))[0],
      coefficients: Array.from({length: predictorCount + 2}, (_, row) => tableValues[row * 4]),
      standardErrors: Array.from(
        {length: predictorCount + 2},
        (_, row) => tableValues[row * 4 + 1]
      ),
      summary: Array.from(await readFloat32(summary.buffer, summaryLength)),
      residuals: Array.from(await readFloat32(residuals.buffer, rowCount))
    };
  } finally {
    context.destroy();
  }
}

const getRelativeError = (value: number, reference: number) => Math.abs(value / reference - 1);

type LagReference = {
  parameter: number;
  slopes: number[];
  intercept: number;
  standardErrors: number[];
  sigmaSquared: number;
};

/** Errors of one fit against its float64 reference, in the units the assertions use. */
function getErrors(
  result: Awaited<ReturnType<typeof runEstimator>>,
  reference: LagReference,
  inputs: ReturnType<typeof createInputs>,
  sigmaSquared: number
) {
  const count = reference.slopes.length;
  // Float64 residuals at the reference coefficients: u = y - rho W y - a - X b.
  const lagged = getLag(inputs.rows, inputs.response);
  const deviation = Math.sqrt(reference.sigmaSquared);
  let largestResidualError = 0;
  for (let row = 0; row < ROWS; row++) {
    let residual = inputs.response[row] - reference.parameter * lagged[row] - reference.intercept;
    reference.slopes.forEach((slope, column) => {
      residual -= slope * inputs.predictors[column][row];
    });
    largestResidualError = Math.max(
      largestResidualError,
      Math.abs(result.residuals[row] - residual)
    );
  }
  return {
    status: result.status,
    parameter: result.coefficients[count + 1],
    parameterErrorOverStandardError:
      Math.abs(result.coefficients[count + 1] - reference.parameter) /
      reference.standardErrors[count + 1],
    slopeError: Math.max(
      ...reference.slopes.map((slope, column) =>
        getRelativeError(result.coefficients[column + 1], slope)
      )
    ),
    interceptErrorOverStandardError:
      Math.abs(result.coefficients[0] - reference.intercept) / reference.standardErrors[0],
    sigmaSquaredError: getRelativeError(sigmaSquared, reference.sigmaSquared),
    standardErrorError: Math.max(
      ...reference.standardErrors.map((standardError, row) =>
        getRelativeError(result.standardErrors[row], standardError)
      )
    ),
    largestResidualErrorOverDeviation: largestResidualError / deviation
  };
}

/** Largest allowed errors of one fit, in the units of `getErrors`. */
type Bounds = {
  parameterOverStandardError: number;
  slope: number;
  interceptOverStandardError: number;
  sigmaSquared: number;
  standardError: number;
  residualOverDeviation: number;
  /** Lag maximum likelihood: absolute log-likelihood units. */
  logLikelihood: number;
  /** Two-stage least squares: `|AK - reference| <= relative * reference + absolute`. */
  anselinKelejianRelative: number;
  anselinKelejianAbsolute: number;
};

/**
 * Bounds, each three times or more the largest error measured on the test GPU over the scenes
 * of its group (the measured values are in the review report, regression.md, "Closing pass,
 * review fixes").
 *
 * - Default (the `lag-*` scenes of frc-regression-02, the two-predictor scene, the ordinary
 *   response on the island weights): what remains is float32 rounding of the centered columns.
 *   Before the fix `sigma2` was up to 4.2% off here and the Anselin-Kelejian statistic 45%.
 * - `high-fit-*`: the residual (0.004) is 1e-5 of the fitted part (slope 200 times a predictor
 *   of 2), so the float32 rounding of the fitted values is 1% to 3% of a residual, and a
 *   float32 step of `rho` is a thirtieth of its standard error. Inherent to float32 columns.
 * - `islands-lag`, lag maximum likelihood: a row without neighbours has no lag, so its residual
 *   moves by `10000 d rho`, and a float32 step of `rho` near 0.4 is 3e-8, an eighth of its
 *   standard error.
 * - `lag-unequal-*` (row sums 2 and 3, response near 10,000), lag maximum likelihood: the lag
 *   differs between rows by 10,000, the standard error of `rho` is 4.5e-8 (three float32 steps),
 *   and the lag column of the second OLS fit is rounded to a float32 step of 10,000.
 * - Two-stage least squares on those two scenes (frc-regression-11 of the re-review). Its
 *   estimate is not at the minimum of `u'u`, so `sigma2` moves by percents for a few float32
 *   steps of `rho`; the float32 solve landed three and five steps off, and `sigma2` was 3% and
 *   0.8% off. One refinement step on residuals formed without the rounding of their products,
 *   with the correction of `delta` kept beside it, brings both inside the default bound
 *   (`islands-lag` keeps a wider `sigma2` bound: the three island rows carry most of `u'u`).
 */
const DEFAULT_BOUNDS: Record<'lag' | 'twoStage', Bounds> = {
  lag: {
    parameterOverStandardError: 5e-3,
    slope: 2e-6,
    interceptOverStandardError: 1e-2,
    sigmaSquared: 4e-5,
    standardError: 2e-4,
    residualOverDeviation: 2.5e-3,
    logLikelihood: 0.01,
    anselinKelejianRelative: 0,
    anselinKelejianAbsolute: 0
  },
  twoStage: {
    parameterOverStandardError: 2e-3,
    slope: 2e-7,
    interceptOverStandardError: 1.6e-2,
    sigmaSquared: 1e-5,
    standardError: 1e-5,
    residualOverDeviation: 8e-4,
    logLikelihood: 0,
    anselinKelejianRelative: 8e-4,
    anselinKelejianAbsolute: 0
  }
};

function getBounds(sceneName: string, estimator: 'lag' | 'twoStage'): Bounds {
  const bounds = DEFAULT_BOUNDS[estimator];
  if (sceneName.startsWith('high-fit')) {
    return estimator === 'lag'
      ? {
          ...bounds,
          parameterOverStandardError: 0.1,
          interceptOverStandardError: 0.4,
          sigmaSquared: 1e-3,
          standardError: 5e-4,
          residualOverDeviation: 0.1,
          logLikelihood: 0.3
        }
      : {
          ...bounds,
          parameterOverStandardError: 0.06,
          interceptOverStandardError: 0.2,
          sigmaSquared: 5e-4,
          standardError: 2.5e-4,
          residualOverDeviation: 0.035,
          anselinKelejianRelative: 0.01,
          anselinKelejianAbsolute: 0.005
        };
  }
  if (sceneName === 'islands-lag') {
    return estimator === 'lag'
      ? {
          ...bounds,
          parameterOverStandardError: 0.4,
          slope: 5e-6,
          interceptOverStandardError: 1.2,
          sigmaSquared: 1.5e-4,
          residualOverDeviation: 0.15,
          logLikelihood: 0.04
        }
      : {...bounds, sigmaSquared: 3e-5, standardError: 2e-5};
  }
  if (sceneName.startsWith('lag-unequal') && estimator === 'lag') {
    return {
      ...bounds,
      parameterOverStandardError: 0.25,
      interceptOverStandardError: 0.6,
      sigmaSquared: 1e-3,
      standardError: 1e-3,
      residualOverDeviation: 0.08,
      logLikelihood: 0.3
    };
  }
  return bounds;
}

for (const scene of fixture.scenes) {
  it(`lag maximum likelihood on ${scene.name} matches the float64 maximum of the concentrated likelihood`, async () => {
    const inputs = createInputs(scene);
    const result = await runEstimator(
      'lag-maximum-likelihood',
      inputs.rows,
      inputs.predictors,
      inputs.response
    );
    const reference = scene.lagMaximumLikelihood;
    const errors = getErrors(
      result,
      reference,
      inputs,
      result.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SIGMA_SQUARED]
    );
    const logLikelihood = result.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_LIKELIHOOD];
    const message = `GPU ${JSON.stringify({
      ...errors,
      logLikelihoodError: logLikelihood - reference.logLikelihood
    })}`;
    const bounds = getBounds(scene.name, 'lag');
    expect(result.status, message).toBe(0);
    expect(result.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_AT_BOUNDARY], message).toBe(0);
    expect(result.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_VARIANCE_DEFINED], message).toBe(
      1
    );
    expect(errors.parameterErrorOverStandardError, message).toBeLessThan(
      bounds.parameterOverStandardError
    );
    expect(errors.slopeError, message).toBeLessThan(bounds.slope);
    expect(errors.interceptErrorOverStandardError, message).toBeLessThan(
      bounds.interceptOverStandardError
    );
    expect(errors.sigmaSquaredError, message).toBeLessThan(bounds.sigmaSquared);
    // Concentrated ML uses the mean squared residual at the final rho, including on scenes
    // where some rows share an exact raw lag and other rows do not.
    const residualVariance =
      result.residuals.reduce((sum, residual) => sum + residual ** 2, 0) / result.residuals.length;
    expect(
      getRelativeError(
        result.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SIGMA_SQUARED],
        residualVariance
      ),
      message
    ).toBeLessThan(2e-6);
    expect(Math.abs(logLikelihood - reference.logLikelihood), message).toBeLessThan(
      bounds.logLikelihood
    );
    // The library directly: never below the likelihood spreg's own search reaches by more than
    // the same bound.
    expect(logLikelihood, message).toBeGreaterThan(
      reference.spreg.logLikelihood - bounds.logLikelihood
    );
    expect(errors.standardErrorError, message).toBeLessThan(bounds.standardError);
    expect(errors.largestResidualErrorOverDeviation, message).toBeLessThan(
      bounds.residualOverDeviation
    );
  });

  it(`two-stage least squares on ${scene.name} matches the float64 solve and spreg GM_Lag`, async () => {
    const inputs = createInputs(scene);
    const result = await runEstimator('two-stage', inputs.rows, inputs.predictors, inputs.response);
    const reference = scene.twoStage;
    const errors = getErrors(
      result,
      reference,
      inputs,
      result.summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_SIGMA_SQUARED]
    );
    const anselinKelejian =
      result.summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_ANSELIN_KELEJIAN];
    const anselinKelejianError = getRelativeError(anselinKelejian, reference.anselinKelejian);
    const message = `GPU ${JSON.stringify({...errors, anselinKelejian, anselinKelejianError})}`;
    const bounds = getBounds(scene.name, 'twoStage');
    expect(result.status, message).toBe(0);
    expect(errors.parameterErrorOverStandardError, message).toBeLessThan(
      bounds.parameterOverStandardError
    );
    expect(errors.slopeError, message).toBeLessThan(bounds.slope);
    expect(errors.interceptErrorOverStandardError, message).toBeLessThan(
      bounds.interceptOverStandardError
    );
    expect(errors.sigmaSquaredError, message).toBeLessThan(bounds.sigmaSquared);
    expect(errors.standardErrorError, message).toBeLessThan(bounds.standardError);
    expect(Math.abs(anselinKelejian - reference.anselinKelejian), message).toBeLessThan(
      bounds.anselinKelejianRelative * reference.anselinKelejian + bounds.anselinKelejianAbsolute
    );
    expect(errors.largestResidualErrorOverDeviation, message).toBeLessThan(
      bounds.residualOverDeviation
    );
    // The library directly, where a shift of the response is an intercept shift (the generator
    // holds the float64 solve to spreg GM_Lag there to 1e-8).
    if (!['islands', 'mixed', 'unequal'].includes(scene.weights)) {
      expect(
        getRelativeError(
          result.summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_SIGMA_SQUARED],
          reference.spreg.sigmaSquared
        ),
        message
      ).toBeLessThan(bounds.sigmaSquared);
    }
  });
}

/**
 * A constant predictor is collinear with the intercept: status 1 for every estimator
 * (frc-regression-04). Before the fix two-stage least squares accepted some constants with
 * status 0: the float32 mean of 600 copies of 12345.678 is not 12345.678, so the "centered"
 * column was one tiny value with a sum of squares above zero, and the fit reported an intercept
 * standard error of 346.6 where the fit without the column has 0.947.
 */
const CONSTANTS = [0.1, 1 / 3, 7, 12345.678, 99999.99, -4096.3, 1e-3, 16777217];
const ESTIMATORS: Estimator[] = [
  'ordinary-least-squares',
  'lag-maximum-likelihood',
  'two-stage',
  'kelejian-prucha',
  'homoskedastic'
];

for (const sceneName of [
  'two-predictors-six-rho-0.4',
  'islands-lag',
  'lag-ring-rho-0.4-deviation-0.004'
])
  for (const estimator of ESTIMATORS)
    it(`${estimator} on ${sceneName} with a constant predictor is status 1 for every constant`, async () => {
      const scene = fixture.scenes.find(candidate => candidate.name === sceneName)!;
      const inputs = createInputs(scene);
      const statuses: number[] = [];
      for (const constant of CONSTANTS) {
        const result = await runEstimator(
          estimator,
          inputs.rows,
          [...inputs.predictors, new Array<number>(ROWS).fill(constant)],
          inputs.response
        );
        statuses.push(result.status);
      }
      expect(statuses, `constants ${CONSTANTS}`).toEqual(CONSTANTS.map(() => 1));
    });

/**
 * Third round of review fixes (2026-11-11): frc2-regression-01, -02 and -07 of
 * `dev-docs/research/math-correctness-review-2026-11-10/fix-review-closing-regression-codex.md`.
 * The inputs are stored in full under `edgeCases` of the fixture (a few rows, or the reviewer's
 * NumPy-generated inputs); the references are spreg 1.9.1 in float64 on those inputs, as the
 * generator's docstring states for each.
 */
const EDGE_CASES = fixture.edgeCases;

function getWeightRowsOf(scene: {
  offsets: number[];
  neighbors: number[];
  weights: number[];
}): WeightRows {
  return Array.from({length: scene.offsets.length - 1}, (_, row) => ({
    neighbors: scene.neighbors.slice(scene.offsets[row], scene.offsets[row + 1]),
    weights: scene.weights.slice(scene.offsets[row], scene.offsets[row + 1])
  }));
}

/** Largest error of the intercept, the slope and `rho`, each over its reference standard error. */
function getCoefficientErrorOverStandardError(
  result: Awaited<ReturnType<typeof runEstimator>>,
  reference: {intercept: number; slopes: number[]; parameter: number; standardErrors: number[]}
): number {
  const expected = [reference.intercept, ...reference.slopes, reference.parameter];
  return Math.max(
    ...expected.map(
      (value, row) => Math.abs(result.coefficients[row] - value) / reference.standardErrors[row]
    )
  );
}

for (const name of ['zeroResidualAtOutsideVertex', 'zeroLagResidual'] as const)
  it(`lag maximum likelihood returns the spreg fit when the residual sums are degenerate (${name}, frc2-regression-01)`, async () => {
    // zeroResidualAtOutsideVertex: e0 = y and e1 = y / 2 exactly, so u'u = 12 (1 - rho / 2)^2:
    // its minimum S is 0, at rho = 2, outside the search interval. zeroLagResidual: W y is the
    // predictor, so e1 = 0 and u'u = e0'e0 for every rho. Both have a likelihood on the whole
    // interval; before the fix the search required S > 0 and e1'e1 > 0 and both were status 3.
    const scene = EDGE_CASES[name];
    const reference = scene.lagMaximumLikelihood;
    const result = await runEstimator(
      'lag-maximum-likelihood',
      getWeightRowsOf(scene),
      [scene.predictor],
      scene.response
    );
    const message = `GPU ${JSON.stringify({coefficients: result.coefficients, standardErrors: result.standardErrors, summary: result.summary.slice(0, 4)})}`;
    expect(result.status, message).toBe(0);
    expect(result.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_AT_BOUNDARY], message).toBe(0);
    expect(result.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_VARIANCE_DEFINED], message).toBe(
      1
    );
    expect(getCoefficientErrorOverStandardError(result, reference), message).toBeLessThan(1e-4);
    expect(
      getRelativeError(
        result.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SIGMA_SQUARED],
        reference.sigmaSquared
      ),
      message
    ).toBeLessThan(1e-5);
    expect(
      Math.abs(
        result.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_LIKELIHOOD] -
          reference.logLikelihood
      ),
      message
    ).toBeLessThan(1e-4);
    reference.standardErrors.forEach((standardError, row) => {
      expect(getRelativeError(result.standardErrors[row], standardError), message).toBeLessThan(
        1e-5
      );
    });
  });

it('lag maximum likelihood gives the same status and rho for a response multiplied by 1e-10, 1 and 1e10 (frc2-regression-02)', async () => {
  // Before the fix the first-order check multiplied two sums of squares: for the response times
  // 1e10 that overflowed float32, the check compared NaN and the fit was status 3.
  const scene = EDGE_CASES.responseScale;
  const rows = getWeightRows('ring');
  const parameters: number[] = [];
  const fits: number[][] = [];
  for (const reference of scene.scales) {
    const response = Float32Array.from(scene.baseResponse, value => value * reference.scale);
    expect(Array.from(response).reduce((sum, value) => sum + value, 0)).toBe(reference.responseSum);
    const result = await runEstimator('lag-maximum-likelihood', rows, [scene.predictor], response);
    const message = `scale ${reference.scale} GPU ${JSON.stringify({coefficients: result.coefficients, standardErrors: result.standardErrors, summary: result.summary.slice(0, 8)})}`;
    expect(result.status, message).toBe(0);
    expect(result.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_AT_BOUNDARY], message).toBe(0);
    // A thousandth of a standard error (measured: 2e-6).
    expect(getCoefficientErrorOverStandardError(result, reference), message).toBeLessThan(1e-3);
    expect(
      getRelativeError(
        result.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SIGMA_SQUARED],
        reference.sigmaSquared
      ),
      message
    ).toBeLessThan(1e-5);
    expect(
      Math.abs(
        result.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_LIKELIHOOD] /
          reference.logLikelihood -
          1
      ),
      message
    ).toBeLessThan(1e-6);
    reference.standardErrors.forEach((standardError, row) => {
      expect(getRelativeError(result.standardErrors[row], standardError), message).toBeLessThan(
        1e-4
      );
    });
    parameters.push(result.coefficients[2]);
    fits.push([
      result.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PSEUDO_R_SQUARED],
      result.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PSEUDO_R_SQUARED_SPATIAL]
    ]);
  }
  // The three inputs differ by the float32 rounding of the scaled response only.
  expect(Math.max(...parameters) - Math.min(...parameters), `rho ${parameters}`).toBeLessThan(2e-7);
  for (const slot of [0, 1]) {
    const values = fits.map(fit => fit[slot]);
    expect(values.every(Number.isFinite), `pseudo R2 ${values}`).toBe(true);
    expect(Math.max(...values) - Math.min(...values), `pseudo R2 ${values}`).toBeLessThan(1e-5);
  }
});

it('lag maximum likelihood accepts the float32 nearest to a maximum whose standard error is below the float32 spacing of rho (frc2-regression-02)', async () => {
  // 64 rows; the float64 standard error of rho is 4.6e-9 and the float32 spacing at 0.4 is
  // 3e-8. The first-order check allowed half a standard error and no more, so no float32 value
  // could pass it: status 3 before the fix. The status and rho are asserted; sigma2 is pinned
  // at the size of its known limit (the end of this test).
  const scene = EDGE_CASES.parameterBelowFloat32Spacing;
  const reference = scene.lagMaximumLikelihood;
  expect(reference.standardErrors[2]).toBeLessThan(1e-8);
  const result = await runEstimator(
    'lag-maximum-likelihood',
    getWeightRowsOf(scene),
    [scene.predictor],
    scene.response
  );
  const message = `GPU ${JSON.stringify({coefficients: result.coefficients, summary: result.summary.slice(0, 4)})}`;
  expect(result.status, message).toBe(0);
  const spacing = 2 ** -25;
  expect(Math.fround(reference.parameter)).toBe(reference.nearestFloat32Parameter);
  expect(Math.abs(result.coefficients[2] - reference.parameter), message).toBeLessThan(2 * spacing);
  expect(Math.abs(result.coefficients[1] - 1), message).toBeLessThan(1e-6);
  // What float32 leaves (frc-regression-14 of the re-review of round three): sigma2 is 33
  // times the float64 value, not because of the residual arithmetic (1.04 to 1.65 times) but
  // because the residuals are e0 - rho e1 of two float32 OLS fits whose slopes are each
  // rounded, so that they are not orthogonal to [1, x]: 91% of their sum of squares lies in
  // that span, and with it removed the ratio is 2.9. Both sizes are pinned from above.
  const rowCount = scene.response.length;
  const sigmaRatio =
    result.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SIGMA_SQUARED] /
    (reference.residualSumOfSquares / rowCount);
  const predictorMean = scene.predictor.reduce((sum, value) => sum + value, 0) / rowCount;
  const residualMean = result.residuals.reduce((sum, value) => sum + value, 0) / rowCount;
  let predictorSquares = 0;
  let cross = 0;
  let residualSquares = 0;
  for (let row = 0; row < rowCount; row++) {
    const centered = scene.predictor[row] - predictorMean;
    predictorSquares += centered * centered;
    cross += centered * result.residuals[row];
    residualSquares += result.residuals[row] * result.residuals[row];
  }
  const spanShare =
    (rowCount * residualMean * residualMean + (cross * cross) / predictorSquares) / residualSquares;
  const limitMessage = `sigma2 ratio ${sigmaRatio}, share in the span of [1, x] ${spanShare}`;
  expect(sigmaRatio, limitMessage).toBeGreaterThan(0);
  expect(sigmaRatio, limitMessage).toBeLessThan(45);
  expect(sigmaRatio * (1 - spanShare), limitMessage).toBeLessThan(6);
});

it('lag maximum likelihood rejects a rho displaced from the stationary point and accepts the search result (frc2-regression-02)', async () => {
  // The mutation the second review asked for: the searched rho is moved between the search and
  // every later node. On a high-fit scene 1e-4 is hundreds of standard errors.
  const highFit = fixture.scenes.find(
    candidate => candidate.name === 'high-fit-slope-200-center-10000-stream-1001'
  )!;
  const inputs = createInputs(highFit);
  const run = (parameterOffset: number) =>
    runEstimator(
      'lag-maximum-likelihood',
      inputs.rows,
      inputs.predictors,
      inputs.response,
      parameterOffset
    );
  const plain = await runEstimator(
    'lag-maximum-likelihood',
    inputs.rows,
    inputs.predictors,
    inputs.response
  );
  // The added node with a zero offset changes nothing.
  const neutral = await run(0);
  expect(plain.status).toBe(0);
  expect(neutral.status).toBe(0);
  expect(neutral.coefficients).toEqual(plain.coefficients);
  for (const parameterOffset of [1e-4, -1e-4]) {
    const displaced = await run(parameterOffset);
    const message = `offset ${parameterOffset} GPU ${JSON.stringify(displaced.coefficients)}`;
    expect(displaced.status, message).toBe(3);
    expect(displaced.coefficients.every(Number.isNaN), message).toBe(true);
    expect(displaced.standardErrors.every(Number.isNaN), message).toBe(true);
    expect(displaced.summary.slice(1).every(Number.isNaN), message).toBe(true);
  }
  // An ordinary fit (the scale scene at scale 1, standard error of rho 0.0186): a displacement
  // of a fifth of a standard error is within the half standard error the check allows, one of
  // a whole standard error is not.
  const scene = EDGE_CASES.responseScale;
  const response = Float32Array.from(scene.baseResponse);
  const standardError = scene.scales[1].standardErrors[2];
  for (const [fraction, status] of [
    [0.2, 0],
    [-0.2, 0],
    [1, 3],
    [-1, 3]
  ]) {
    const result = await runEstimator(
      'lag-maximum-likelihood',
      getWeightRows('ring'),
      [scene.predictor],
      response,
      fraction * standardError
    );
    expect(result.status, `displaced by ${fraction} standard errors`).toBe(status);
  }
});

it('two-stage least squares on signed weights whose total is zero has a defined Anselin-Kelejian statistic (frc2-regression-07)', async () => {
  // Four reciprocal pairs with weights +1, +1, -1, -1: S0 = 0. Moran's I divides by S0; the
  // statistic does not (S0 cancels). Before the fix the statistic was infinity over infinity.
  const scene = EDGE_CASES.signedWeightsOfZeroTotal;
  const reference = scene.twoStage;
  expect(scene.weights.reduce((sum, value) => sum + value, 0)).toBe(0);
  const result = await runEstimator(
    'two-stage',
    getWeightRowsOf(scene),
    [scene.predictor],
    scene.response
  );
  const message = `GPU ${JSON.stringify({coefficients: result.coefficients, standardErrors: result.standardErrors, summary: result.summary.slice(0, 7)})}`;
  expect(result.status, message).toBe(0);
  expect(getCoefficientErrorOverStandardError(result, reference), message).toBeLessThan(1e-5);
  reference.standardErrors.forEach((standardError, row) => {
    expect(getRelativeError(result.standardErrors[row], standardError), message).toBeLessThan(1e-5);
  });
  expect(
    getRelativeError(
      result.summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_SIGMA_SQUARED],
      reference.sigmaSquared
    ),
    message
  ).toBeLessThan(1e-5);
  expect(
    getRelativeError(
      result.summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_ANSELIN_KELEJIAN],
      reference.anselinKelejian
    ),
    message
  ).toBeLessThan(1e-5);
  // Moran's I of the residuals is undefined: NaN, not infinity.
  expect(
    Number.isNaN(result.summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_MORAN_I]),
    message
  ).toBe(true);
});
