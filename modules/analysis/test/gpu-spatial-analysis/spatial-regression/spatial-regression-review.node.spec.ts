// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUSpatialTwoStageLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';
import {computeSpatialTwoStageLeastSquaresOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-two-stage-least-squares-operation';
import {computeSpatialErrorGMOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-error-gm-operation';
import {computeSpatialRegressionDiagnosticsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-regression-diagnostics-operation';
import {GPUSpatialErrorGMOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm-operation';
import {GPUSpatialTwoStageLeastSquaresOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares-operation';
import {GPUSpatialRegressionDiagnosticsOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regression-diagnostics-operation';
import {getGPUSpatialTwoStageLeastSquaresModelResult} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-two-stage-least-squares-model-result';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {constructRegressionOperationContributor} from './regression-operation-test-utils';
import {addModelResultReadouts} from '../../../../../examples/deck/spatial-analysis-explorer/modes/regression-result-readouts';
import {getRegressionResidualForDisplay} from '../../../../../examples/deck/spatial-analysis-shared/regression-residual-display';
import {
  createReviewInput,
  createReviewKernel,
  createAffineReviewInput,
  createRationalAffineReviewInput,
  createConstantFittedReviewInput,
  createLongAffineReviewInput,
  createCycleReviewInput,
  createCompleteReviewInput,
  createSaturatedReviewInput
} from './spatial-regression-review-fixtures';
import {
  createBipartiteReviewInput,
  createTranslatedRationalAffineInput,
  createWeightedConstantFittedInput
} from './regression-diagnostics-w5-fixtures';

for (const estimator of ['homoskedastic', 'kelejian-prucha'] as const)
  it(`review 2 R3 ${estimator} recognizes a rational slope and preserves a one-ULP departure`, () => {
    for (const departure of [false, true]) {
      const result = computeSpatialErrorGMOnCPU(createRationalAffineReviewInput(departure), {
        predictorCount: 1,
        estimator
      });
      // Exact relation x=7y; changing y[0] increases augmented rational rank from 2 to 3.
      expect(Array.from(result.status)).toEqual([departure ? 0 : 3]);
    }
  });

it('review 2 R2 constant fitted lag has J=T=64 and undefined robust tests', () => {
  const result = computeSpatialRegressionDiagnosticsOnCPU(createConstantFittedReviewInput(), {
    predictorCount: 1
  });
  expect(Array.from(result.status)).toEqual([0]);
  expect(result.summary[3]).toBe(64);
  for (const row of [0, 1]) expect(result.tests[3 * row]).toBe(0);
  for (const row of [2, 3, 4]) expect(result.tests[3 * row]).toBeNaN();
});

for (const [increment, variance] of [
  [2 ** -20, 8.7005227e-16],
  [1 / 32, 9.3212977e-7]
])
  it(`review 2 R7 nearly complete graph preserves positive variance at increment ${increment}`, () => {
    const result = computeSpatialRegressionDiagnosticsOnCPU(
      createCompleteReviewInput(8, increment),
      {predictorCount: 1}
    );
    expect(result.summary[7] / variance).toBeCloseTo(1, 6);
    expect(result.summary[8]).toBeCloseTo(1.15470054, 6);
    expect(result.summary[9]).toBeCloseTo(0.24821308, 6);
  });

it('review 2 R6 explorer renders the decoded HAC standard error convention', () => {
  const result = computeSpatialTwoStageLeastSquaresOnCPU(
    {...createReviewInput(), robust: {method: 'HAC', kernel: createReviewKernel()}},
    {predictorCount: 2}
  );
  const readouts = new Map<string, string>();
  const controls = {
    addReadout: (label: string) => ({setValue: (value: string) => readouts.set(label, value)})
  };
  addModelResultReadouts(controls, '', []).update(
    getGPUSpatialTwoStageLeastSquaresModelResult(result)
  );
  expect(readouts.get('σ² (error scale) / ML σ²')).toContain('SEs use robust covariance');
});

it('review 2 R11 tooltips hide finite residuals after code 2 and recover on code 0', () => {
  for (const residual of [0, -0.125, 1.25]) {
    expect(getRegressionResidualForDisplay(2, residual)).toBeUndefined();
    expect(getRegressionResidualForDisplay(0, residual)).toBe(residual);
  }
  expect(getRegressionResidualForDisplay(undefined, 0)).toBeUndefined();
  expect(getRegressionResidualForDisplay(0, NaN)).toBeUndefined();
});

it('R1 robust 2SLS uses active residual rows with spare capacity and a nonzero offset', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const view = <Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, name, format, length);
    const backing = view('residual-storage', 'float32', 16);
    const residuals = graph.createDataView(backing.buffer, {
      format: 'float32',
      byteOffset: backing.byteOffset + 8,
      length: 13
    });
    const contributor = new GPUSpatialTwoStageLeastSquares({
      predictorCount: 2,
      predictors: view('predictors', 'float32', 24),
      response: view('response', 'float32', 12),
      weights: {
        offsets: view('offsets', 'uint32', 13),
        neighbors: view('neighbors', 'uint32', 22),
        weights: view('weights', 'float32', 22)
      },
      robust: {
        method: 'HC0',
        output: {
          covariance: view('covariance', 'float32', 16),
          status: {code: view('covariance-status', 'uint32', 1)}
        }
      },
      output: {
        table: view('table', 'float32', 16),
        summary: view('summary', 'float32', 16),
        residuals,
        status: {code: view('code', 'uint32', 1)}
      }
    });
    const scores = contributor
      .getCommandNodes(graph)
      .find(node => node.id.endsWith('-robust-scores'))!;
    const active = scores.resources
      ?.flatMap(resource => ('buffer' in resource ? [resource.buffer as GraphDataView] : []))
      .find(view => view.buffer === residuals.buffer);
    expect(active?.length).toBe(12);
    expect(active?.byteOffset).toBe(residuals.byteOffset);
  } finally {
    device.destroy();
  }
});

for (const estimator of ['homoskedastic', 'kelejian-prucha'] as const) {
  it(`R3 ${estimator} rejects exactly affine response without rejecting a one-ULP departure`, () => {
    const exact = computeSpatialErrorGMOnCPU(createAffineReviewInput(), {
      predictorCount: 2,
      estimator
    });
    expect(Array.from(exact.status)).toEqual([3]);
    expect(exact.table.every(Number.isNaN)).toBe(true);
    const perturbed = computeSpatialErrorGMOnCPU(createAffineReviewInput(true), {
      predictorCount: 2,
      estimator
    });
    expect(Array.from(perturbed.status)).toEqual([0]);
    expect(perturbed.summary[4]).toBeGreaterThan(0);
    const tooSmall = computeSpatialErrorGMOnCPU(
      {
        predictors: Float32Array.of(0, 1, 2),
        response: Float32Array.of(1, 2, 3),
        weights: {
          offsets: Uint32Array.of(0, 1, 2, 3),
          neighbors: Uint32Array.of(1, 2, 0),
          weights: Float32Array.of(1, 1, 1)
        }
      },
      {predictorCount: 1, estimator}
    );
    expect(Array.from(tooSmall.status)).toEqual([2]);
  });
}

it('R4 saturated first-stage F is undefined while the point fit and partial R2 are valid', () => {
  const result = computeSpatialTwoStageLeastSquaresOnCPU(createSaturatedReviewInput(), {
    predictorCount: 2
  });
  expect(Array.from(result.status)).toEqual([0]);
  expect(result.summary[14]).toBeNaN();
  expect(result.summary[15]).toBe(1);
});

it('R3 an unsampled one-ULP departure is a genuine residual in the 300-row affine fixture', () => {
  // review.oracle.py exact rational elimination: rank(X)=3, rank([X,y])=3/4.
  for (const departure of [false, true]) {
    const result = computeSpatialErrorGMOnCPU(createLongAffineReviewInput(departure), {
      predictorCount: 2,
      estimator: 'kelejian-prucha'
    });
    expect(Array.from(result.status)).toEqual([departure ? 0 : 3]);
    if (departure) expect(result.summary[4]).toBeGreaterThan(0);
  }
});

it('R6 model readback preserves HAC method and separate inference failure', () => {
  for (const loss of [0, 1]) {
    const result = computeSpatialTwoStageLeastSquaresOnCPU(
      {
        ...createReviewInput(),
        robust: {
          method: 'HAC',
          kernel: {...createReviewKernel(), status: {candidateOverflow: Uint32Array.of(loss)}}
        }
      },
      {predictorCount: 2}
    );
    const decoded = getGPUSpatialTwoStageLeastSquaresModelResult(result);
    expect(decoded.status.ok).toBe(true);
    expect(decoded.standardErrorSigmaSquared).toBe('robust');
    expect(decoded).toHaveProperty('covarianceInference.method', 'HAC');
    expect(decoded).toHaveProperty('covarianceInference.status.code', loss ? 4 : 0);
    expect(decoded).toHaveProperty('covarianceInference.status.ok', !loss);
    if (loss)
      expect(
        decoded.coefficients.every(coefficient => Number.isNaN(coefficient.standardError))
      ).toBe(true);
    else expect(decoded.coefficients[0].standardError).toBeCloseTo(0.51024556, 6);
  }
});

for (const count of [8, 12])
  it(`R7 complete ${count}-row graph has zero Moran variance and undefined z/p`, () => {
    // W=11'-I and M1=0 imply MWM=-M. Thus I=-1/(n-1) identically, Var(I)=0.
    const result = computeSpatialRegressionDiagnosticsOnCPU(createCompleteReviewInput(count), {
      predictorCount: 1
    });
    expect(Array.from(result.status)).toEqual([0]);
    expect(result.summary[5]).toBeCloseTo(-1 / (count - 1), 7);
    expect(result.summary[7]).toBe(0);
    for (const slot of [8, 9]) expect(result.summary[slot]).toBeNaN();
    for (const slot of [15, 17]) expect(result.tests[slot]).toBeNaN();
  });

it('R2 CPU directed-cycle oracle retains J and all LM statistics under response translation', () => {
  for (const offset of [0, 4096]) {
    const result = computeSpatialRegressionDiagnosticsOnCPU(createCycleReviewInput(16, offset), {
      predictorCount: 1
    });
    // x and Wx are orthogonal, e=Wx, W fitted=-x shifted by offset: ||MW fitted||²=16.
    expect(result.summary[3]).toBe(32);
    for (const [row, value] of [8, 0, 16, 8, 16].entries())
      expect(result.tests[row * 3]).toBe(value);
  }
});

for (const exponent of [18, 29, 30])
  it(`R8/R12 CPU homoskedastic GM is response-scale equivariant at 2^${exponent}`, () => {
    const input = createReviewInput();
    const result = computeSpatialErrorGMOnCPU(
      {...input, response: Float32Array.from(input.response, value => value * 2 ** exponent)},
      {predictorCount: 2}
    );
    expect(Array.from(result.status)).toEqual([0]);
    expect(result.table[12]).toBeCloseTo(-0.9621903664806921, 6);
    expect(result.table[13]).toBeCloseTo(0.5414640317274417, 6);
    expect(result.summary[3]).toBeCloseTo(0.7094877554580391, 6);
  });

for (const Operation of [
  GPUSpatialErrorGMOperation,
  GPUSpatialTwoStageLeastSquaresOperation,
  GPUSpatialRegressionDiagnosticsOperation
])
  it(`R9 ${Operation.name} enforces its declared row count`, () => {
    const operation = new Operation({
      id: 'declared-rows',
      rowCount: 8,
      predictorCount: 2,
      weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      predictors: 'predictors',
      response: 'response',
      residuals: 'residuals',
      output: {table: 'table', summary: 'summary', tests: 'tests', status: {code: 'code'}}
    });
    expect(() =>
      constructRegressionOperationContributor(operation, {
        offsets: {format: 'uint32', length: 13},
        neighbors: {format: 'uint32', length: 22},
        weights: {format: 'float32', length: 22},
        predictors: {format: 'float32', length: 24},
        response: {format: 'float32', length: 12},
        residuals: {format: 'float32', length: 12}
      })
    ).toThrow(/declared-rows.*response.*rowCount/);
  });

it('R10 HAC rejects a nonzero first CSR offset without discarding the point fit', () => {
  const result = computeSpatialTwoStageLeastSquaresOnCPU(
    {...createReviewInput(), robust: {method: 'HAC', kernel: createReviewKernel(true)}},
    {predictorCount: 2}
  );
  expect(Array.from(result.status)).toEqual([0]);
  expect(Array.from(result.robust!.status)).toEqual([4]);
  expect(result.robust!.covariance.every(Number.isNaN)).toBe(true);
});

it('W5-30 K(8,8) with a partition design has zero Moran variance and undefined inference', () => {
  // W = ab' + ba' with a, b in the design span: MWM = 0 exactly, so Var(I) = 0 and z/p are 0/0.
  // Before the fix, projection roundoff gave Var 3.17e-33, z 2.46 and p 0.0138.
  const result = computeSpatialRegressionDiagnosticsOnCPU(createBipartiteReviewInput(), {
    predictorCount: 1
  });
  expect(Array.from(result.status)).toEqual([0]);
  expect(result.summary[5]).toBe(0);
  expect(result.summary[7]).toBe(0);
  for (const slot of [8, 9]) expect(result.summary[slot]).toBeNaN();
  for (const slot of [15, 17]) expect(result.tests[slot]).toBeNaN();
});

// NumPy float64 dense Cliff-Ord moments of the same K(8,8) operator with the first link raised
// (`.typecheck/r19/scratch/spatial-stats-w5-regression-weights-and-diagnostics/k88_reference.py`).
for (const [increment, variance, z] of [
  [2 ** -20, 4.857225660356756e-17, 1.0690449676496976],
  [1 / 32, 5.2128607649655715e-8, 1.0690449676496976]
])
  it(`W5-30 nearly bipartite control keeps its positive variance at increment ${increment}`, () => {
    const result = computeSpatialRegressionDiagnosticsOnCPU(createBipartiteReviewInput(increment), {
      predictorCount: 1
    });
    expect(Number.isFinite(result.summary[7])).toBe(true);
    expect(Math.abs(result.summary[7] / variance - 1)).toBeLessThan(1e-6);
    expect(Math.abs(result.summary[8] - z)).toBeLessThan(1e-6);
  });

it('W5-31 CPU weighted constant fitted lag has J=T=35 and undefined robust tests', () => {
  // tr(W'W) = 32 + 32 * 32 / 32^2 = 33 and tr(W^2) = 2 * 32 / 32 = 2, so T = 35; MW fitted = 0.
  const result = computeSpatialRegressionDiagnosticsOnCPU(createWeightedConstantFittedInput(), {
    predictorCount: 1
  });
  expect(Array.from(result.status)).toEqual([0]);
  expect(result.summary[2]).toBe(35);
  expect(result.summary[3]).toBe(35);
  for (const row of [2, 3, 4]) expect(result.tests[3 * row]).toBeNaN();
});

for (const estimator of ['homoskedastic', 'kelejian-prucha'] as const)
  it(`W5-32 CPU ${estimator} certifies 7y = x + 16777217 and keeps a one-ULP departure`, () => {
    for (const departure of [false, true]) {
      const result = computeSpatialErrorGMOnCPU(createTranslatedRationalAffineInput(departure), {
        predictorCount: 1,
        estimator
      });
      expect(Array.from(result.status)).toEqual([departure ? 0 : 3]);
    }
  });
