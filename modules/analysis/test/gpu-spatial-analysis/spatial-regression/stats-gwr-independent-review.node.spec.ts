// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {decodeGPUMultiscaleGeographicallyWeightedRegressionStatusFields} from '../../../src/gpu-spatial-analysis/spatial-regression/regression-status-decoders';
import {getGPUGeographicallyWeightedRegressionParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {
  computeMultiscaleGeographicallyWeightedRegressionOnCPU,
  CPUMultiscaleGeographicallyWeightedRegressionOperation
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-multiscale-geographically-weighted-regression-operation';
import {
  GPUMultiscaleGeographicallyWeightedRegressionOperation,
  type MultiscaleGeographicallyWeightedRegressionOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression-operation';
import {constructRegressionOperationContributor} from './regression-operation-test-utils';
import {getMultiscaleBandwidthAllocationLengths} from '../../../src/gpu-spatial-analysis/spatial-regression/mgwr-bandwidth-selection';

for (const targetLength of [5, 6]) {
  it(`packs the indexed ArrayLike length despite a conflicting iterator: target=${targetLength}`, () => {
    const bandwidths = {
      length: 1,
      0: 2,
      *[Symbol.iterator]() {
        yield 2;
        yield 3;
      }
    };
    const target = new Float32Array(targetLength).fill(123);
    expect(() =>
      getGPUGeographicallyWeightedRegressionParameterValues({bandwidths}, 1, target)
    ).not.toThrow();
    expect(Array.from(target)).toEqual(targetLength === 5 ? [1, 0, 1, 0, 2] : [1, 0, 1, 0, 2, 123]);
  });
}

it('reads the supplied bandwidth count once before indexed snapshotting', () => {
  let reads = 0;
  const bandwidths = {
    get length() {
      reads++;
      return reads === 1 ? 1 : 2;
    },
    0: 2,
    1: 3
  };
  const target = new Float32Array(6).fill(123);
  getGPUGeographicallyWeightedRegressionParameterValues({bandwidths}, 1, target);
  expect(reads).toBe(1);
  expect(Array.from(target)).toEqual([1, 0, 1, 0, 2, 123]);
});

it('suppresses a coupled fit whose float64 slope overflows public float32 coefficients', () => {
  const result = computeMultiscaleGeographicallyWeightedRegressionOnCPU(
    {
      positions: new Float32Array(4),
      predictors: Float32Array.of(-1e-12, 1e-12),
      response: Float32Array.of(-1e28, 1e28),
      parameters: Float32Array.of(1e-6, 1, 1)
    },
    {predictorCount: 1, kernel: 'gaussian', maximumIterations: 1, inference: {}}
  );
  expect(Array.from(result.codes)).toEqual([4, 4]);
  expect(Array.from(result.coefficients).every(Number.isNaN)).toBe(true);
  expect(Array.from(result.fitted).every(Number.isNaN)).toBe(true);
  expect(result.inference!.status[0]).toBe(1);
});

it('suppresses a coupled fit whose finite coefficients give an unrepresentable fitted value', () => {
  const result = computeMultiscaleGeographicallyWeightedRegressionOnCPU(
    {
      positions: new Float32Array(6),
      predictors: Float32Array.of(-0.5, 0, 1),
      response: Float32Array.of(-1e38, 1e38, 3.1e38),
      parameters: Float32Array.of(1e-6, 1, 1)
    },
    {predictorCount: 1, kernel: 'gaussian', maximumIterations: 1, inference: {}}
  );
  // Mean y=31/30*1e38 and slope=37/15*1e38 are representable; the last prediction is 3.5e38.
  expect(Array.from(result.codes)).toEqual([4, 4, 4]);
  expect(Array.from(result.coefficients).every(Number.isNaN)).toBe(true);
  expect(Array.from(result.fitted).every(Number.isNaN)).toBe(true);
});

it('reports component search exhaustion separately from a converged backfit', () => {
  const result = computeMultiscaleGeographicallyWeightedRegressionOnCPU(
    {
      positions: Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0),
      predictors: Float32Array.of(1, -1, 1, -1),
      response: Float32Array.of(1, 6, 2, 8),
      parameters: Float32Array.of(100, 1, 1),
      bandwidthCandidates: Float32Array.of(1, 4, 1, 4)
    },
    {
      predictorCount: 1,
      kernel: 'gaussian',
      maximumIterations: 1,
      inference: {},
      bandwidthCandidateCount: 2,
      bandwidthSelection: {searchMethod: 'golden-section', maximumSearchIterations: 1}
    }
  );
  expect(Array.from(result.codes)).toEqual([0, 0, 0, 0]);
  expect(result.inference!.status[0]).toBe(0);
  expect(Array.from(result.selectedBandwidths)).toEqual([4, 4]);
  expect(Array.from(result.componentSearchIterationLimitReached)).toEqual([1, 1]);
  const tied = computeMultiscaleGeographicallyWeightedRegressionOnCPU(
    {
      positions: Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0),
      predictors: Float32Array.of(1, -1, 1, -1),
      response: new Float32Array(4).fill(5),
      parameters: Float32Array.of(100, 1, 1),
      bandwidthCandidates: Float32Array.of(1, 4, 1, 4)
    },
    {
      predictorCount: 1,
      kernel: 'gaussian',
      maximumIterations: 1,
      inference: {},
      bandwidthCandidateCount: 2,
      bandwidthSelection: {searchMethod: 'golden-section', maximumSearchIterations: 1}
    }
  );
  expect(Array.from(tied.componentSearchIterationLimitReached)).toEqual([0, 0]);
});

it('keeps a constructed CPU operation sweep budget across two executions', async () => {
  const props: MultiscaleGeographicallyWeightedRegressionOperationProps = {
    rowCount: 3,
    predictorCount: 1,
    kernel: 'bisquare',
    maximumIterations: 1,
    positions: 'positions',
    predictors: 'predictors',
    response: 'response',
    parameters: 'parameters',
    output: {
      coefficients: 'coefficients',
      fitted: 'fitted',
      status: {codes: 'codes', iterationCount: 'iterationCount'}
    }
  };
  const operation = new CPUMultiscaleGeographicallyWeightedRegressionOperation(props);
  const gpuOperation = new GPUMultiscaleGeographicallyWeightedRegressionOperation(props);
  const vector = (values: Float32Array, format: 'float32' | 'float32x2' = 'float32') => ({
    format,
    length: values.length / (format === 'float32x2' ? 2 : 1),
    chunks: [{values, length: values.length / (format === 'float32x2' ? 2 : 1)}],
    pack: () => values
  });
  const inputs = {
    positions: vector(Float32Array.of(0, 0, 1, 0, 2, 0), 'float32x2'),
    predictors: vector(Float32Array.of(1, -1, 1)),
    response: vector(Float32Array.of(2, 4, 8)),
    parameters: vector(Float32Array.of(1e-6, 1, 1))
  };
  const first = await operation.execute({inputs});
  props.maximumIterations = 2;
  const contributor = constructRegressionOperationContributor(gpuOperation, {
    positions: {format: 'float32x2', length: 3},
    predictors: {format: 'float32', length: 3},
    response: {format: 'float32', length: 3},
    parameters: {format: 'float32', length: 3}
  });
  expect(contributor).toHaveProperty('props.maximumIterations', 1);
  const second = await operation.execute({inputs});
  for (const result of [first, second]) {
    expect(result['iterationCount']).toEqual(Uint32Array.of(1));
    expect(result['codes']).toEqual(Uint32Array.of(3, 3, 3));
  }
});

it('allocates all 66 three-word MGWR search rows at the 64-step limit', () => {
  expect(
    getMultiscaleBandwidthAllocationLengths(4, 2, {
      searchMethod: 'golden-section',
      maximumSearchIterations: 64
    })['bandwidth table']
  ).toBe(198);
});

it('decodes excluded MGWR rows and finite last iterates independently of website wording', () => {
  const input = {
    positions: Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0),
    predictors: Float32Array.of(1, -1, 1, 2),
    response: Float32Array.of(2, 4, 8, 100),
    parameters: Float32Array.of(1e-6, 1, 1),
    mask: Uint32Array.of(1, 1, 1, 0)
  };
  const unfinished = computeMultiscaleGeographicallyWeightedRegressionOnCPU(input, {
    predictorCount: 1,
    kernel: 'bisquare',
    maximumIterations: 1
  });
  const statuses = Array.from(
    unfinished.codes,
    decodeGPUMultiscaleGeographicallyWeightedRegressionStatusFields
  );
  expect(statuses.map(status => status.name)).toEqual([
    'iteration-limit',
    'iteration-limit',
    'iteration-limit',
    'excluded'
  ]);
  expect(statuses.map(status => status.excluded)).toEqual([false, false, false, true]);
  expect(statuses.map(status => status.converged)).toEqual([false, false, false, false]);
  expect(statuses.map(status => status.failed)).toEqual([false, false, false, false]);
  // h=1 bisquare admits only self. Global OLS is [4.5, 0.5]; the first intercept update is
  // y - 0.5*x and the unchanged slope is 0.5. All three fitted values therefore equal y.
  expect(Array.from(unfinished.coefficients.slice(0, 6)).every(Number.isFinite)).toBe(true);
  expect(Array.from(unfinished.coefficients)).toEqual([1.5, 0.5, 4.5, 0.5, 7.5, 0.5, NaN, NaN]);
  expect(Array.from(unfinished.fitted.slice(0, 3)).every(Number.isFinite)).toBe(true);
  expect(Array.from(unfinished.fitted)).toEqual([2, 4, 8, NaN]);
  const finished = computeMultiscaleGeographicallyWeightedRegressionOnCPU(input, {
    predictorCount: 1,
    kernel: 'bisquare',
    maximumIterations: 2
  });
  expect(
    Array.from(finished.codes, decodeGPUMultiscaleGeographicallyWeightedRegressionStatusFields).map(
      status => status.name
    )
  ).toEqual(['converged', 'converged', 'converged', 'excluded']);
});
