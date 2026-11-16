// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCommandGraph,
  createTransientView,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {expect, expectTypeOf, it} from 'vitest';
import type {GPUTemporalContext} from '../../../src/analysis-core/contracts/temporal-context';
import {GPULocalKriging} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {
  GPUSpaceTimeKriging,
  type GPUSpaceTimeKrigingProps
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-space-time-kriging';
import {
  getGPULocalKrigingParameterLength,
  getGPUSpaceTimeKrigingParameterValues,
  GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/prediction-covariance';
import {createNullWebGPUDevice, createVectorView} from '../../utils/gpu-contributor-test-utils';

const CONTEXT: GPUTemporalContext = {
  unit: 'second',
  epoch: 100n,
  encoding: 'float32-relative',
  axis: {kind: 'irregular'}
};

function createViews() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const base = {
    positions: view('float32x2', 4),
    times: view('float32', 4),
    observations: view('float32x4', 4),
    queries: view('float32x2', 2),
    queryTimes: view('float32', 2),
    mode: 'ordinary' as const,
    parameters: view('float32', GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH),
    output: {values: view('float32x4', 2), status: view('uint32', 2)}
  };
  return {graph, view, base};
}

it('preserves packed public types and infers the supplied query and output view kinds', () => {
  const {base} = createViews();
  const packedProps: GPUSpaceTimeKrigingProps = {...base, temporalContext: CONTEXT};
  const packed: GPUSpaceTimeKriging = new GPUSpaceTimeKriging(packedProps);
  const inferredPacked = new GPUSpaceTimeKriging(packedProps);
  expectTypeOf(packed.props.queries).toEqualTypeOf<GraphDataView<'float32x2'>>();
  expectTypeOf(packed.props.queryTimes).toEqualTypeOf<GraphDataView<'float32'>>();
  expectTypeOf(packed.props.output.values).toEqualTypeOf<GraphDataView<'float32x4'>>();
  expectTypeOf(packed.props.output.status).toEqualTypeOf<GraphDataView<'uint32'>>();
  expectTypeOf(inferredPacked.props.output.values).toEqualTypeOf<GraphDataView<'float32x4'>>();
  const chunked = new GPUSpaceTimeKriging({
    ...packedProps,
    queries: createVectorView('queries', 'float32x2', [base.queries]),
    queryTimes: createVectorView('query-times', 'float32', [base.queryTimes]),
    output: {
      values: createVectorView('values', 'float32x4', [base.output.values]),
      status: createVectorView('status', 'uint32', [base.output.status])
    }
  });
  expectTypeOf(chunked.props.queries).toEqualTypeOf<GraphVectorView<'float32x2'>>();
  expectTypeOf(chunked.props.queryTimes).toEqualTypeOf<GraphVectorView<'float32'>>();
  expectTypeOf(chunked.props.output.values).toEqualTypeOf<GraphVectorView<'float32x4'>>();
  expectTypeOf(chunked.props.output.status).toEqualTypeOf<GraphVectorView<'uint32'>>();
  const mixed = new GPUSpaceTimeKriging({...packedProps, output: chunked.props.output});
  expectTypeOf(mixed.props.queries).toEqualTypeOf<GraphDataView<'float32x2'>>();
  expectTypeOf(mixed.props.output.values).toEqualTypeOf<GraphVectorView<'float32x4'>>();
  expect(packed.props.output).toBe(base.output);
  expect(mixed.props.output).toBe(chunked.props.output);
});

it('resolves the temporal context and checks sample and query compatibility', () => {
  const {base} = createViews();
  expect(() => new GPUSpaceTimeKriging({...base, temporalContext: CONTEXT})).not.toThrow();
  expect(
    () =>
      new GPUSpaceTimeKriging({
        ...base,
        temporalContext: {...CONTEXT, encoding: 'int64-words', epoch: 'unix'}
      })
  ).toThrow('float32-relative');
  expect(
    () =>
      new GPUSpaceTimeKriging({
        ...base,
        temporalContext: CONTEXT,
        queryTemporalContext: {...CONTEXT, unit: 'millisecond'}
      })
  ).toThrow('incompatible temporal contexts');
  expect(
    () =>
      new GPUSpaceTimeKriging({
        ...base,
        temporalContext: CONTEXT,
        queryTemporalContext: {...CONTEXT, epoch: 200n}
      })
  ).toThrow('incompatible temporal contexts');
});

it('packs the documented space-time layout for each model and validates it', () => {
  const packed = getGPUSpaceTimeKrigingParameterValues({
    covariance: {
      type: 'metric',
      joint: {model: 'matern-5/2', nugget: 0.5, sill: 2, range: 3},
      anisotropy: 0.25
    },
    searchRadius: 10,
    maximumTimeLag: 6,
    mean: 7
  });
  expect(packed[0]).toBe(2.5);
  expect(packed[1]).toBe(2);
  expect(packed[2]).toBe(0.25);
  expect(Array.from(packed.subarray(12, 16))).toEqual([4, 0.5, 2, 3]);
  expect(Array.from(packed.subarray(24, 28))).toEqual([10, 6, 0.25, 7]);
  const gneiting = getGPUSpaceTimeKrigingParameterValues({
    covariance: {
      type: 'gneiting',
      nugget: 0.1,
      sill: 1,
      spatialRange: 2,
      temporalRange: 4,
      alpha: 0.5,
      gamma: 0.5,
      beta: 0.5,
      tau: 1
    },
    searchAnisotropy: 1
  });
  expect(gneiting[1]).toBe(4);
  expect(gneiting[18]).toBeCloseTo((Math.exp(3) - 1) / 4, 5);
  expect(gneiting[19]).toBeCloseTo(3 / 2, 5);
  expect(() =>
    getGPUSpaceTimeKrigingParameterValues({
      covariance: {
        type: 'separable',
        sill: 1,
        spatial: {model: 'exponential', nugget: 0.1, sill: 0.5, range: 1},
        temporal: {model: 'exponential', nugget: 0, sill: 1, range: 1}
      },
      searchAnisotropy: 1
    })
  ).toThrow('standardized');
  expect(() =>
    getGPUSpaceTimeKrigingParameterValues({
      covariance: {
        type: 'gneiting',
        nugget: 0,
        sill: 1,
        spatialRange: 1,
        temporalRange: 1,
        alpha: 0.5,
        gamma: 0.5,
        beta: 0.8,
        tau: 0.5
      },
      searchAnisotropy: 1
    })
  ).toThrow('Gneiting');
  expect(() =>
    getGPUSpaceTimeKrigingParameterValues({
      covariance: {
        type: 'gneiting',
        nugget: 0,
        sill: 1,
        spatialRange: 1,
        temporalRange: 1,
        alpha: 0.5,
        gamma: 0.5,
        beta: 0,
        tau: 1
      }
    })
  ).toThrow('searchAnisotropy');
});

it('rejects a neighbor budget outside the supported capacities', () => {
  const {base} = createViews();
  expect(
    () =>
      new GPUSpaceTimeKriging({
        ...base,
        temporalContext: CONTEXT,
        neighborCount: 8,
        neighborSearch: 'staged',
        stagedCandidateCount: 4
      })
  ).toThrow('stagedCandidateCount');
  expect(
    () =>
      new GPUSpaceTimeKriging({
        ...base,
        temporalContext: CONTEXT,
        neighborSearch: 'octant',
        neighborsPerOctant: 9
      })
  ).toThrow('neighborsPerOctant');
});

it('rejects non-finite, fractional and oversized interpolated capacities at construction', () => {
  const {base} = createViews();
  for (const property of ['neighborCount', 'neighborsPerOctant', 'stagedCandidateCount']) {
    for (const value of [NaN, Infinity, -Infinity, 1e100, 0x100000000, 1.5]) {
      expect(
        () =>
          new GPUSpaceTimeKriging({
            ...base,
            id: 'literal-validation',
            temporalContext: CONTEXT,
            [property]: value
          })
      ).toThrow(`literal-validation ${property} must be an integer in [`);
    }
  }
});

it('fails with a clear error when the kernel would exceed the device storage-binding limit', () => {
  const {graph, view, base} = createViews();
  // Drift + weights + parameters: 9 storage bindings on a device limited to 8.
  const contributor = new GPULocalKriging({
    positions: base.positions,
    observations: base.observations,
    supports: base.queries,
    mode: 'external-drift',
    driftCount: 1,
    sampleDrift: view('float32', 4),
    queryDrift: view('float32', 2),
    parameters: view('float32', getGPULocalKrigingParameterLength()),
    output: {
      values: view('float32x4', 2),
      status: view('uint32', 2),
      weights: view('float32', 16 * 2)
    }
  });
  expect(() => graph.add(contributor)).toThrow('local-kriging storage bindings must be at most 8');
});
