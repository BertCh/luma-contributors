// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {expect, expectTypeOf, it} from 'vitest';
import {
  GPUNearestFeatureJoin,
  type GPUNearestFeatureJoinOutput
} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import type {GPUInverseDistanceWeightingProps} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-inverse-distance-weighting';
import type {GPUNearestFeatureWeightsProps} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-weights';
import {createNullWebGPUDevice, createVectorView} from '../../utils/gpu-contributor-test-utils';

it('keeps exported packed defaults and infers concrete packed and chunked views', () => {
  expectTypeOf<GPUNearestFeatureJoinOutput['nearestFeatureIds']>().toEqualTypeOf<
    GraphDataView<'uint32'> | undefined
  >();
  expectTypeOf<GPUNearestFeatureJoinOutput['nearestDistances']>().toEqualTypeOf<
    GraphDataView<'float32'> | undefined
  >();
  expectTypeOf<GPUInverseDistanceWeightingProps['output']['values']>().toEqualTypeOf<
    GraphDataView<'float32'>
  >();
  expectTypeOf<GPUNearestFeatureWeightsProps['output']['neighbors']>().toEqualTypeOf<
    GraphDataView<'uint32'>
  >();
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32x2' | 'float32' | 'uint32'>(format: Format) =>
    createTransientView(graph, `nearest-types-${serial++}`, format, 1);
  const positions = view('float32x2');
  const ids = view('uint32');
  const distances = view('float32');
  const packed = new GPUNearestFeatureJoin({
    queries: {kind: 'points', positions},
    features: {kind: 'points', positions: view('float32x2')},
    radius: view('float32'),
    candidateCapacity: 2,
    output: {
      nearestFeatureIds: ids,
      nearestDistances: distances,
      status: {overflow: view('uint32')}
    }
  });
  expectTypeOf(packed.props.output.nearestFeatureIds!).toEqualTypeOf<GraphDataView<'uint32'>>();
  expectTypeOf(packed.props.output.nearestDistances!).toEqualTypeOf<GraphDataView<'float32'>>();
  expectTypeOf(packed.props.queries.positions).toEqualTypeOf<GraphDataView<'float32x2'>>();
  const packedClass: GPUNearestFeatureJoin = packed;
  expect(packedClass.props.output.nearestFeatureIds).toBe(ids);

  const chunkedIds = createVectorView('chunked-type-ids', 'uint32', [view('uint32')]);
  const chunkedDistances = createVectorView('chunked-type-distances', 'float32', [view('float32')]);
  const chunked = new GPUNearestFeatureJoin({
    queries: {
      kind: 'points',
      positions: createVectorView('chunked-type-points', 'float32x2', [view('float32x2')])
    },
    features: {kind: 'points', positions: view('float32x2')},
    radius: view('float32'),
    candidateCapacity: 2,
    output: {
      nearestFeatureIds: chunkedIds,
      nearestDistances: chunkedDistances,
      status: {overflow: view('uint32')}
    }
  });
  expectTypeOf(chunked.props.output.nearestFeatureIds!).toEqualTypeOf<GraphVectorView<'uint32'>>();
  expectTypeOf(chunked.props.output.nearestDistances!).toEqualTypeOf<GraphVectorView<'float32'>>();
  expectTypeOf(chunked.props.queries.positions).toEqualTypeOf<GraphVectorView<'float32x2'>>();
  expect(chunked.props.output.nearestFeatureIds).toBe(chunkedIds);
  device.destroy();
});
