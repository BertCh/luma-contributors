// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {
  GPUMinimumBoundsOperation,
  type MinimumBoundsOperationProps
} from '../../../src/gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds-operation';
import {
  GPUConstrainedDelaunayTriangulationOperation,
  type ConstrainedDelaunayTriangulationOperationProps
} from '../../../src/gpu-spatial-analysis/polygon-triangulation/gpu-constrained-delaunay-triangulation-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

it('GPU minimum-bounds operation executes its original route and outputs after descriptor mutation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const spatialContext: GPUSpatialContext = {coordinateSpace: 'planar', metric: 'native'};
  const props: MinimumBoundsOperationProps = {
    positions: 'positions',
    labels: 'labels',
    rowCount: 3,
    groupCount: 1,
    spatialContext,
    maximumHullVertices: 8,
    output: {
      sourceBounds: 'bounds',
      hullSizes: 'sizes',
      status: {overflow: 'overflow', groupOverflow: 'groupOverflow'}
    }
  };
  const operation = new GPUMinimumBoundsOperation(props);
  Object.assign(props, {
    labels: undefined,
    ringOffsets: 'missing',
    polygonOffsets: 'missing',
    featureOffsets: 'missing',
    featureCount: 99,
    groupCount: 99,
    rowCount: 99,
    maximumHullVertices: 3
  });
  props.output.sourceBounds = undefined;
  props.output.hullSizes = 'changed';
  props.output.status.groupOverflow = undefined;
  props.output.status.overflow = 'changed';
  spatialContext.coordinateSpace = 'longitude-latitude';
  const result = await runOperationGraph(device, {
    inputs: {
      positions: {format: 'float32x2', values: new Float32Array([0, 0, 4, 0, 0, 3])},
      labels: {format: 'uint32', values: new Uint32Array([0, 0, 0])}
    },
    operations: [operation],
    outputs: ['bounds', 'sizes', 'overflow', 'groupOverflow']
  });
  // Vertex extrema of the 3-4-5 right triangle; all three points are hull vertices.
  expect(result['bounds'].length).toBe(4);
  result['bounds'].forEach(value => expect(Number.isFinite(value)).toBe(true));
  expect(Array.from(result['bounds'])).toEqual([0, 0, 4, 3]);
  expect(Array.from(result['sizes'])).toEqual([3]);
  expect(Array.from(result['overflow'])).toEqual([0]);
  expect(Array.from(result['groupOverflow'])).toEqual([0]);
});

it('GPU constrained operation executes original options and optional outputs after descriptor mutation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const spatialContext: GPUSpatialContext = {coordinateSpace: 'planar', metric: 'native'};
  const props: ConstrainedDelaunayTriangulationOperationProps = {
    spatialContext,
    positions: 'positions',
    featureOffsets: 'features',
    polygonOffsets: 'polygons',
    ringOffsets: 'rings',
    polygonCount: 1,
    indexCapacity: 6,
    maximumWork: 64,
    maximumFlips: 8,
    flipMethod: 'serial',
    largePolygonMethod: 'none',
    output: {
      indices: 'indices',
      valid: 'valid',
      triangleCount: 'count',
      triangleOffsets: 'offsets',
      status: {converged: 'converged'}
    }
  };
  const operation = new GPUConstrainedDelaunayTriangulationOperation(props);
  Object.assign(props, {
    polygonCount: 99,
    indexCapacity: 3,
    maximumWork: 1,
    maximumFlips: 1,
    flipMethod: 'parallel',
    largePolygonMethod: 'monotone',
    positions: 'changed'
  });
  props.output.triangleCount = undefined;
  props.output.triangleOffsets = undefined;
  props.output.status.converged = 'changed';
  spatialContext.coordinateSpace = 'longitude-latitude';
  const result = await runOperationGraph(device, {
    inputs: {
      positions: {format: 'float32x2', values: new Float32Array([0, 0, 4, 0, 4, 4, 0, 1])},
      features: {format: 'uint32', values: new Uint32Array([0, 1])},
      polygons: {format: 'uint32', values: new Uint32Array([0, 1])},
      rings: {format: 'uint32', values: new Uint32Array([0, 4])}
    },
    operations: [operation],
    outputs: ['indices', 'valid', 'count', 'offsets', 'converged']
  });
  // Only diagonal 1--3 is Delaunay. Centers (19/8,2) and (2,1/2) have squared
  // radii 425/64 and 17/4; opposite-point squared distances are 617/64 and 65/4.
  expect(Array.from(result['indices'])).toEqual([3, 1, 2, 1, 3, 0]);
  expect(Array.from(result['valid'])).toEqual([1]);
  expect(Array.from(result['count'])).toEqual([2]);
  expect(Array.from(result['offsets'])).toEqual([0, 2]);
  expect(Array.from(result['converged'])).toEqual([1]);
});
