// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUSpatialJoinPrepared} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared';
import {GPUPointInPolygonJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join';
import {GPUSpatialPredicateJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import {GPUSpatialJoinCandidates} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-candidates';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import type {GPUSpatialJoinGeometry} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-join-types';

/** One square containing the origin; translating every vertex by ten makes every query miss. */
export const PREPARED_OLD_SQUARE = Float32Array.of(-1, -1, 1, -1, 1, 1, -1, 1);
export const PREPARED_NEW_SQUARE = Float32Array.from(PREPARED_OLD_SQUARE, value => value + 10);

export function createPreparedConsumerNodes(
  graph: GPUCommandGraph,
  prepared: GPUSpatialJoinPrepared<void>,
  polygons: Extract<GPUSpatialJoinGeometry, {kind: 'polygons'}>,
  query: GraphDataView<'float32x2'>,
  radius: GraphDataView<'float32'>,
  allocate: (name: string) => GraphDataView<'uint32'>,
  consumerIndex?: number
) {
  const outputs: Record<string, GraphDataView<'uint32'>> = {};
  const output = (name: string) => {
    const view = allocate(name);
    outputs[name] = view;
    return view;
  };
  const pairs = (prefix: string) => ({
    leftRows: output(`${prefix}-left`),
    rightRows: output(`${prefix}-right`),
    status: {count: output(`${prefix}-count`), overflow: output(`${prefix}-overflow`)}
  });
  const points = {kind: 'points' as const, positions: query};
  const factories = [
    () =>
      new GPUPointInPolygonJoin({
        id: 'snapshot-pip',
        points,
        polygons,
        prepared,
        candidateCapacity: 4,
        output: {
          pointFeatureIds: output('pip-id'),
          pointFeatureRows: output('pip-row'),
          polygonAssignmentCounts: output('pip-count'),
          status: {overflow: output('pip-overflow')}
        }
      }),
    () =>
      new GPUSpatialPredicateJoin({
        id: 'snapshot-predicate',
        left: points,
        right: polygons,
        prepared,
        predicate: 'within',
        engine: 'fast',
        candidateCapacity: 4,
        output: {pairs: pairs('predicate')}
      }),
    () =>
      new GPUSpatialJoinCandidates({
        id: 'snapshot-candidates',
        left: points,
        right: polygons,
        prepared,
        output: pairs('candidates')
      }),
    ...[false, true].map(neighbors => () => {
      const prefix = neighbors ? 'neighbors' : 'nearest';
      return new GPUNearestFeatureJoin({
        id: `snapshot-${prefix}`,
        queries: points,
        features: polygons,
        prepared,
        radius,
        output: {
          status: {overflow: output(`${prefix}-overflow`)},
          ...(neighbors
            ? {
                neighborIds: output(`${prefix}-id`),
                neighborRows: output(`${prefix}-row`),
                neighborCounts: output(`${prefix}-count`)
              }
            : {
                nearestFeatureIds: output(`${prefix}-id`),
                nearestFeatureRows: output(`${prefix}-row`)
              })
        }
      });
    })
  ];
  const contributors = factories
    .filter((_, index) => consumerIndex === undefined || index === consumerIndex)
    .map(create => create());
  return {outputs, nodes: contributors.flatMap(contributor => contributor.getCommandNodes(graph))};
}

export function getPreparedConsumerExpectations(
  hit: boolean,
  sourceId: number
): Record<string, number[]> {
  return {
    'pip-id': [hit ? sourceId : 0xffffffff],
    'pip-row': [hit ? 0 : 0xffffffff],
    'pip-count': [Number(hit)],
    'predicate-count': [Number(hit)],
    'candidates-count': [Number(hit)],
    'nearest-id': [hit ? sourceId : 0xffffffff],
    'nearest-row': [hit ? 0 : 0xffffffff],
    'neighbors-id': [hit ? sourceId : 0xffffffff],
    'neighbors-row': [hit ? 0 : 0xffffffff],
    'neighbors-count': [Number(hit)],
    ...Object.fromEntries(
      ['pip', 'predicate', 'candidates', 'nearest', 'neighbors'].map(name => [
        `${name}-overflow`,
        [0]
      ])
    ),
    ...(hit ? {'predicate-right': [0], 'candidates-right': [0]} : {})
  };
}

/** Graph producer writes through the fresh graph's canonical geometry handle. */
export function createPreparedProducerNode(
  graph: GPUCommandGraph,
  source: GraphDataView<'float32x2'>,
  destination: GraphDataView<'float32x2'>
) {
  return createWGSLKernelNode(graph, {
    id: 'fresh-producer',
    operation: 'PreparedRetryTest',
    invocationCount: 8,
    bindings: [
      {name: 'source', view: source, type: 'u32', access: 'read'},
      {name: 'destination', view: destination, type: 'u32', access: 'read_write'}
    ],
    body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
  });
}
