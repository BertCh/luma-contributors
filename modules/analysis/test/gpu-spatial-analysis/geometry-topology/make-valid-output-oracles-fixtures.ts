// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPUMakeValid,
  type GPUMakeValidProps
} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-make-valid';
import type {GPUPolygonizeOutput} from '../../../src/gpu-spatial-analysis/geometry-topology/topology-types';

export const COMPONENTS = {uint32: 1, float32: 1, float32x2: 2, float32x4: 4} as const;
export type Format = keyof typeof COMPONENTS;
export type ViewFactory = <Value extends Format>(
  format: Value,
  length: number,
  values?: number[]
) => GraphDataView<Value>;
export const POISON = 777;
export const NONE = 0xffffffff;
export const UNSUPPORTED_STATUS_FIELDS = [
  'invalidCount',
  'iterationCount',
  'converged',
  'iterationLimitReached',
  'approximation'
] as const;
export const STATUS_DESTINATIONS = ['noded', 'directed', 'top'] as const;
export type StatusDestination = (typeof STATUS_DESTINATIONS)[number];
export type Scene = 'squares' | 'bow-tie' | 'dangle' | 'bridge' | 'uncertain' | 'hole';

// Shapely reference and canonical coordinates: make-valid-output-oracles.oracle.py.
// Ordinary-scene coordinates, midpoint parameters and areas are exact float32 values.
// The uncertainty scene deliberately exceeds the exact predicates' 200-binary-order span.
export const SQUARE_SEGMENTS = [
  [1, 2, 3, 2],
  [3, 2, 3, 4],
  [3, 4, 1, 4],
  [1, 4, 1, 2],
  [6, 1, 9, 1],
  [9, 1, 9, 3],
  [9, 3, 6, 3],
  [6, 3, 6, 1]
];
export const BOW_TIE_SEGMENTS = [
  [0, 0, 1, 1],
  [1, 1, 2, 2],
  [2, 2, 0, 2],
  [0, 2, 1, 1],
  [1, 1, 2, 0],
  [2, 0, 0, 0]
];
export const SQUARE_CANONICAL_RINGS = [
  [
    [1, 2],
    [1, 4],
    [3, 2],
    [3, 4]
  ],
  [
    [6, 1],
    [6, 3],
    [9, 1],
    [9, 3]
  ]
];

export function createOracleScene(
  view: ViewFactory,
  scene: Scene = 'squares',
  uncertainty?: StatusDestination,
  snapped = false,
  segmentCapacity = 12
) {
  const ringCapacity = 8;
  const scalar = () => view('uint32', 1);
  const status = () => ({
    count: scalar(),
    requiredCount: scalar(),
    overflow: scalar(),
    candidateOverflow: scalar()
  });
  const output: GPUPolygonizeOutput = {
    nodedSegments: {
      endpoints: view('float32x4', segmentCapacity),
      sourceFeatureIds: view('uint32', segmentCapacity),
      sourceRingIds: view('uint32', segmentCapacity),
      sourceEdgeIds: view('uint32', segmentCapacity),
      sourceStartParameters: view('float32', segmentCapacity),
      sourceEndParameters: view('float32', segmentCapacity),
      status: status()
    },
    directedEdges: {
      endpoints: view('float32x4', 2 * segmentCapacity),
      sourceSegmentIds: view('uint32', 2 * segmentCapacity),
      sourceFeatureIds: view('uint32', 2 * segmentCapacity),
      twinIds: view('uint32', 2 * segmentCapacity),
      ringIds: view('uint32', 2 * segmentCapacity),
      flags: view('uint32', 2 * segmentCapacity),
      status: status()
    },
    polygons: {
      kind: 'polygons',
      positions: view('float32x2', 64),
      featureOffsets: view('uint32', ringCapacity + 1),
      polygonOffsets: view('uint32', ringCapacity + 1),
      ringOffsets: view('uint32', ringCapacity + 1),
      sourceIds: view('uint32', ringCapacity)
    },
    diagnostics: {
      edgeClasses: view('uint32', segmentCapacity),
      closedRingCount: scalar(),
      openChainCount: scalar(),
      cutEdgeCount: scalar(),
      dangleCount: scalar(),
      unusedEdgeCount: scalar()
    },
    status: {
      count: scalar(),
      requiredRingCount: scalar(),
      overflow: scalar(),
      candidateOverflow: scalar()
    }
  };
  const statuses = {
    noded: output.nodedSegments.status,
    directed: output.directedEdges.status,
    top: output.status
  };
  if (uncertainty) statuses[uncertainty].uncertainCount = scalar();
  const positions =
    scene === 'squares'
      ? [1, 2, 3, 2, 3, 4, 1, 4, 6, 1, 9, 1, 9, 3, 6, 3]
      : scene === 'bow-tie'
        ? [0, 0, 2, 2, 0, 2, 2, 0]
        : scene === 'dangle'
          ? [1, 2, 3, 2]
          : scene === 'uncertain'
            ? [1e-30, 1e-30, 2e-30, 2e-30, 1.5e-30, 1.5e-30, 1e38, 1e38]
            : scene === 'hole'
              ? [0, 0, 4, 0, 4, 4, 0, 4, 1, 1, 1, 3, 3, 3, 3, 1]
              : [0, 0, 2, 0, 2, 2, 0, 2, 0, 0, -2, 0, -4, 0, -4, 2, -2, 2, -2, 0];
  const ringOffsets =
    scene === 'squares' || scene === 'hole'
      ? [0, 4, 8]
      : scene === 'uncertain'
        ? [0, 2, 4]
        : [0, positions.length / 2];
  const twoFeatures = scene === 'squares' || scene === 'uncertain';
  const offsets = twoFeatures ? [0, 1, 2] : [0, 1];
  const lattice = snapped ? view('float32', 4) : undefined;
  const callerProps: GPUMakeValidProps = {
    id: 'make-valid-oracle',
    polygons: {
      kind: 'polygons',
      positions: view('float32x2', positions.length / 2, positions),
      ringOffsets: view('uint32', ringOffsets.length, ringOffsets),
      polygonOffsets: view('uint32', offsets.length, scene === 'hole' ? [0, 2] : offsets),
      featureOffsets: view('uint32', offsets.length, offsets),
      sourceIds: view('uint32', offsets.length - 1, twoFeatures ? [70, 80] : [70])
    },
    intersectionCapacity: 32,
    precision: {vertexTolerance: 1 / 1024, predicates: 'exact-or-uncertain', storage: 'float32'},
    ...(lattice ? {snapRounding: {gridSize: 0.3, lattice}} : {}),
    output
  };
  const contributor = new GPUMakeValid(callerProps);
  return {contributor, callerProps, output, statuses, lattice};
}

/** Compare sets of corners, independent of ring starting vertex and winding. */
export function canonicalizeRing(positions: number[], start: number, end: number) {
  const corners = new Map<string, number[]>();
  for (let vertex = start; vertex < end; vertex++) {
    const point = positions.slice(2 * vertex, 2 * vertex + 2);
    corners.set(point.join(','), point);
  }
  return [...corners.values()].sort(
    (first, second) => first[0] - second[0] || first[1] - second[1]
  );
}
