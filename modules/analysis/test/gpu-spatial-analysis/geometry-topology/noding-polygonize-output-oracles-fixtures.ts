// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPULineNoding} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-line-noding';
import {GPUSnapRoundedNoding} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-snap-rounded-noding';
import {GPUPolygonize} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-polygonize';
import type {GPUNodedSegmentPort} from '../../../src/gpu-spatial-analysis/geometry-topology/topology-types';
import {createTopologyOutput} from '../geometry-refinement/topology-output';

export type OracleFormat = 'uint32' | 'uint32x4' | 'float32' | 'float32x2' | 'float32x4';
export type OracleAllocate = <Format extends OracleFormat>(
  name: string,
  format: Format,
  length: number,
  values?: readonly number[]
) => GraphDataView<Format>;

export const UNSUPPORTED_TOPOLOGY_STATUS = [
  'invalidCount',
  'iterationCount',
  'converged',
  'iterationLimitReached',
  'approximation'
] as const;

export function createNodingOracleCase(
  graph: GPUCommandGraph,
  allocate: OracleAllocate,
  uncertainty = false,
  unsupportedStatus?: string,
  mode: 'line' | 'snap' = 'line'
) {
  // Hand derivation: horizontal (2,2)->(10,2), vertical (4,0)->(4,8) intersect
  // at (4,2), a quarter of the way along each. Edge rows are 0 and 2; labels
  // 70/80 deliberately differ from both ring rows and edge rows. All numbers
  // are exactly representable in float32: equality also rejects NaN.
  const lines = {
    kind: 'lines' as const,
    positions: allocate('positions', 'float32x2', 4, [2, 2, 10, 2, 4, 0, 4, 8]),
    lineOffsets: allocate('line-offsets', 'uint32', 3, [0, 2, 4]),
    sourceIds: allocate('source-ids', 'uint32', 2, [70, 80])
  };
  const scalar = (name: string) => allocate(name, 'uint32', 1);
  const output: GPUNodedSegmentPort = {
    endpoints: allocate('endpoints', 'float32x4', 5),
    sourceFeatureIds: allocate('features', 'uint32', 5),
    sourceRingIds: allocate('rings', 'uint32', 5),
    sourceEdgeIds: allocate('edges', 'uint32', 5),
    sourceStartParameters: allocate('starts', 'float32', 5),
    sourceEndParameters: allocate('ends', 'float32', 5),
    status: {
      count: scalar('count'),
      requiredCount: scalar('required'),
      overflow: scalar('overflow'),
      candidateOverflow: scalar('candidate'),
      ...(uncertainty ? {uncertainCount: scalar('uncertain')} : {}),
      ...(unsupportedStatus ? {[unsupportedStatus]: scalar(unsupportedStatus)} : {})
    }
  };
  const expected: Record<string, number[]> = {
    endpoints: [2, 2, 4, 2, 4, 2, 10, 2, 4, 0, 4, 2, 4, 2, 4, 8, 0, 0, 0, 0],
    sourceFeatureIds: [70, 70, 80, 80, 0xffffffff],
    sourceRingIds: [0, 0, 1, 1, 0xffffffff],
    sourceEdgeIds: [0, 0, 2, 2, 0xffffffff],
    sourceStartParameters: [0, 0.25, 0, 0.25, 0],
    sourceEndParameters: [0.25, 1, 0.25, 1, 0],
    'status.count': [4],
    'status.requiredCount': [4],
    'status.overflow': [0],
    'status.candidateOverflow': [0],
    ...(uncertainty ? {'status.uncertainCount': [0]} : {})
  };
  const lattice = mode === 'snap' ? allocate('lattice', 'float32', 4) : undefined;
  const latticeEndpoints =
    mode === 'snap' ? allocate('lattice-endpoints', 'float32x4', 5) : undefined;
  const contributor =
    mode === 'snap'
      ? new GPUSnapRoundedNoding({
          lines,
          output: {...output, lattice: lattice!, latticeEndpoints: latticeEndpoints!},
          intersectionCapacity: 4,
          gridSize: 0.3
        })
      : new GPULineNoding({lines, output, intersectionCapacity: 4});
  // Unit-independent quarter intervals stay the same on a 0.5 grid. The origin is
  // (2,0), so lattice coordinates are 2*(world-origin), with a zero sentinel tail.
  if (mode === 'snap')
    Object.assign(expected, {
      lattice: [2, 0, 0.5, 0],
      latticeEndpoints: [0, 4, 4, 4, 4, 4, 16, 4, 4, 0, 4, 4, 4, 4, 4, 16, 0, 0, 0, 0]
    });
  return {
    output: {...output, ...(mode === 'snap' ? {lattice, latticeEndpoints} : {})},
    expected,
    nodes: contributor.getCommandNodes(graph)
  };
}

export function getOracleView(output: object, path: string): GraphDataView {
  return path
    .split('.')
    .reduce<unknown>(
      (value, key) => (value as Record<string, unknown>)[key],
      output
    ) as GraphDataView;
}

export function createPolygonizeOracleCase(
  graph: GPUCommandGraph,
  allocate: OracleAllocate,
  uncertainty: 'none' | 'top' | 'noded' | 'directed' | 'all' = 'none',
  snapped = false,
  unsupported?: {path: 'status' | 'nodedSegments.status' | 'directedEdges.status'; field: string}
) {
  let serial = 0;
  const output = createTopologyOutput((format, length) =>
    allocate(`polygon-${serial++}`, format, length)
  );
  for (const [name, status] of [
    ['top', output.status],
    ['noded', output.nodedSegments.status],
    ['directed', output.directedEdges.status]
  ] as const) {
    if (uncertainty === 'all' || uncertainty === name)
      status.uncertainCount = allocate(`polygon-${name}-uncertain`, 'uint32', 1);
  }
  if (unsupported) {
    const status =
      unsupported.path === 'status'
        ? output.status
        : unsupported.path === 'nodedSegments.status'
          ? output.nodedSegments.status
          : output.directedEdges.status;
    Object.assign(status, {[unsupported.field]: allocate('unsupported', 'uint32', 1)});
  }
  // Two CCW unit squares joined by one bridge. Shapely polygonize_full in the
  // adjacent oracle script gives two area-1 polygons, one cut and no dangles.
  // Noding does not split any edge. Each square has a bounded left-face cycle;
  // the connected graph has one exterior walk, including BOTH bridge sides.
  // Directed rows 2*s and 2*s+1 are forward and reverse source segment s.
  // The cycles' least directed rows are 0, 1, 8 -> ring IDs 0, 1, 2.
  const endpoints = [
    0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 3, 0, 4, 0, 4, 0, 4, 1, 4, 1, 3, 1, 3, 1, 3, 0,
    1, 0, 3, 0
  ];
  const expected: Record<string, number[]> = {
    'nodedSegments.endpoints': endpoints,
    'nodedSegments.sourceFeatureIds': [70, 70, 70, 70, 80, 80, 80, 80, 90],
    'nodedSegments.sourceRingIds': [0, 0, 0, 0, 1, 1, 1, 1, 2],
    'nodedSegments.sourceEdgeIds': [0, 1, 2, 3, 5, 6, 7, 8, 10],
    'nodedSegments.sourceStartParameters': [0, 0, 0, 0, 0, 0, 0, 0, 0],
    'nodedSegments.sourceEndParameters': [1, 1, 1, 1, 1, 1, 1, 1, 1],
    'directedEdges.endpoints': [
      0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0,
      1, 3, 0, 4, 0, 4, 0, 3, 0, 4, 0, 4, 1, 4, 1, 4, 0, 4, 1, 3, 1, 3, 1, 4, 1, 3, 1, 3, 0, 3, 0,
      3, 1, 1, 0, 3, 0, 3, 0, 1, 0
    ],
    'directedEdges.sourceSegmentIds': [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
    'directedEdges.sourceFeatureIds': [
      70, 70, 70, 70, 70, 70, 70, 70, 80, 80, 80, 80, 80, 80, 80, 80, 90, 90
    ],
    'directedEdges.twinIds': [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16],
    'directedEdges.ringIds': [0, 1, 0, 1, 0, 1, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1],
    // TOUCHING=1 when the end has two possible continuations after excluding
    // the twin. Only bridge attachment vertices (1,0) and (3,0) have degree 3.
    'directedEdges.flags': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1],
    'polygons.featureOffsets': [0, 1, 2, 2, 2, 2, 2, 2, 2],
    'polygons.polygonOffsets': [0, 1, 2, 2, 2, 2, 2, 2, 2],
    'polygons.ringOffsets': [0, 4, 8, 8, 8, 8, 8, 8, 8],
    'polygons.sourceIds': [0, 0],
    'diagnostics.edgeClasses': [1, 1, 1, 1, 1, 1, 1, 1, 22],
    'diagnostics.closedRingCount': [8],
    'diagnostics.openChainCount': [1],
    'diagnostics.cutEdgeCount': [1],
    'diagnostics.dangleCount': [0],
    'diagnostics.unusedEdgeCount': [1],
    'status.count': [2],
    'status.requiredRingCount': [3],
    'status.overflow': [0],
    'status.candidateOverflow': [0]
  };
  for (const [prefix, count] of [
    ['nodedSegments', 9],
    ['directedEdges', 18]
  ] as const)
    for (const name of ['count', 'requiredCount', 'overflow', 'candidateOverflow'])
      expected[`${prefix}.status.${name}`] = [
        name === 'count' || name === 'requiredCount' ? count : 0
      ];
  if (output.status.uncertainCount) expected['status.uncertainCount'] = [0];
  if (output.nodedSegments.status.uncertainCount)
    expected['nodedSegments.status.uncertainCount'] = [0];
  if (output.directedEdges.status.uncertainCount)
    expected['directedEdges.status.uncertainCount'] = [0];
  const lattice = snapped ? allocate('polygon-lattice', 'float32', 4) : undefined;
  if (snapped) expected['lattice'] = [0, 0, 0.5, 0];
  for (const path of [
    'nodedSegments.endpoints',
    'nodedSegments.sourceFeatureIds',
    'nodedSegments.sourceRingIds',
    'nodedSegments.sourceEdgeIds',
    'nodedSegments.sourceStartParameters',
    'nodedSegments.sourceEndParameters',
    'directedEdges.endpoints',
    'directedEdges.sourceSegmentIds',
    'directedEdges.sourceFeatureIds',
    'directedEdges.twinIds',
    'directedEdges.ringIds',
    'directedEdges.flags',
    'diagnostics.edgeClasses'
  ]) {
    const view = getOracleView(output, path);
    const padding = path.endsWith('Ids') ? 0xffffffff : 0;
    expected[path] = expected[path].concat(
      new Array((view.length * view.rowByteLength) / 4 - expected[path].length).fill(padding)
    );
  }
  const contributor = new GPUPolygonize({
    lines: {
      kind: 'lines',
      positions: allocate(
        'polygon-input',
        'float32x2',
        12,
        [0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 3, 0, 4, 0, 4, 1, 3, 1, 3, 0, 1, 0, 3, 0]
      ),
      lineOffsets: allocate('polygon-input-offsets', 'uint32', 4, [0, 5, 10, 12]),
      sourceIds: allocate('polygon-input-ids', 'uint32', 3, [70, 80, 90])
    },
    intersectionCapacity: 16,
    precision: {vertexTolerance: 1 / 64, predicates: 'exact-or-uncertain', storage: 'float32'},
    output,
    ...(snapped ? {snapRounding: {gridSize: 0.3, lattice: lattice!}} : {})
  });
  return {
    output: {...output, ...(snapped ? {lattice} : {})},
    expected,
    nodes: contributor.getCommandNodes(graph)
  };
}
