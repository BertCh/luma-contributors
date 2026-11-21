// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPULineNoding} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-line-noding';
import {GPUSnapRoundedNoding} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-snap-rounded-noding';
import {GPUPolygonize} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-polygonize';
import {GPULineSplit} from '../../../src/gpu-spatial-analysis/line-split/gpu-line-split';
import {GPULineLoopRemoval} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-line-loop-removal';
import {GPUGeometryValidity} from '../../../src/gpu-spatial-analysis/geometry-validity/gpu-geometry-validity';
import {GPUBufferSurface} from '../../../src/gpu-spatial-analysis/polygon-buffer/gpu-buffer-surface';
import {GPUSharedPaths} from '../../../src/gpu-spatial-analysis/line-clip/gpu-shared-paths';
import {GPULineClipByPolygon} from '../../../src/gpu-spatial-analysis/line-clip/gpu-line-clip-by-polygon';
import {GPUTrackSimilarity} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity';
import {GPUTrackSimilaritySearch} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity-search';
import {GPUGeometryPredicates} from '../../../src/gpu-spatial-analysis/geometry-predicates/gpu-geometry-predicates';
import {GPUGeometryMeasures} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geometry-measures';
import type {TrackSimilarityOperationProps} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity-operation';
import type {OperationGraphInput} from '../../utils/mixed-execution-test-utils';

export type MultipartTestFormat =
  | 'float32'
  | 'float32x2'
  | 'float32x4'
  | 'uint32'
  | 'uint32x2'
  | 'uint32x4';
export type MultipartTestAllocate = <Format extends MultipartTestFormat>(
  name: string,
  format: Format,
  length: number,
  values?: number[]
) => GraphDataView<Format>;

export const MULTIPART_NODED_EXPECTED = {
  endpoints: [0, 0, 1, 1, 1, 1, 2, 2, 0, 2, 1, 1, 1, 1, 2, 0, 5, 0, 7, 0],
  sourceFeatureIds: [17, 17, 17, 17, 42],
  sourceRingIds: [0, 0, 1, 1, 2],
  sourceEdgeIds: [0, 0, 2, 2, 4],
  sourceStartParameters: [0, 0.5, 0, 0.5, 0],
  sourceEndParameters: [0.5, 1, 0.5, 1, 1]
};

export function createEmptyMultipartCase(
  graph: GPUCommandGraph,
  allocate: MultipartTestAllocate,
  kind: 'points' | 'lines' = 'points',
  predicatesOnly = false
) {
  const common = {
    positions: allocate('empty-input', 'float32x2', 0, []),
    featureOffsets: allocate('empty-features', 'uint32', 3, [0, 0, 0])
  };
  const geometry =
    kind === 'points'
      ? {kind, ...common}
      : {kind, ...common, lineOffsets: allocate('empty-parts', 'uint32', 1, [0])};
  const columns = [
    'isSimple',
    'isRing',
    'isClosed',
    'isCcw',
    'numPoints',
    'numCoordinates',
    'numInteriorRings',
    'numGeometries',
    'equalsExact',
    'equalsIdentical'
  ] as const;
  const predicateColumns = Object.fromEntries(
    columns.map(name => [name, allocate(`empty-${name}`, 'uint32', 2)])
  ) as Record<(typeof columns)[number], GraphDataView<'uint32'>>;
  const predicates = {
    ...predicateColumns,
    uniquePositions: allocate('empty-unique', 'float32x2', 1),
    uniqueOffsets: allocate('empty-unique-offsets', 'uint32', 3),
    status: {
      overflow: allocate('empty-overflow', 'uint32', 1),
      intersectionCount: allocate('empty-intersections', 'uint32', 1)
    }
  };
  const measureColumns = (prefix: string, count: number) => ({
    lengths: allocate(`${prefix}-length`, 'float32', count),
    centroids: allocate(`${prefix}-centroid`, 'float32x2', count),
    bounds: allocate(`${prefix}-bounds`, 'float32x4', count),
    vertexCounts: allocate(`${prefix}-vertices`, 'uint32', count),
    ...(prefix === 'empty-measures'
      ? {extremeVertices: allocate(`${prefix}-extremes`, 'uint32x4', count)}
      : {}),
    featureCounts: allocate(`${prefix}-feature-counts`, 'uint32', count)
  });
  const measures = {
    ...measureColumns('empty-measures', 2),
    status: {invalidCount: allocate('empty-invalid', 'uint32', 1)}
  };
  const groups = measureColumns('empty-groups', 3);
  const expected: Record<string, number[]> = {
    'predicates.uniqueOffsets': [0, 0, 0],
    'predicates.status.overflow': [0],
    'predicates.status.intersectionCount': [0],
    'measures.status.invalidCount': [0]
  };
  for (const name of columns)
    expected[`predicates.${name}`] =
      name === 'isSimple' || name.startsWith('equals') ? [1, 1] : [0, 0];
  for (const [prefix, count] of [
    ['measures', 2],
    ['groups', 3]
  ] as const) {
    expected[`${prefix}.lengths`] = new Array(count).fill(0);
    expected[`${prefix}.centroids`] = new Array(count * 2).fill(NaN);
    expected[`${prefix}.bounds`] = new Array(count * 4).fill(NaN);
    expected[`${prefix}.vertexCounts`] = new Array(count).fill(0);
    if (prefix === 'measures')
      expected[`${prefix}.extremeVertices`] = new Array(count * 4).fill(0xffffffff);
    expected[`${prefix}.featureCounts`] = prefix === 'measures' ? [1, 1] : [1, 1, 0];
  }
  const nodes = [
    ...new GPUGeometryPredicates({
      geometry,
      other: geometry,
      intersectionCapacity: 1,
      parameters: allocate('empty-tolerance', 'float32', 4, [0, 0, 0, 0]),
      output: predicates
    }).getCommandNodes(graph),
    ...(predicatesOnly
      ? []
      : new GPUGeometryMeasures({
          ...(geometry.kind === 'lines'
            ? {...geometry, geometryType: 'lines' as const}
            : {...geometry, geometryType: 'points' as const}),
          output: measures,
          groupIds: allocate('empty-group-ids', 'uint32', 2, [1, 0]),
          groupCount: 3,
          groupOutput: groups
        }).getCommandNodes(graph))
  ];
  if (predicatesOnly)
    for (const path of Object.keys(expected))
      if (!path.startsWith('predicates.')) delete expected[path];
  return {output: {predicates, measures, groups}, expected, nodes};
}

export const MULTIPART_TRACK_INPUT = {
  linesA: {
    positions: Float32Array.of(0, 0, 1, 0, 9, 0, 10, 0, 0, 2, 1, 2),
    lineOffsets: Uint32Array.of(0, 2, 4, 6),
    featureOffsets: Uint32Array.of(0, 0, 2, 3)
  },
  linesB: {
    positions: Float32Array.of(0, 0, 1, 0, 9, 0, 10, 0, 5, 0, 5, 1),
    lineOffsets: Uint32Array.of(0, 2, 4, 6),
    featureOffsets: Uint32Array.of(0, 3)
  },
  pairA: Uint32Array.of(1, 2, 0, 99, 1),
  pairB: Uint32Array.of(0, 0, 0, 0, 0),
  activePairCount: Uint32Array.of(4)
};
export const MULTIPART_TRACK_EXPECTED = {
  hausdorff: [4.123105625617661, 9.219544457292887, NaN, NaN, NaN],
  symmetricSegmentPathDistance: [0.6769254688014718, 3.505083107428787, NaN, NaN, NaN],
  maximumDistance: [10, 10.19803902718557, NaN, NaN, NaN],
  pairStatus: [0, 0, 1, 4, 4]
};
export function createMultipartTrackOperationCase() {
  const inputs: Record<string, OperationGraphInput> = {};
  for (const role of ['A', 'B'] as const) {
    const input = MULTIPART_TRACK_INPUT[role === 'A' ? 'linesA' : 'linesB'];
    inputs[`positions${role}`] = {format: 'float32x2', values: input.positions};
    inputs[`offsets${role}`] = {format: 'uint32', values: input.lineOffsets};
    inputs[`features${role}`] = {format: 'uint32', values: input.featureOffsets};
  }
  for (const name of ['pairA', 'pairB', 'activePairCount'] as const)
    inputs[name] = {format: 'uint32', values: MULTIPART_TRACK_INPUT[name]};
  const props: TrackSimilarityOperationProps = {
    linesA: {positions: 'positionsA', lineOffsets: 'offsetsA', featureOffsets: 'featuresA'},
    linesB: {positions: 'positionsB', lineOffsets: 'offsetsB', featureOffsets: 'featuresB'},
    pairA: 'pairA',
    pairB: 'pairB',
    activePairCount: 'activePairCount',
    pairCount: 5,
    densify: 0.5,
    output: {
      hausdorff: 'hausdorff',
      symmetricSegmentPathDistance: 'symmetricSegmentPathDistance',
      maximumDistance: 'maximumDistance',
      pairStatus: 'pairStatus'
    }
  };
  return {inputs, props};
}

export function createMultipartTrackSearchCase(
  graph: GPUCommandGraph,
  allocate: MultipartTestAllocate,
  metric: 'hausdorff' | 'symmetric-segment-path' = 'hausdorff'
) {
  const lines = (role: 'linesA' | 'linesB') => {
    const input = MULTIPART_TRACK_INPUT[role];
    return {
      kind: 'lines' as const,
      positions: allocate(`${role}-search-positions`, 'float32x2', 6, Array.from(input.positions)),
      lineOffsets: allocate(`${role}-search-parts`, 'uint32', 4, Array.from(input.lineOffsets)),
      featureOffsets: allocate(
        `${role}-search-features`,
        'uint32',
        input.featureOffsets.length,
        Array.from(input.featureOffsets)
      )
    };
  };
  const output = {
    distance: allocate('search-distance', 'float32', 4),
    lowerBound: allocate('search-bounds', 'float32', 4),
    matched: allocate('search-matched', 'uint32', 4),
    status: allocate('search-status', 'uint32', 4),
    survivorSummary: allocate('search-survivors', 'uint32', 3),
    topKIndices: allocate('search-top-indices', 'uint32', 3),
    topKSummary: allocate('search-top-summary', 'uint32x2', 3)
  };
  const expected: Record<string, number[]> = {
    distance: [metric === 'hausdorff' ? 4.123105625617661 : 0.6769254688014718, NaN, NaN, NaN],
    lowerBound:
      metric === 'hausdorff'
        ? [1, 9.219544457292887, NaN, NaN]
        : [0.08333323333333333, 3.005080222549004, NaN, NaN],
    matched: [1, 0, 0, 0],
    status: [0, 1, 2, 2],
    survivorSummary: [1, 1, 0],
    topKIndices: [0xffffffff, 0, 0xffffffff],
    topKSummary: [0, 0, 1, 1, 0, 1]
  };
  const nodes = new GPUTrackSimilaritySearch({
    linesA: lines('linesA'),
    linesB: lines('linesB'),
    pairA: allocate('search-pairs-a', 'uint32', 4, [1, 2, 0, 99]),
    pairB: allocate('search-pairs-b', 'uint32', 4, [0, 0, 0, 0]),
    metric,
    threshold: allocate('search-threshold', 'float32', 1, [metric === 'hausdorff' ? 5 : 1]),
    survivorCapacity: 1,
    topK: 1,
    output
  }).getCommandNodes(graph);
  return {output, expected, nodes};
}
export function createMultipartTrackCase(graph: GPUCommandGraph, allocate: MultipartTestAllocate) {
  const createLines = (role: 'linesA' | 'linesB') => {
    const input = MULTIPART_TRACK_INPUT[role];
    return {
      kind: 'lines' as const,
      positions: allocate(`${role}-positions`, 'float32x2', 6, Array.from(input.positions)),
      lineOffsets: allocate(`${role}-parts`, 'uint32', 4, Array.from(input.lineOffsets)),
      featureOffsets: allocate(
        `${role}-features`,
        'uint32',
        input.featureOffsets.length,
        Array.from(input.featureOffsets)
      )
    };
  };
  const linesA = createLines('linesA');
  const linesB = createLines('linesB');
  const output = {
    hausdorff: allocate('track-hausdorff', 'float32', 5),
    symmetricSegmentPathDistance: allocate('track-segment-path', 'float32', 5),
    maximumDistance: allocate('track-maximum', 'float32', 5),
    pairStatus: allocate('track-status', 'uint32', 5)
  };
  return {
    output,
    expected: MULTIPART_TRACK_EXPECTED,
    nodes: new GPUTrackSimilarity({
      linesA,
      linesB,
      pairA: allocate('track-pair-a', 'uint32', 5, Array.from(MULTIPART_TRACK_INPUT.pairA)),
      pairB: allocate('track-pair-b', 'uint32', 5, Array.from(MULTIPART_TRACK_INPUT.pairB)),
      activePairCount: allocate('track-active', 'uint32', 1, [4]),
      densify: 0.5,
      output
    }).getCommandNodes(graph)
  };
}

export function createMultipartClipCase(
  graph: GPUCommandGraph,
  allocate: MultipartTestAllocate,
  mode: 'inside' | 'outside' = 'inside'
) {
  const lines = {
    kind: 'lines' as const,
    positions: allocate('clip-input', 'float32x2', 4, [0, 0, 2, 0, 5, 0, 7, 0]),
    lineOffsets: allocate('clip-parts', 'uint32', 3, [0, 2, 4]),
    featureOffsets: allocate('clip-features', 'uint32', 3, [0, 0, 2]),
    sourceIds: allocate('clip-identities', 'uint32', 2, [90, 17])
  };
  const polygons = {
    kind: 'polygons' as const,
    positions: allocate('clip-mask', 'float32x2', 4, [1, -1, 6, -1, 6, 1, 1, 1]),
    featureOffsets: allocate('clip-mask-features', 'uint32', 2, [0, 1]),
    polygonOffsets: allocate('clip-mask-polygons', 'uint32', 2, [0, 1]),
    ringOffsets: allocate('clip-mask-rings', 'uint32', 2, [0, 4])
  };
  const scalar = (name: string) => allocate(name, 'uint32', 1);
  const output = {
    pieces: {
      geometry: {
        kind: 'lines' as const,
        positions: allocate('clip-output', 'float32x2', 4),
        lineOffsets: allocate('clip-output-parts', 'uint32', 3),
        featureOffsets: allocate('clip-output-features', 'uint32', 3)
      },
      sourceIds: allocate('clip-output-part-rows', 'uint32', 2),
      sourcePartRows: allocate('clip-output-explicit-part-rows', 'uint32', 2),
      sourceFeatureRows: allocate('clip-output-feature-rows', 'uint32', 2),
      sourceFeatureIds: allocate('clip-output-identities', 'uint32', 2),
      vertexSourceEdges: allocate('clip-output-edges', 'uint32', 4),
      status: {
        count: scalar('clip-count'),
        vertexCount: scalar('clip-vertices'),
        requiredCount: scalar('clip-required'),
        requiredVertexCount: scalar('clip-required-vertices'),
        overflow: scalar('clip-overflow'),
        candidateOverflow: scalar('clip-candidate')
      }
    },
    status: {uncertainCount: scalar('clip-uncertain')}
  };
  const expected: Record<string, number[]> = {
    'pieces.geometry.positions':
      mode === 'inside' ? [1, 0, 2, 0, 5, 0, 6, 0] : [0, 0, 1, 0, 6, 0, 7, 0],
    'pieces.geometry.lineOffsets': [0, 2, 4],
    'pieces.geometry.featureOffsets': [0, 0, 2],
    'pieces.sourceIds': [17, 17],
    'pieces.sourcePartRows': [0, 1],
    'pieces.sourceFeatureRows': [1, 1],
    'pieces.sourceFeatureIds': [17, 17],
    'pieces.vertexSourceEdges': [0, 0, 2, 2],
    'pieces.status.count': [2],
    'pieces.status.vertexCount': [4],
    'pieces.status.requiredCount': [2],
    'pieces.status.requiredVertexCount': [4],
    'pieces.status.overflow': [0],
    'pieces.status.candidateOverflow': [0],
    'status.uncertainCount': [0]
  };
  return {
    output,
    expected,
    nodes: new GPULineClipByPolygon({
      lines,
      polygons,
      mode,
      intersectionCapacity: 8,
      candidateCapacity: 32,
      output
    }).getCommandNodes(graph)
  };
}

export function createMultipartSharedPathsCase(
  graph: GPUCommandGraph,
  allocate: MultipartTestAllocate,
  grouped = true
) {
  const left = {
    kind: 'lines' as const,
    positions: allocate('shared-left', 'float32x2', 4, [0, 0, 2, 0, 5, 0, 7, 0]),
    lineOffsets: allocate('shared-left-parts', 'uint32', 3, [0, 2, 4]),
    ...(grouped ? {featureOffsets: allocate('shared-left-features', 'uint32', 3, [0, 0, 2])} : {}),
    sourceIds: allocate('shared-left-identities', 'uint32', 2, grouped ? [90, 17] : [17, 88])
  };
  const right = {
    kind: 'lines' as const,
    positions: allocate('shared-right', 'float32x2', 4, [1, 0, 2, 0, 7, 0, 6, 0]),
    lineOffsets: allocate('shared-right-parts', 'uint32', 3, [0, 2, 4]),
    ...(grouped
      ? {featureOffsets: allocate('shared-right-features', 'uint32', 4, [0, 1, 1, 2])}
      : {}),
    sourceIds: allocate(
      'shared-right-identities',
      'uint32',
      grouped ? 3 : 2,
      grouped ? [42, 90, 73] : [42, 73]
    )
  };
  const scalar = (name: string) => allocate(name, 'uint32', 1);
  const output = {
    runs: {
      leftSourceIds: allocate('shared-left-sources', 'uint32', 2),
      rightSourceIds: allocate('shared-right-sources', 'uint32', 2),
      leftPartRows: allocate('shared-left-part-rows', 'uint32', 2),
      rightPartRows: allocate('shared-right-part-rows', 'uint32', 2),
      forward: allocate('shared-forward', 'uint32', 2),
      offsets: allocate('shared-offsets', 'uint32', 3),
      positions: allocate('shared-positions', 'float32x2', 4)
    },
    status: {
      count: scalar('shared-count'),
      requiredCount: scalar('shared-required'),
      overflow: scalar('shared-overflow'),
      candidateOverflow: scalar('shared-candidate'),
      vertexCount: scalar('shared-vertices'),
      requiredVertexCount: scalar('shared-required-vertices'),
      uncertainCount: scalar('shared-uncertain')
    }
  };
  const expected: Record<string, number[]> = {
    'runs.positions': [1, 0, 2, 0, 6, 0, 7, 0],
    'runs.offsets': [0, 2, 4],
    'runs.leftSourceIds': grouped ? [17, 17] : [17, 88],
    'runs.rightSourceIds': [42, 73],
    'runs.leftPartRows': [0, 1],
    'runs.rightPartRows': [0, 1],
    'runs.forward': [1, 0],
    'status.count': [2],
    'status.requiredCount': [2],
    'status.vertexCount': [4],
    'status.requiredVertexCount': [4],
    'status.overflow': [0],
    'status.candidateOverflow': [0],
    'status.uncertainCount': [0]
  };
  return {
    output,
    expected,
    nodes: new GPUSharedPaths({left, right, intersectionCapacity: 8, output}).getCommandNodes(graph)
  };
}

export function createMultipartBufferCase(graph: GPUCommandGraph, allocate: MultipartTestAllocate) {
  const scalar = (name: string) => allocate(name, 'uint32', 1);
  const output = {
    geometry: {
      kind: 'polygons' as const,
      positions: allocate('buffer-output', 'float32x2', 16),
      featureOffsets: allocate('buffer-features', 'uint32', 5),
      polygonOffsets: allocate('buffer-polygons', 'uint32', 5),
      ringOffsets: allocate('buffer-rings', 'uint32', 5)
    },
    status: {
      count: scalar('buffer-count'),
      ringCount: scalar('buffer-ring-count'),
      requiredRingCount: scalar('buffer-required'),
      overflow: scalar('buffer-overflow'),
      candidateOverflow: scalar('buffer-candidate'),
      uncertainCount: scalar('buffer-uncertain'),
      nodedSegmentRequiredCount: scalar('buffer-noded-required')
    },
    boundary: {
      endpoints: allocate('buffer-boundary', 'float32x4', 8),
      operandIds: allocate('buffer-operands', 'uint32', 8),
      sourceFeatureIds: allocate('buffer-sources', 'uint32', 8),
      sourceFeatureRows: allocate('buffer-boundary-source-rows', 'uint32', 8),
      status: {
        count: scalar('buffer-boundary-count'),
        requiredCount: scalar('buffer-boundary-required'),
        overflow: scalar('buffer-boundary-overflow'),
        candidateOverflow: scalar('buffer-boundary-candidate')
      }
    },
    featureSources: {
      sourceRows: allocate('buffer-source-rows', 'uint32', 4),
      operandIds: allocate('buffer-source-operands', 'uint32', 4),
      mixed: allocate('buffer-mixed', 'uint32', 4)
    }
  };
  const geometry = {
    kind: 'lines' as const,
    positions: allocate('buffer-input', 'float32x2', 4, [0, 0, 2, 0, 5, 0, 7, 0]),
    lineOffsets: allocate('buffer-lines', 'uint32', 3, [0, 2, 4]),
    featureOffsets: allocate('buffer-input-features', 'uint32', 3, [0, 0, 2]),
    sourceIds: allocate('buffer-identities', 'uint32', 2, [90, 17])
  };
  const contributor = new GPUBufferSurface({
    geometry,
    parameters: allocate('buffer-parameters', 'float32', 4, [0.5, 5, 0, 0]),
    joinStyle: 'mitre',
    capStyle: 'flat',
    vertexTolerance: 0.125,
    capacity: {intersections: 16, nodedSegments: 16},
    output
  });
  return {geometry, output, contributor};
}

export function createMultipartValidityCase(
  graph: GPUCommandGraph,
  allocate: MultipartTestAllocate,
  kind: 'points' | 'lines'
) {
  const geometry =
    kind === 'lines'
      ? {
          kind: 'lines' as const,
          positions: allocate('valid-input', 'float32x2', 8, [
            0,
            0,
            1,
            0,
            5,
            0,
            6,
            0,
            2,
            2,
            2,
            2,
            0,
            0,
            Number.NaN,
            1
          ]),
          lineOffsets: allocate('valid-lines', 'uint32', 5, [0, 2, 4, 6, 8]),
          featureOffsets: allocate('valid-features', 'uint32', 6, [0, 0, 2, 3, 4, 4])
        }
      : {
          kind: 'points' as const,
          positions: allocate('valid-input', 'float32x2', 4, [0, 0, Number.NaN, 1, 1, 2, 1, 2]),
          featureOffsets: allocate('valid-features', 'uint32', 4, [0, 0, 2, 4])
        };
  const count = kind === 'lines' ? 5 : 3;
  const output = {
    issueCode: allocate('valid-issues', 'uint32', count),
    validity: allocate('valid-usable', 'uint32', count),
    status: {
      overflow: allocate('valid-overflow', 'uint32', 1),
      intersectionCount: allocate('valid-intersections', 'uint32', 1)
    }
  };
  const props = geometry.kind === 'lines' ? {lines: geometry, output} : {points: geometry, output};
  const expected: Record<string, number[]> = {
    issueCode: kind === 'lines' ? [0, 0, 512, 1, 0] : [0, 1, 0],
    validity: kind === 'lines' ? [1, 1, 0, 0, 1] : [1, 0, 1],
    'status.overflow': [0],
    'status.intersectionCount': [0]
  };
  return {output, expected, nodes: new GPUGeometryValidity(props).getCommandNodes(graph)};
}

export function createMultipartLoopCase(graph: GPUCommandGraph, allocate: MultipartTestAllocate) {
  const lines = {
    kind: 'lines' as const,
    positions: allocate(
      'loop-input',
      'float32x2',
      8,
      [0, 0, 2, 2, 0, 2, 2, 0, 5, 0, 7, 0, 10, 0, 12, 0]
    ),
    lineOffsets: allocate('loop-lines', 'uint32', 4, [0, 4, 6, 8]),
    featureOffsets: allocate('loop-features', 'uint32', 4, [0, 0, 2, 3])
  };
  const output = {
    positions: allocate('loop-positions', 'float32x2', 7),
    lineOffsets: allocate('loop-output-lines', 'uint32', 4),
    featureOffsets: allocate('loop-output-features', 'uint32', 4),
    sourceEdges: allocate('loop-edges', 'uint32', 7),
    status: allocate('loop-status', 'uint32x4', 1)
  };
  const expected: Record<string, number[]> = {
    positions: [0, 0, 1, 1, 2, 0, 5, 0, 7, 0, 10, 0, 12, 0],
    lineOffsets: [0, 3, 5, 7],
    featureOffsets: [0, 0, 2, 3],
    sourceEdges: [0, 2, 2, 4, 4, 6, 6],
    status: [7, 7, 0, 0]
  };
  return {
    output,
    expected,
    nodes: new GPULineLoopRemoval({lines, intersectionCapacity: 8, output}).getCommandNodes(graph)
  };
}

export function createMultipartSplitCase(
  graph: GPUCommandGraph,
  allocate: MultipartTestAllocate,
  capacity = 5
) {
  const lines = {
    kind: 'lines' as const,
    positions: allocate('split-input', 'float32x2', 6, [0, 0, 2, 2, 0, 2, 2, 0, 5, 0, 7, 0]),
    lineOffsets: allocate('split-lines', 'uint32', 4, [0, 2, 4, 6]),
    featureOffsets: allocate('split-features', 'uint32', 4, [0, 0, 2, 3]),
    sourceIds: allocate('split-identities', 'uint32', 3, [90, 17, 42])
  };
  const output = {
    pieces: {
      geometry: {
        kind: 'lines' as const,
        positions: allocate('split-positions', 'float32x2', capacity * 2),
        lineOffsets: allocate('split-offsets', 'uint32', capacity + 1),
        featureOffsets: allocate('split-output-features', 'uint32', 4)
      },
      sourceIds: allocate('split-part-rows', 'uint32', capacity),
      sourcePartRows: allocate('split-explicit-part-rows', 'uint32', capacity),
      sourceFeatureRows: allocate('split-feature-rows', 'uint32', capacity),
      sourceFeatureIds: allocate('split-feature-identities', 'uint32', capacity),
      vertexSourceEdges: allocate('split-edges', 'uint32', capacity * 2),
      status: {
        count: allocate('split-count', 'uint32', 1),
        requiredCount: allocate('split-required', 'uint32', 1),
        overflow: allocate('split-overflow', 'uint32', 1),
        candidateOverflow: allocate('split-candidate', 'uint32', 1),
        vertexCount: allocate('split-vertices', 'uint32', 1),
        requiredVertexCount: allocate('split-required-vertices', 'uint32', 1)
      }
    },
    status: {uncertainCount: allocate('split-uncertain', 'uint32', 1)}
  };
  const expected: Record<string, number[]> = {
    'pieces.geometry.positions': MULTIPART_NODED_EXPECTED.endpoints,
    'pieces.geometry.lineOffsets': [0, 2, 4, 6, 8, 10],
    'pieces.geometry.featureOffsets': [0, 0, 4, 5],
    'pieces.sourceIds': [17, 17, 17, 17, 42],
    'pieces.sourcePartRows': [0, 0, 1, 1, 2],
    'pieces.sourceFeatureRows': [1, 1, 1, 1, 2],
    'pieces.sourceFeatureIds': [17, 17, 17, 17, 42],
    'pieces.status.count': [5],
    'pieces.status.requiredCount': [5],
    'pieces.status.overflow': [0],
    'pieces.status.candidateOverflow': [0],
    'pieces.status.vertexCount': [10],
    'pieces.status.requiredVertexCount': [10],
    'status.uncertainCount': [0]
  };
  if (capacity === 3)
    Object.assign(expected, {
      'pieces.geometry.positions': MULTIPART_NODED_EXPECTED.endpoints.slice(0, 12),
      'pieces.geometry.lineOffsets': [0, 2, 4, 6],
      'pieces.geometry.featureOffsets': [0, 0, 3, 3],
      'pieces.sourceIds': [17, 17, 17],
      'pieces.sourcePartRows': [0, 0, 1],
      'pieces.sourceFeatureRows': [1, 1, 1],
      'pieces.sourceFeatureIds': [17, 17, 17],
      'pieces.status.count': [3],
      'pieces.status.vertexCount': [6],
      'pieces.status.overflow': [1]
    });
  return {
    output,
    expected,
    nodes: new GPULineSplit({lines, intersectionCapacity: 8, output}).getCommandNodes(graph)
  };
}

export function createMultipartPolygonizeCase(
  graph: GPUCommandGraph,
  allocate: MultipartTestAllocate,
  grouped = false,
  snapped = false
) {
  const endpoints = [0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 5, 0, 7, 0, 7, 0, 5, 2, 5, 2, 5, 0];
  const lines = {
    kind: 'lines' as const,
    positions: allocate('polygon-input', 'float32x2', 12, endpoints),
    lineOffsets: allocate('polygon-lines', 'uint32', 7, [0, 2, 4, 6, 8, 10, 12]),
    featureOffsets: allocate('polygon-features', 'uint32', 4, [0, 0, 3, 6]),
    sourceIds: allocate('polygon-identities', 'uint32', 3, [90, 17, 42])
  };
  const scalar = (name: string) => allocate(name, 'uint32', 1);
  const status = (name: string) => ({
    count: scalar(`${name}-count`),
    requiredCount: scalar(`${name}-required`),
    overflow: scalar(`${name}-overflow`),
    candidateOverflow: scalar(`${name}-candidate`),
    uncertainCount: scalar(`${name}-uncertain`)
  });
  const output = {
    nodedSegments: {
      endpoints: allocate('polygon-noded', 'float32x4', 6),
      sourceFeatureIds: allocate('polygon-noded-features', 'uint32', 6),
      sourceRingIds: allocate('polygon-noded-parts', 'uint32', 6),
      sourceEdgeIds: allocate('polygon-noded-edges', 'uint32', 6),
      sourceStartParameters: allocate('polygon-starts', 'float32', 6),
      sourceEndParameters: allocate('polygon-ends', 'float32', 6),
      status: status('polygon-noded')
    },
    directedEdges: {
      endpoints: allocate('polygon-directed', 'float32x4', 12),
      sourceSegmentIds: allocate('polygon-directed-segments', 'uint32', 12),
      sourceFeatureIds: allocate('polygon-directed-features', 'uint32', 12),
      twinIds: allocate('polygon-twins', 'uint32', 12),
      ringIds: allocate('polygon-rings', 'uint32', 12),
      flags: allocate('polygon-flags', 'uint32', 12),
      status: status('polygon-directed')
    },
    polygons: {
      kind: 'polygons' as const,
      // Four directed triangular walks (bounded and unbounded), each with its closing vertex,
      // need 16 assembly scratch rows even though the final two polygons use only six rows.
      positions: allocate('polygon-positions', 'float32x2', 16),
      featureOffsets: allocate('polygon-output-features', 'uint32', 5),
      polygonOffsets: allocate('polygon-output-polygons', 'uint32', 5),
      ringOffsets: allocate('polygon-output-rings', 'uint32', 5),
      sourceIds: allocate('polygon-output-identities', 'uint32', 4)
    },
    diagnostics: {
      edgeClasses: allocate('polygon-classes', 'uint32', 6),
      closedRingCount: scalar('polygon-closed'),
      openChainCount: scalar('polygon-open'),
      cutEdgeCount: scalar('polygon-cuts'),
      dangleCount: scalar('polygon-dangles'),
      unusedEdgeCount: scalar('polygon-unused')
    },
    status: {
      count: scalar('polygon-count'),
      requiredRingCount: scalar('polygon-required-rings'),
      overflow: scalar('polygon-overflow'),
      candidateOverflow: scalar('polygon-candidate'),
      uncertainCount: scalar('polygon-uncertain')
    }
  };
  const lattice = snapped ? allocate('polygon-lattice', 'float32', 4) : undefined;
  const contributor = new GPUPolygonize({
    lines,
    intersectionCapacity: 8,
    precision: {vertexTolerance: 0.001, predicates: 'exact-or-uncertain', storage: 'float32'},
    output,
    ...(grouped ? {groupIds: allocate('polygon-groups', 'uint32', 3, [90, 7, 19])} : {}),
    ...(snapped ? {snapRounding: {gridSize: 1, lattice: lattice!}} : {})
  });
  const expected: Record<string, number[]> = {
    'nodedSegments.endpoints': endpoints,
    'nodedSegments.sourceFeatureIds': [17, 17, 17, 42, 42, 42],
    'nodedSegments.sourceRingIds': [0, 1, 2, 3, 4, 5],
    'nodedSegments.sourceEdgeIds': [0, 2, 4, 6, 8, 10],
    'nodedSegments.sourceStartParameters': [0, 0, 0, 0, 0, 0],
    'nodedSegments.sourceEndParameters': [1, 1, 1, 1, 1, 1],
    'directedEdges.endpoints': Array.from({length: 6}, (_, row) => {
      const [firstX, firstY, lastX, lastY] = endpoints.slice(row * 4, row * 4 + 4);
      return [firstX, firstY, lastX, lastY, lastX, lastY, firstX, firstY];
    }).flat(),
    'directedEdges.sourceSegmentIds': [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    'directedEdges.sourceFeatureIds': [17, 17, 17, 17, 17, 17, 42, 42, 42, 42, 42, 42],
    'directedEdges.twinIds': [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10],
    'directedEdges.ringIds': [0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3],
    'directedEdges.flags': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'polygons.positions': [0, 0, 2, 0, 0, 2, 5, 0, 7, 0, 5, 2],
    'polygons.featureOffsets': [0, 1, 2, 2, 2],
    'polygons.polygonOffsets': [0, 1, 2, 2, 2],
    'polygons.ringOffsets': [0, 3, 6, 6, 6],
    'polygons.sourceIds': grouped ? [7, 19] : [1, 2],
    'diagnostics.edgeClasses': [1, 1, 1, 1, 1, 1],
    'diagnostics.closedRingCount': [6],
    'diagnostics.openChainCount': [0],
    'diagnostics.cutEdgeCount': [0],
    'diagnostics.dangleCount': [0],
    'diagnostics.unusedEdgeCount': [0],
    'status.count': [2],
    'status.requiredRingCount': [4],
    'status.overflow': [0],
    'status.candidateOverflow': [0],
    'status.uncertainCount': [0]
  };
  for (const [prefix, count] of [
    ['nodedSegments', 6],
    ['directedEdges', 12]
  ] as const)
    for (const name of [
      'count',
      'requiredCount',
      'overflow',
      'candidateOverflow',
      'uncertainCount'
    ])
      expected[`${prefix}.status.${name}`] = [
        name === 'count' || name === 'requiredCount' ? count : 0
      ];
  return {output, expected, lattice, nodes: contributor.getCommandNodes(graph)};
}

export function getMultipartNestedView(root: object, path: string): GraphDataView {
  return path
    .split('.')
    .reduce(
      (value, key) => (value as Record<string, unknown>)[key],
      root as unknown
    ) as GraphDataView;
}

export function createMultipartNodingCase(
  graph: GPUCommandGraph,
  allocate: MultipartTestAllocate,
  sourceIds = true,
  snapped = false
) {
  const lines = {
    kind: 'lines' as const,
    positions: allocate('input-positions', 'float32x2', 6, [0, 0, 2, 2, 0, 2, 2, 0, 5, 0, 7, 0]),
    lineOffsets: allocate('input-lines', 'uint32', 4, [0, 2, 4, 6]),
    featureOffsets: allocate('input-features', 'uint32', 4, [0, 0, 2, 3]),
    ...(sourceIds ? {sourceIds: allocate('input-identities', 'uint32', 3, [90, 17, 42])} : {})
  };
  const output = {
    endpoints: allocate('endpoints', 'float32x4', 5),
    sourceFeatureIds: allocate('feature-identities', 'uint32', 5),
    sourceRingIds: allocate('part-rows', 'uint32', 5),
    sourceEdgeIds: allocate('edge-rows', 'uint32', 5),
    sourceStartParameters: allocate('start-parameters', 'float32', 5),
    sourceEndParameters: allocate('end-parameters', 'float32', 5),
    status: {
      count: allocate('count', 'uint32', 1),
      requiredCount: allocate('required', 'uint32', 1),
      overflow: allocate('overflow', 'uint32', 1),
      candidateOverflow: allocate('candidate-overflow', 'uint32', 1),
      uncertainCount: allocate('uncertain', 'uint32', 1)
    }
  };
  const lattice = snapped ? allocate('lattice', 'float32', 4) : undefined;
  const latticeEndpoints = snapped ? allocate('lattice-endpoints', 'float32x4', 5) : undefined;
  const contributor = snapped
    ? new GPUSnapRoundedNoding({
        lines,
        intersectionCapacity: 8,
        gridSize: 1,
        output: {...output, lattice: lattice!, latticeEndpoints: latticeEndpoints!}
      })
    : new GPULineNoding({lines, intersectionCapacity: 8, output});
  return {lines, output, lattice, latticeEndpoints, nodes: contributor.getCommandNodes(graph)};
}
