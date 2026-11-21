// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPUGeometryPredicates,
  type GPUGeometryPredicatesOutput
} from '../../../src/gpu-spatial-analysis/geometry-predicates/gpu-geometry-predicates';
import type {MultipartTestAllocate} from '../geometry-topology/multipart-line-forms-fixtures';
import {GPULineSplit} from '../../../src/gpu-spatial-analysis/line-split/gpu-line-split';
import {GPUTrackSimilarity} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity';
import {GPUTrackSimilaritySearch} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity-search';
import {GPUMinimumClearance} from '../../../src/gpu-spatial-analysis/minimum-clearance/gpu-minimum-clearance';
import {GPUGeometryMeasures} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geometry-measures';
import {GPULineLoopRemoval} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-line-loop-removal';
export function createIdentityLoopReviewCase(
  graph: GPUCommandGraph,
  allocate: MultipartTestAllocate
) {
  const lines = {
    kind: 'lines' as const,
    positions: allocate('identity-loop-input', 'float32x2', 2, [0, 0, 2, 0]),
    lineOffsets: allocate('identity-loop-parts', 'uint32', 2, [0, 2])
  };
  const output = {
    positions: allocate('identity-loop-output', 'float32x2', 2),
    lineOffsets: allocate('identity-loop-output-parts', 'uint32', 2),
    featureOffsets: allocate('identity-loop-output-features', 'uint32', 2),
    sourceEdges: allocate('identity-loop-output-edges', 'uint32', 2),
    status: allocate('identity-loop-status', 'uint32x4', 1)
  };
  return {
    output,
    expected: {
      positions: [0, 0, 2, 0],
      lineOffsets: [0, 2],
      featureOffsets: [0, 1],
      sourceEdges: [0, 0],
      status: [2, 2, 0, 0]
    } as Record<string, number[]>,
    nodes: new GPULineLoopRemoval({lines, intersectionCapacity: 4, output}).getCommandNodes(graph)
  };
}
export function createInheritedMeasureReviewCase(
  graph: GPUCommandGraph,
  allocate: MultipartTestAllocate
) {
  const positions = allocate('getter-positions', 'float32x2', 4, [0, 0, 1, 0, 2, 0, 3, 0]);
  const lineOffsets = allocate('getter-parts', 'uint32', 3, [0, 2, 4]);
  const featureOffsets = allocate('getter-features', 'uint32', 2, [0, 2]);
  const lengths = allocate('getter-length', 'float32', 1);
  let inputReads = 0;
  let outputReads = 0;
  const props = Object.assign(
    Object.create({
      get featureOffsets() {
        inputReads++;
        return featureOffsets;
      }
    }),
    {
      geometryType: 'lines' as const,
      positions,
      lineOffsets,
      output: Object.create({
        get lengths() {
          outputReads++;
          return lengths;
        }
      })
    }
  );
  const nodes = new GPUGeometryMeasures(props).getCommandNodes(graph);
  if (inputReads !== 1 || outputReads !== 1)
    throw new Error(`getter snapshot read counts ${inputReads}/${outputReads}, expected 1/1`);
  return {output: {lengths}, expected: {lengths: [2]} as Record<string, number[]>, nodes};
}
export function createZeroAreaReviewCase(
  graph: GPUCommandGraph,
  allocate: MultipartTestAllocate,
  kind: 'points' | 'lines'
) {
  const common = {
    positions: allocate(
      'zero-area-input',
      'float32x2',
      kind === 'points' ? 2 : 4,
      kind === 'points' ? [0, 0, 2, 0] : [0, 0, 1, 0, 2, 0, 3, 0]
    ),
    featureOffsets: allocate('zero-area-features', 'uint32', 3, [0, 1, 2])
  };
  const layout =
    kind === 'points'
      ? {...common, geometryType: 'points' as const}
      : {
          ...common,
          geometryType: 'lines' as const,
          lineOffsets: allocate('zero-area-parts', 'uint32', 3, [0, 2, 4])
        };
  const output = {areas: allocate('zero-area-output', 'float32', 2)};
  const groups = {
    areas: allocate('zero-area-groups', 'float32', 3),
    featureCounts: allocate('zero-area-counts', 'uint32', 3)
  };
  return {
    output: {output, groups},
    expected: {
      'output.areas': [0, 0],
      'groups.areas': [0, 0, 0],
      'groups.featureCounts': [1, 1, 0]
    } as Record<string, number[]>,
    nodes: new GPUGeometryMeasures({
      ...layout,
      output,
      groupIds: allocate('zero-area-ids', 'uint32', 2, [1, 0]),
      groupCount: 3,
      groupOutput: groups
    }).getCommandNodes(graph)
  };
}
export function createMeasureReviewCase(
  graph: GPUCommandGraph,
  allocate: MultipartTestAllocate,
  mode: 'bounds-points' | 'bounds-lines' | 'collapsed',
  cooperative = false,
  repeats = 4
) {
  const boundsCase = mode !== 'collapsed';
  const coordinates = boundsCase
    ? [1000000, 0, 0.03125, 0, 0.015625, 0]
    : [0, 0, 0, 0, ...new Array(repeats).fill([9, 0]).flat()];
  const shared = {
    positions: allocate('measure-review-input', 'float32x2', coordinates.length / 2, coordinates),
    featureOffsets: allocate('measure-review-features', 'uint32', 2, [
      0,
      mode === 'bounds-points' ? 3 : boundsCase ? 1 : 2
    ])
  };
  const layout =
    mode === 'bounds-points'
      ? {...shared, geometryType: 'points' as const}
      : {
          ...shared,
          geometryType: 'lines' as const,
          lineOffsets: allocate(
            'measure-review-parts',
            'uint32',
            boundsCase ? 2 : 3,
            boundsCase ? [0, 3] : [0, 2, repeats + 2]
          )
        };
  const output = boundsCase
    ? {
        bounds: allocate('review-bounds', 'float32x4', 1),
        extremeVertices: allocate('review-extremes', 'uint32x4', 1)
      }
    : {
        centroids: allocate('review-centroid', 'float32x2', 1),
        lengths: allocate('review-length', 'float32', 1)
      };
  const expected: Record<string, number[]> = boundsCase
    ? {bounds: [0.015625, 0, 1000000, 0], extremeVertices: [2, 0, 0, 0]}
    : {centroids: [4.5, 0], lengths: [0]};
  return {
    output,
    expected,
    nodes: new GPUGeometryMeasures({
      id: `review-measures-${mode}-${repeats}-${cooperative}`,
      ...layout,
      cooperativeRingRows: cooperative ? 1 : 0,
      output
    }).getCommandNodes(graph)
  };
}
export function createExtremeBoundsReviewCase(
  graph: GPUCommandGraph,
  allocate: MultipartTestAllocate,
  kind: 'points' | 'lines',
  cooperative = false
) {
  // FLT_MAX is exactly representable. Each feature owns repeated coordinates, so local
  // moment arithmetic remains zero; only the absolute bounds exercise the full range.
  const maximum = 3.4028234663852886e38;
  const positions = allocate('maximum-positions', 'float32x2', 4, [
    maximum,
    0,
    maximum,
    0,
    -maximum,
    0,
    -maximum,
    0
  ]);
  const output = {
    bounds: allocate('maximum-bounds', 'float32x4', 2),
    extremeVertices: allocate('maximum-extremes', 'uint32x4', 2)
  };
  const groups = {bounds: allocate('maximum-group-bounds', 'float32x4', 3)};
  const layout =
    kind === 'points'
      ? {
          geometryType: 'points' as const,
          positions,
          featureOffsets: allocate('maximum-features', 'uint32', 3, [0, 2, 4])
        }
      : {
          geometryType: 'lines' as const,
          positions,
          lineOffsets: allocate('maximum-parts', 'uint32', 3, [0, 2, 4]),
          featureOffsets: allocate('maximum-features', 'uint32', 3, [0, 1, 2])
        };
  return {
    output: {output, groups},
    expected: {
      'output.bounds': [maximum, 0, maximum, 0, -maximum, 0, -maximum, 0],
      'output.extremeVertices': [0, 0, 0, 0, 2, 2, 2, 2],
      'groups.bounds': [maximum, 0, maximum, 0, -maximum, 0, -maximum, 0, NaN, NaN, NaN, NaN]
    },
    nodes: new GPUGeometryMeasures({
      ...layout,
      output,
      groupOutput: groups,
      groupCount: 3,
      groupIds: allocate('maximum-group-ids', 'uint32', 2, [0, 1]),
      cooperativeRingRows: cooperative ? 1 : 0
    }).getCommandNodes(graph)
  };
}
export function createCollapsedClearanceReviewCase(
  graph: GPUCommandGraph,
  allocate: MultipartTestAllocate,
  single = false,
  tie = false,
  spatialSort = false
) {
  const geometry = {
    kind: 'lines' as const,
    positions: allocate(
      'collapsed-clearance-input',
      'float32x2',
      single ? 2 : 6,
      single
        ? [0, 0, 0, 0]
        : tie
          ? [0, 0, 0, 0, 1, 0, 1, 0, -1, 0, -1, 0]
          : [0, 0, 0, 0, 9, 0, 9, 0, 9, 0, 9, 0]
    ),
    lineOffsets: allocate(
      'collapsed-clearance-parts',
      'uint32',
      single ? 2 : tie ? 4 : 3,
      single ? [0, 2] : tie ? [0, 2, 4, 6] : [0, 2, 6]
    ),
    featureOffsets: allocate('collapsed-clearance-features', 'uint32', 2, [
      0,
      single ? 1 : tie ? 3 : 2
    ])
  };
  const output = {
    clearances: allocate('collapsed-clearance-distance', 'float32', 1),
    lines: allocate('collapsed-clearance-line', 'float32x4', 1),
    vertexIds: allocate('collapsed-clearance-row', 'uint32', 1),
    status: {candidateOverflow: allocate('collapsed-clearance-status', 'uint32', 1)}
  };
  return {
    output,
    expected: {
      clearances: [single ? Infinity : tie ? 1 : 9],
      lines: single ? [NaN, NaN, NaN, NaN] : [0, 0, tie ? 1 : 9, 0],
      vertexIds: [single ? 0xffffffff : 0],
      'status.candidateOverflow': [0]
    } as Record<string, number[]>,
    nodes: new GPUMinimumClearance({geometry, output, spatialSort}).getCommandNodes(graph)
  };
}
export function createEmptyTrackReviewCase(
  graph: GPUCommandGraph,
  allocate: MultipartTestAllocate
) {
  const lines = {
    kind: 'lines' as const,
    positions: allocate('empty-track-input', 'float32x2', 0, []),
    lineOffsets: allocate('empty-track-parts', 'uint32', 1, [0]),
    featureOffsets: allocate('empty-track-features', 'uint32', 3, [0, 0, 0])
  };
  const pairA = allocate('empty-track-a', 'uint32', 1, [0]);
  const pairB = allocate('empty-track-b', 'uint32', 1, [1]);
  const track = {
    hausdorff: allocate('empty-h', 'float32', 1),
    maximumDistance: allocate('empty-max', 'float32', 1),
    symmetricSegmentPathDistance: allocate('empty-path', 'float32', 1),
    pairStatus: allocate('empty-track-status', 'uint32', 1)
  };
  const search = {
    distance: allocate('empty-search-distance', 'float32', 1),
    lowerBound: allocate('empty-search-bound', 'float32', 1),
    matched: allocate('empty-search-matched', 'uint32', 1),
    status: allocate('empty-search-status', 'uint32', 1),
    survivorSummary: allocate('empty-search-summary', 'uint32', 3)
  };
  const clearance = {
    clearances: allocate('empty-clearance', 'float32', 2),
    lines: allocate('empty-clearance-lines', 'float32x4', 2),
    vertexIds: allocate('empty-clearance-vertex', 'uint32', 2),
    status: {candidateOverflow: allocate('empty-clearance-overflow', 'uint32', 1)}
  };
  const nodes = [
    ...new GPUTrackSimilarity({
      id: 'review-empty-track',
      linesA: lines,
      pairA,
      pairB,
      output: track
    }).getCommandNodes(graph),
    ...new GPUTrackSimilaritySearch({
      id: 'review-empty-search',
      linesA: lines,
      pairA,
      pairB,
      metric: 'hausdorff',
      threshold: allocate('empty-search-threshold', 'float32', 1, [1]),
      output: search
    }).getCommandNodes(graph),
    ...new GPUMinimumClearance({geometry: lines, output: clearance}).getCommandNodes(graph)
  ];
  return {
    output: {track, search, clearance},
    expected: {
      'track.hausdorff': [NaN],
      'track.maximumDistance': [NaN],
      'track.symmetricSegmentPathDistance': [NaN],
      'track.pairStatus': [1],
      'search.distance': [NaN],
      'search.lowerBound': [NaN],
      'search.matched': [0],
      'search.status': [2],
      'search.survivorSummary': [0, 0, 0],
      'clearance.clearances': [Infinity, Infinity],
      'clearance.lines': new Array(8).fill(NaN),
      'clearance.vertexIds': [0xffffffff, 0xffffffff],
      'clearance.status.candidateOverflow': [0]
    } as Record<string, number[]>,
    nodes
  };
}
export function createPaddedSplitReviewCase(
  graph: GPUCommandGraph,
  allocate: MultipartTestAllocate
) {
  const lines = {
    kind: 'lines' as const,
    positions: allocate('padded-input', 'float32x2', 2, [0, 0, 2, 0]),
    lineOffsets: allocate('padded-input-parts', 'uint32', 2, [0, 2]),
    featureOffsets: allocate('padded-input-features', 'uint32', 2, [0, 1])
  };
  const output = {
    pieces: {
      geometry: {
        kind: 'lines' as const,
        positions: allocate('padded-positions', 'float32x2', 8),
        lineOffsets: allocate('padded-parts', 'uint32', 5),
        featureOffsets: allocate('padded-features', 'uint32', 2)
      },
      sourceIds: allocate('padded-source', 'uint32', 4),
      status: {
        count: allocate('padded-count', 'uint32', 1),
        vertexCount: allocate('padded-vertices', 'uint32', 1),
        requiredCount: allocate('padded-required', 'uint32', 1),
        requiredVertexCount: allocate('padded-required-vertices', 'uint32', 1),
        overflow: allocate('padded-overflow', 'uint32', 1),
        candidateOverflow: allocate('padded-candidate', 'uint32', 1)
      }
    }
  };
  return {
    output,
    expected: {
      'pieces.geometry.positions': [0, 0, 2, 0],
      'pieces.geometry.lineOffsets': [0, 2, 2, 2, 2],
      'pieces.geometry.featureOffsets': [0, 1],
      'pieces.sourceIds': [0, 0xffffffff, 0xffffffff, 0xffffffff],
      'pieces.status.count': [1],
      'pieces.status.vertexCount': [2],
      'pieces.status.requiredCount': [1],
      'pieces.status.requiredVertexCount': [2],
      'pieces.status.overflow': [0],
      'pieces.status.candidateOverflow': [0]
    } as Record<string, number[]>,
    nodes: new GPULineSplit({lines, intersectionCapacity: 4, output}).getCommandNodes(graph)
  };
}
export const REVIEW_TOPOLOGICAL_NAMES = [
  'equals',
  'intersects',
  'contains',
  'within',
  'covers',
  'coveredBy',
  'touches',
  'crosses',
  'overlaps',
  'containsProperly'
] as const;
export function createEmptyPointRelateReviewCase(
  graph: GPUCommandGraph,
  allocate: MultipartTestAllocate,
  invalid = false
) {
  const geometry = {
    kind: 'points' as const,
    positions: allocate('empty-point-left', 'float32x2', invalid ? 1 : 0, invalid ? [NaN, 0] : []),
    featureOffsets: allocate('empty-point-left-features', 'uint32', 2, [0, invalid ? 1 : 0])
  };
  const other = {
    kind: 'points' as const,
    positions: allocate('empty-point-right', 'float32x2', 0, []),
    featureOffsets: allocate('empty-point-right-features', 'uint32', 2, [0, 0])
  };
  const output = {
    relate: allocate('empty-point-relate', 'uint32', 1),
    topological: Object.fromEntries(
      REVIEW_TOPOLOGICAL_NAMES.map(name => [name, allocate(`empty-point-${name}`, 'uint32', 1)])
    )
  };
  const expected: Record<string, number[]> = {relate: [invalid ? 0 : 196608]};
  for (const name of REVIEW_TOPOLOGICAL_NAMES)
    expected[`topological.${name}`] = [!invalid && name === 'equals' ? 1 : 0];
  return {
    output,
    expected,
    nodes: new GPUGeometryPredicates({geometry, other, output}).getCommandNodes(graph)
  };
}
export function createAlignedRelateReviewCase(
  graph: GPUCommandGraph,
  allocate: MultipartTestAllocate,
  variant: number
) {
  const geometry = {
    kind: 'lines' as const,
    positions: allocate(
      'aligned-left',
      'float32x2',
      variant === 3 || variant === 4 ? 0 : 4,
      variant === 3 || variant === 4
        ? []
        : variant === 2
          ? [0, 0, 1, 0, 3, 0, 4, 0]
          : [0, 0, 1, 0, 1, 0, 2, 0]
    ),
    lineOffsets: allocate(
      'aligned-left-parts',
      'uint32',
      3,
      variant === 3 || variant === 4 ? [0, 0, 0] : [0, 2, 4]
    ),
    featureOffsets: allocate(
      'aligned-left-features',
      'uint32',
      2,
      variant === 3 || variant === 4 ? [0, 0] : [0, 2]
    )
  };
  const other =
    variant === 1
      ? {kind: 'points' as const, positions: allocate('aligned-point', 'float32x2', 1, [1, 0])}
      : {
          kind: 'lines' as const,
          positions: allocate(
            'aligned-right',
            'float32x2',
            variant === 3 || variant === 5 ? 0 : 2,
            variant === 3 || variant === 5 ? [] : [0, 0, 2, 0]
          ),
          lineOffsets: allocate(
            'aligned-right-parts',
            'uint32',
            2,
            variant === 3 || variant === 5 ? [0, 0] : [0, 2]
          ),
          featureOffsets: allocate(
            'aligned-right-features',
            'uint32',
            2,
            variant === 3 || variant === 5 ? [0, 0] : [0, 1]
          )
        };
  const topological = Object.fromEntries(
    REVIEW_TOPOLOGICAL_NAMES.map(name => [name, allocate(`aligned-${name}`, 'uint32', 1)])
  ) as Record<(typeof REVIEW_TOPOLOGICAL_NAMES)[number], GraphDataView<'uint32'>>;
  const output = {
    relate: allocate('aligned-matrix', 'uint32', 1),
    topological,
    ...(variant !== 1
      ? {
          equalsExact: allocate('aligned-exact', 'uint32', 1),
          equalsIdentical: allocate('aligned-identical', 'uint32', 1)
        }
      : {})
  };
  const booleans = [
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  ][variant];
  const expected: Record<string, number[]> = {
    // Shapely matrices 1FFF0FFF2, 0F1FF0FF2, 1F1000102. Each cell uses
    // two bits (F=0, 0=1, 1=2, 2=3), starting with II in the low bits.
    relate: [[196866, 197665, 222562, 196608, 221184, 197664][variant]]
  };
  for (const [index, name] of REVIEW_TOPOLOGICAL_NAMES.entries())
    expected[`topological.${name}`] = [booleans[index]];
  if (variant !== 1) {
    expected['equalsExact'] = [variant === 3 ? 1 : 0];
    expected['equalsIdentical'] = [variant === 3 ? 1 : 0];
  }
  return {
    output,
    expected,
    nodes: new GPUGeometryPredicates({
      geometry,
      other,
      parameters: allocate('aligned-tolerance', 'float32', 4, [0, 0, 0, 0]),
      output: output as GPUGeometryPredicatesOutput
    }).getCommandNodes(graph)
  };
}
