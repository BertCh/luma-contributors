// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  getWGSLFloatLiteral,
  createWGSLKernelNode
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GEODESIC_WGSL} from '../geometry-measures/geodesic-wgsl';

/** Row, staypoint and location sentinel: the row is outside every result. */
export const GPU_MOVEMENT_SENTINEL = 0xffffffff;

/** WGSL literal of {@link GPU_MOVEMENT_SENTINEL}. @internal */
export const MOVEMENT_SENTINEL_WGSL = '0xffffffffu';

/**
 * WGSL of `rowPosition(row)` and `rowDistance(a, b)` over a `positions` float32x2 binding: planar
 * Euclidean, or haversine great-circle distance on a sphere of the given radius (longitude and
 * latitude in degrees). @internal
 */
export function getMovementDistanceWGSL(sphereRadius: number | undefined): string {
  const distance =
    sphereRadius === undefined
      ? 'length(rowPosition(a) - rowPosition(b))'
      : `geodesicHaversineDistance(rowPosition(a), rowPosition(b), ${getWGSLFloatLiteral(sphereRadius)})`;
  return /* wgsl */ `${sphereRadius === undefined ? '' : GEODESIC_WGSL}
fn rowPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

fn rowDistance(a: u32, b: u32) -> f32 {
  return ${distance};
}
`;
}

/** Rows of `rowStarts`, `rowEnds` and `rowTracks` per row of a trajectory port. @internal */
export type MovementRowBounds = {
  rowStarts: GraphDataView<'uint32'>;
  rowEnds: GraphDataView<'uint32'>;
  rowTracks: GraphDataView<'uint32'>;
};

/**
 * Nodes that write each row's track start, track end (exclusive) and track index; rows outside
 * every track get the sentinel. @internal
 */
export function createRowBoundsNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    trackOffsets: GraphDataView<'uint32'>;
    rowCount: number;
  } & MovementRowBounds
): GPUCommandNode<Parameters>[] {
  const {id, operation, trackOffsets, rowCount, rowStarts, rowEnds, rowTracks} = props;
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-reset`,
      operation,
      variant: 'row-bounds',
      invocationCount: rowCount,
      bindings: [
        {name: 'rowStarts', view: rowStarts, type: 'u32', access: 'read_write'},
        {name: 'rowEnds', view: rowEnds, type: 'u32', access: 'read_write'},
        {name: 'rowTracks', view: rowTracks, type: 'u32', access: 'read_write'}
      ],
      body: `rowStarts[rowStartsOffset + index] = ${MOVEMENT_SENTINEL_WGSL};
  rowEnds[rowEndsOffset + index] = ${MOVEMENT_SENTINEL_WGSL};
  rowTracks[rowTracksOffset + index] = ${MOVEMENT_SENTINEL_WGSL};`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-fill`,
      operation,
      variant: 'row-bounds',
      invocationCount: trackOffsets.length - 1,
      bindings: [
        {name: 'trackOffsets', view: trackOffsets, type: 'u32', access: 'read'},
        {name: 'rowStarts', view: rowStarts, type: 'u32', access: 'read_write'},
        {name: 'rowEnds', view: rowEnds, type: 'u32', access: 'read_write'},
        {name: 'rowTracks', view: rowTracks, type: 'u32', access: 'read_write'}
      ],
      body: `let begin = min(trackOffsets[trackOffsetsOffset + index], ${rowCount}u);
  let end = min(trackOffsets[trackOffsetsOffset + index + 1u], ${rowCount}u);
  for (var row = begin; row < end; row++) {
    rowStarts[rowStartsOffset + row] = begin;
    rowEnds[rowEndsOffset + row] = end;
    rowTracks[rowTracksOffset + row] = index;
  }`
    })
  ];
}

/** Node writing `rowIds[i] = i`. @internal */
export function createRowIdsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {id: string; operation: string; rowIds: GraphDataView<'uint32'>}
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'row-ids',
    invocationCount: props.rowIds.length,
    bindings: [{name: 'rowIds', view: props.rowIds, type: 'u32', access: 'read_write'}],
    body: 'rowIds[rowIdsOffset + index] = index;'
  });
}

/** Number of doubling passes that resolve every chain inside `rowCount` rows. @internal */
export function getChainPassCount(rowCount: number): number {
  return Math.max(1, 32 - Math.clz32(Math.max(rowCount - 1, 1)));
}

/**
 * Pointer-doubling chain membership. Each track's first row seeds `reach`; `next` maps a row to
 * its successor (a row that ends its chain points at itself). After `getChainPassCount` passes
 * `reach` holds 1 for every row on a chain from a track start: distances `0..2^passes - 1` are the
 * sums of distinct powers of two, so the pass count is bounded by `log2(rowCount)`, independent of
 * the data. Every mark is on a chain, so a racing load never adds a false member; marks of earlier
 * passes are complete at a pass boundary, so a racing load never loses one either. @internal
 */
export function createChainReachNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    trackOffsets: GraphDataView<'uint32'>;
    next: GraphDataView<'uint32'>;
    reach: GraphDataView<'uint32'>;
    rowCount: number;
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation, trackOffsets, next, reach, rowCount} = props;
  const passCount = getChainPassCount(rowCount);
  const jumpEven = createTransientView(graph, `${id}-jump-even`, 'uint32', rowCount);
  const jumpOdd = createTransientView(graph, `${id}-jump-odd`, 'uint32', rowCount);
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-reset`,
      operation,
      variant: 'chain-reach',
      invocationCount: rowCount,
      bindings: [{name: 'reach', view: reach, type: 'u32', access: 'read_write'}],
      body: 'reach[reachOffset + index] = 0u;'
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-seed`,
      operation,
      variant: 'chain-reach',
      invocationCount: trackOffsets.length - 1,
      bindings: [
        {name: 'trackOffsets', view: trackOffsets, type: 'u32', access: 'read'},
        {name: 'reach', view: reach, type: 'atomic<u32>', access: 'read_write'}
      ],
      body: `let begin = min(trackOffsets[trackOffsetsOffset + index], ${rowCount}u);
  let end = min(trackOffsets[trackOffsetsOffset + index + 1u], ${rowCount}u);
  if (begin < end) { atomicStore(&reach[reachOffset + begin], 1u); }`
    })
  ];
  for (let pass = 0; pass < passCount; pass++) {
    const jumpIn = pass === 0 ? next : pass % 2 === 1 ? jumpEven : jumpOdd;
    const jumpOut = pass % 2 === 0 ? jumpEven : jumpOdd;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-pass-${pass}`,
        operation,
        variant: 'chain-reach',
        invocationCount: rowCount,
        bindings: [
          {name: 'reach', view: reach, type: 'atomic<u32>', access: 'read_write'},
          {name: 'jumpIn', view: jumpIn, type: 'u32', access: 'read'},
          {name: 'jumpOut', view: jumpOut, type: 'u32', access: 'read_write'}
        ],
        body: `let jump = jumpIn[jumpInOffset + index];
  if (atomicLoad(&reach[reachOffset + index]) != 0u) { atomicStore(&reach[reachOffset + jump], 1u); }
  jumpOut[jumpOutOffset + index] = jumpIn[jumpInOffset + jump];`
      })
    );
  }
  return nodes;
}

/**
 * Node writing `offsets[track]` = the rows of a sorted-by-track `tracks` column belonging to tracks
 * before `track`, with one terminal entry. One serial thread over the table. @internal
 */
export function createTrackTableOffsetsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    tracks: GraphDataView<'uint32'>;
    count: GraphDataView<'uint32'>;
    offsets: GraphDataView<'uint32'>;
    capacity: number;
    trackCount: number;
  }
): GPUCommandNode<Parameters> {
  const {id, operation, tracks, count, offsets, capacity, trackCount} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id,
    operation,
    variant: 'track-offsets',
    invocationCount: 1,
    bindings: [
      {name: 'tracks', view: tracks, type: 'u32', access: 'read'},
      {name: 'total', view: count, type: 'u32', access: 'read'},
      {name: 'offsets', view: offsets, type: 'u32', access: 'read_write'}
    ],
    body: `let total = min(total[totalOffset], ${capacity}u);
  var row = 0u;
  offsets[offsetsOffset] = 0u;
  for (var track = 0u; track < ${trackCount}u; track++) {
    while (row < total && tracks[tracksOffset + row] == track) { row++; }
    offsets[offsetsOffset + track + 1u] = row;
  }`
  });
}
