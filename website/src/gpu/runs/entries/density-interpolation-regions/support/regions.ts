// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Inputs shared by the regionalization runs: contiguous tracts with a two-column attribute table
// (poverty and unemployment shares), population and rook weights.

import {createTracts, type TractsInput} from '../../../inputs';
import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
import type {RunBuffer, RunBuilder, RunLayer} from '../../../define-run';

export type RegionScene = {
  tracts: TractsInput;
  columnCount: number;
  /** Row-major: tract i, column c at i * columnCount + c. */
  attributes: Float32Array;
  /** An extensive attribute (population) per tract. */
  population: Float32Array;
};

export function createRegionScene(
  random: () => number,
  world: {width: number; height: number},
  rows = 7,
  columnCount = 2
): RegionScene {
  const tracts = createTracts(random, world, rows, 0);
  const attributes = new Float32Array(tracts.count * columnCount);
  for (let tract = 0; tract < tracts.count; tract++) {
    attributes.set(
      [tracts.povertyShare[tract], tracts.unemploymentShare[tract]],
      tract * columnCount
    );
  }
  return {tracts, columnCount, attributes, population: tracts.population};
}

/** `[row, neighbor]` pair per adjacency, for drawing the contiguity graph. */
export function createAdjacencyEdges(tracts: TractsInput): Uint32Array {
  const pairs: number[] = [];
  const {offsets, neighbors} = tracts.contiguity;
  for (let row = 0; row < tracts.count; row++) {
    for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
      if (neighbors[slot] > row) pairs.push(row, neighbors[slot]);
    }
  }
  return Uint32Array.from(pairs);
}

/** Views a uint32 buffer of `[a, b]` pairs as the uint32x2 edge buffer the `segments` layer reads. */
export function asPairs(buffer: RunBuffer<'uint32'>): RunBuffer {
  return {...buffer, format: 'uint32x2', length: buffer.length / 2};
}

/** Label-colored tracts plus outline, for every regionalization result. */
export function drawLabels(
  scene: RegionScene,
  mesh: {
    vertices: RunBuffer<'float32x2'>;
    indices: RunBuffer<'uint32'>;
    owners: RunBuffer<'uint32'>;
    outline: RunBuffer<'float32x4'>;
  },
  labels: RunBuffer<'uint32'>
) {
  return [
    {
      layer: 'polygons' as const,
      positions: mesh.vertices,
      indices: mesh.indices,
      owners: mesh.owners,
      values: labels,
      triangleCount: scene.tracts.triangleCount
    },
    {
      layer: 'lines' as const,
      context: true,
      segments: mesh.outline,
      count: scene.tracts.outlineCount,
      alpha: 0.6
    }
  ];
}

/** Outline segments of the tracts labelled `label`, written into a render-only lines buffer (NaN elsewhere). */
export function writeRegionOutline(
  scene: Pick<RegionScene, 'tracts'>,
  labels: ArrayLike<number>,
  label: number,
  target: RunBuffer<'float32x4'>
): number {
  const {tracts} = scene;
  const lit = new Float32Array(tracts.outlineCount * 4).fill(Number.NaN);
  let tractCount = 0;
  for (let tract = 0; tract < tracts.count; tract++) {
    if (labels[tract] !== label) continue;
    tractCount++;
    const start = tracts.ringOffsets[tract];
    const end = tracts.ringOffsets[tract + 1];
    lit.set(tracts.outline.subarray(start * 4, end * 4), start * 4);
  }
  target.write(lit);
  return tractCount;
}

/** An empty render-only outline buffer for `writeRegionOutline`. */
export function allocateRegionOutline(
  builder: RunBuilder,
  scene: Pick<RegionScene, 'tracts'>,
  id = 'highlight'
) {
  return builder.upload(
    id,
    'float32x4',
    new Float32Array(scene.tracts.outlineCount * 4).fill(Number.NaN)
  );
}

/** The bright outline layer of a highlighted region. */
export function drawRegionOutline(
  scene: Pick<RegionScene, 'tracts'>,
  highlight: RunBuffer<'float32x4'>
): RunLayer {
  return {
    layer: 'lines',
    segments: highlight,
    count: scene.tracts.outlineCount,
    radius: 2.2,
    color: [1, 0.9, 0.35, 1]
  };
}

/**
 * The shared edges between tracts that carry different `labels`, as segments: the borders of a
 * partition. Edges are matched by their end points.
 */
export function createPartitionBorders(
  scene: RegionScene,
  labels: ArrayLike<number>
): Float32Array {
  const {tracts} = scene;
  const owners = new Map<string, number>();
  const borders: number[] = [];
  const key = (x0: number, y0: number, x1: number, y1: number) =>
    x0 < x1 || (x0 === x1 && y0 < y1) ? `${x0},${y0},${x1},${y1}` : `${x1},${y1},${x0},${y0}`;
  for (let tract = 0; tract < tracts.count; tract++) {
    for (let row = tracts.ringOffsets[tract]; row < tracts.ringOffsets[tract + 1]; row++) {
      const [x0, y0, x1, y1] = tracts.outline.subarray(row * 4, row * 4 + 4);
      const edgeKey = key(x0, y0, x1, y1);
      const other = owners.get(edgeKey);
      if (other === undefined) owners.set(edgeKey, tract);
      else if (labels[other] !== labels[tract]) borders.push(x0, y0, x1, y1);
    }
  }
  return Float32Array.from(borders);
}

/**
 * Adds a node that copies `segments` (one per row) where `flags` is nonzero and writes NaN
 * elsewhere, so a lines layer shows only the flagged ones.
 */
export function addFlaggedSegments(
  builder: RunBuilder,
  flags: RunBuffer<'uint32'>,
  segments: RunBuffer<'float32x4'>,
  id: string
): RunBuffer<'float32x4'> {
  const target = builder.allocate(id, 'float32x4', segments.length);
  builder.graph.add({
    getCommandNodes: graph => [
      createWGSLKernelNode(graph, {
        id,
        operation: 'flagged-segments',
        bindings: [
          {name: 'segmentFlags', view: flags.view, type: 'u32', access: 'read'},
          {name: 'segmentSource', view: segments.view, type: 'f32', access: 'read'},
          {name: 'segmentTarget', view: target.view, type: 'f32', access: 'read_write'}
        ],
        invocationCount: segments.length,
        body: `let isFlagged = segmentFlags[segmentFlagsOffset + index] != 0u;
for (var component = 0u; component < 4u; component = component + 1u) {
  segmentTarget[segmentTargetOffset + index * 4u + component] = select(bitcast<f32>(segmentFlags[segmentFlagsOffset + index] | 0x7fc00000u), segmentSource[segmentSourceOffset + index * 4u + component], isFlagged);
}`
      })
    ]
  });
  return target;
}

/**
 * Adds a node writing `segments[ids[row]]` per row and NaN for ids outside the table (such as an
 * unused cut-log entry), so a lines layer shows only the ids that point at a segment.
 */
export function addSegmentGather(
  builder: RunBuilder,
  ids: RunBuffer<'uint32'>,
  segments: RunBuffer<'float32x4'>,
  id: string
): RunBuffer<'float32x4'> {
  const target = builder.allocate(id, 'float32x4', ids.length);
  builder.graph.add({
    getCommandNodes: graph => [
      createWGSLKernelNode(graph, {
        id,
        operation: 'segment-gather',
        bindings: [
          {name: 'gatherIds', view: ids.view, type: 'u32', access: 'read'},
          {name: 'gatherSource', view: segments.view, type: 'f32', access: 'read'},
          {name: 'gatherTarget', view: target.view, type: 'f32', access: 'read_write'}
        ],
        invocationCount: ids.length,
        body: `let slot = gatherIds[gatherIdsOffset + index];
let isValid = slot < ${segments.length}u;
let safeSlot = min(slot, ${segments.length - 1}u);
for (var component = 0u; component < 4u; component = component + 1u) {
  gatherTarget[gatherTargetOffset + index * 4u + component] = select(bitcast<f32>(slot | 0x7fc00000u), gatherSource[gatherSourceOffset + safeSlot * 4u + component], isValid);
}`
      })
    ]
  });
  return target;
}
