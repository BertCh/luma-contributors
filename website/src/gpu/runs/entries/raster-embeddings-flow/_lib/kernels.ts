// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Display-only kernels for the raster-embedding and Lagrangian runs: they reshape what an operation
// wrote (interleaved vectors, pixel ids, start and end positions) into what a layer draws. They
// never change the operation's result.

import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
import type {RunBuffer, RunBuilder} from '../../../define-run';

/**
 * Copies component `component[0]` of every `stride`-wide row of `source` into `target`. The
 * component is a buffer, so choosing another one never recompiles.
 */
export function addComponentExtract(
  builder: RunBuilder,
  id: string,
  source: RunBuffer,
  stride: number,
  component: RunBuffer,
  target: RunBuffer
): void {
  builder.graph.add({
    getCommandNodes: graph => [
      createWGSLKernelNode(graph, {
        id,
        operation: 'embedding-component-extract',
        bindings: [
          {name: 'sourceValues', view: source.view, type: 'f32', access: 'read'},
          {name: 'component', view: component.view, type: 'u32', access: 'read'},
          {name: 'targetValues', view: target.view, type: 'f32', access: 'read_write'}
        ],
        invocationCount: target.length,
        body: `targetValues[targetValuesOffset + index] = sourceValues[sourceValuesOffset + index * ${stride}u + component[componentOffset]];`
      })
    ]
  });
}

/** Pixel id to the world position of its center (NaN for `0xffffffff`); the raster fills the world. */
export function addPixelPositions(
  builder: RunBuilder,
  id: string,
  pixelIds: RunBuffer,
  raster: {columns: number; rows: number},
  world: {width: number; height: number},
  target: RunBuffer
): void {
  const cellWidth = world.width / raster.columns;
  const cellHeight = world.height / raster.rows;
  builder.graph.add({
    getCommandNodes: graph => [
      createWGSLKernelNode(graph, {
        id,
        operation: 'embedding-pixel-positions',
        bindings: [
          {name: 'pixelIds', view: pixelIds.view, type: 'u32', access: 'read'},
          {name: 'targetValues', view: target.view, type: 'f32', access: 'read_write'}
        ],
        invocationCount: pixelIds.length,
        body: `let pixel = pixelIds[pixelIdsOffset + index];
  let missing = bitcast<f32>(0x7fc00000u | (index & 0u));
  let column = f32(pixel % ${raster.columns}u);
  let row = f32(pixel / ${raster.columns}u);
  let valid = pixel < ${raster.columns * raster.rows}u;
  targetValues[targetValuesOffset + index * 2u] = select(missing, (column + 0.5) * ${cellWidth.toFixed(8)}, valid);
  targetValues[targetValuesOffset + index * 2u + 1u] = select(missing, ${world.height.toFixed(8)} - (row + 0.5) * ${cellHeight.toFixed(8)}, valid);`
      })
    ]
  });
}

/** Segments from each start position to the matching end position (float32x4 rows). */
export function addDisplacementSegments(
  builder: RunBuilder,
  id: string,
  starts: RunBuffer,
  ends: RunBuffer,
  target: RunBuffer
): void {
  builder.graph.add({
    getCommandNodes: graph => [
      createWGSLKernelNode(graph, {
        id,
        operation: 'flow-displacement-segments',
        bindings: [
          {name: 'starts', view: starts.view, type: 'f32', access: 'read'},
          {name: 'ends', view: ends.view, type: 'f32', access: 'read'},
          {name: 'targetValues', view: target.view, type: 'f32', access: 'read_write'}
        ],
        invocationCount: target.length,
        body: `targetValues[targetValuesOffset + index * 4u] = starts[startsOffset + index * 2u];
  targetValues[targetValuesOffset + index * 4u + 1u] = starts[startsOffset + index * 2u + 1u];
  targetValues[targetValuesOffset + index * 4u + 2u] = ends[endsOffset + index * 2u];
  targetValues[targetValuesOffset + index * 4u + 3u] = ends[endsOffset + index * 2u + 1u];`
      })
    ]
  });
}

/**
 * Per-pixel picture of a per-zone output: `target[pixel] = source[zone[pixel] * stride +
 * component[0]]`, NaN for a pixel in no zone.
 */
export function addZoneGather(
  builder: RunBuilder,
  id: string,
  source: RunBuffer,
  stride: number,
  zones: RunBuffer,
  component: RunBuffer,
  target: RunBuffer
): void {
  const zoneCount = source.length / stride;
  builder.graph.add({
    getCommandNodes: graph => [
      createWGSLKernelNode(graph, {
        id,
        operation: 'embedding-zone-gather',
        bindings: [
          {name: 'sourceValues', view: source.view, type: 'f32', access: 'read'},
          {name: 'zones', view: zones.view, type: 'u32', access: 'read'},
          {name: 'component', view: component.view, type: 'u32', access: 'read'},
          {name: 'targetValues', view: target.view, type: 'f32', access: 'read_write'}
        ],
        invocationCount: target.length,
        body: `let zone = zones[zonesOffset + index];
  let missing = bitcast<f32>(0x7fc00000u | (index & 0u));
  let safeZone = min(zone, ${zoneCount - 1}u);
  let value = sourceValues[sourceValuesOffset + safeZone * ${stride}u + component[componentOffset]];
  targetValues[targetValuesOffset + index] = select(missing, value, zone < ${zoneCount}u);`
      })
    ]
  });
}

/** Reverses the row order of a `columns`-wide float raster (row 0 smallest y to row 0 at the top). */
export function addRowFlip(
  builder: RunBuilder,
  id: string,
  source: RunBuffer,
  columns: number,
  target: RunBuffer
): void {
  const rows = target.length / columns;
  builder.graph.add({
    getCommandNodes: graph => [
      createWGSLKernelNode(graph, {
        id,
        operation: 'flow-row-flip',
        bindings: [
          {name: 'sourceValues', view: source.view, type: 'f32', access: 'read'},
          {name: 'targetValues', view: target.view, type: 'f32', access: 'read_write'}
        ],
        invocationCount: target.length,
        body: `let column = index % ${columns}u;
  let row = index / ${columns}u;
  targetValues[targetValuesOffset + index] = sourceValues[sourceValuesOffset + (${rows - 1}u - row) * ${columns}u + column];`
      })
    ]
  });
}
