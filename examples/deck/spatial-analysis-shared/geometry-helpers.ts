// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {DrawCommandBuffer} from '@luma.gl/graph/gpu-core';

/** Uniform bytes every storage-model layer owns (room for a palette and config). */
export const STORAGE_MODEL_STYLE_BYTE_LENGTH = 160;

/** Copies the GPU-written vertex count into word 1 (instance count) of an indirect record. */
export function copyCountToDrawRecord(
  commandEncoder: CommandEncoder,
  count: Buffer,
  drawCommands: DrawCommandBuffer
): void {
  commandEncoder.copyBufferToBuffer({
    sourceBuffer: count,
    sourceOffset: 0,
    destinationBuffer: drawCommands.buffer,
    destinationOffset: 4,
    size: 4
  });
}

/** Point bounds padded by a few percent so cells never clip the outermost points. */
export function getPaddedBounds(positions: Float32Array): [number, number, number, number] {
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (let index = 0; index < positions.length; index += 2) {
    minimumX = Math.min(minimumX, positions[index]);
    maximumX = Math.max(maximumX, positions[index]);
    minimumY = Math.min(minimumY, positions[index + 1]);
    maximumY = Math.max(maximumY, positions[index + 1]);
  }
  const padding = 0.02 * Math.max(maximumX - minimumX, maximumY - minimumY, 1);
  return [minimumX - padding, minimumY - padding, maximumX + padding, maximumY + padding];
}

export function writeShapeStyle(
  styleBuffer: Buffer,
  props: ShapeStyleProps,
  extra: {sizePixels?: number; segments?: number; shape?: number; mode?: number} = {}
): void {
  const data = new ArrayBuffer(STORAGE_MODEL_STYLE_BYTE_LENGTH);
  const floats = new Float32Array(data);
  const words = new Uint32Array(data);
  for (let index = 0; index < SHAPE_PALETTE_SIZE; index++) {
    const color = props.palette[index % props.palette.length];
    floats.set([color[0] / 255, color[1] / 255, color[2] / 255, color[3] / 255], index * 4);
  }
  floats.set([props.widthPixels ?? 1.5, props.opacity ?? 1, extra.sizePixels ?? 6, 0], 32);
  words.set([props.groupCount, extra.segments ?? 1, extra.shape ?? 0, extra.mode ?? 0], 36);
  styleBuffer.write(new Uint8Array(data));
}

/**
 * The raster grid of the viewport bounds: rows advance north from the bounds' minimum corner, so
 * row 0 is the southernmost row, as the map layers read it.
 */
export function getViewRasterGrid(
  bounds: readonly [number, number, number, number],
  rasterWidth: number,
  rasterHeight: number
) {
  const [minX, minY, maxX, maxY] = bounds;
  return makeGPURasterMetadata({
    width: rasterWidth,
    height: rasterHeight,
    cellSize: [(maxX - minX) / rasterWidth, (maxY - minY) / rasterHeight],
    origin: [minX, minY],
    rowDirection: 'north'
  });
}

export const SHAPE_PALETTE_SIZE = 8;

export type ShapeStyleProps = {
  /** Number of groups the shape buffers hold. */
  groupCount: number;
  /** Colors indexed by `group % 8`, RGBA 0-255. Markers use the first entry. */
  palette: readonly ShapeColor[];
  /** Line width in CSS pixels. Defaults to 1.5. */
  widthPixels?: number;
  /** Overall opacity multiplier. Defaults to 1. */
  opacity?: number;
};

export type ShapeColor = readonly [number, number, number, number];
