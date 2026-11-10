// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import packedReturns from './king-county.b64?raw';
import type {Random, World} from '../runs/inputs';

const RECORD_BYTES = 7;
let observations: DataView | undefined;

/** USGS 3DEP returns. XY fits the view uniformly, Z remains NAVD88 elevation in metres. */
export function createPointCloud(_random: Random, world: World, count = 6000) {
  observations ??= new DataView(
    Uint8Array.from(atob(packedReturns), byte => byte.charCodeAt(0)).buffer
  );
  const recordCount = observations.byteLength / RECORD_BYTES;
  const scale = Math.min(world.width / 128, world.height / 72);
  const left = (world.width - 128 * scale) / 2;
  const bottom = (world.height - 72 * scale) / 2;
  const positions = new Float32Array(count * 2);
  const heights = new Float32Array(count);
  const classes = new Uint32Array(count);
  for (let row = 0; row < count; row++) {
    const offset = (row % recordCount) * RECORD_BYTES;
    positions[row * 2] = left + observations.getUint16(offset, true) * 0.01 * scale;
    positions[row * 2 + 1] = bottom + observations.getUint16(offset + 2, true) * 0.01 * scale;
    heights[row] = observations.getUint16(offset + 4, true) * 0.01;
    classes[row] = observations.getUint8(offset + 6);
  }
  return {count, positions, heights, classes, scale};
}
