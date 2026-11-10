// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import packedFlashes from './midwest-flashes.b64?raw';
import type {Random, World, TimeSeriesInput} from '../runs/inputs';

const RECORD_BYTES = 10;
let observations: DataView | undefined;

/** Read original GOES flash records from the fixed Midwest sample. See series-sources.md. */
export function sampleFlashes(world: World, count: number, start = 0) {
  observations ??= new DataView(
    Uint8Array.from(atob(packedFlashes), byte => byte.charCodeAt(0)).buffer
  );
  const recordCount = observations.byteLength / RECORD_BYTES;
  const positions = new Float32Array(count * 2);
  const hours = new Float32Array(count);
  const energy = new Float32Array(count);
  const area = new Float32Array(count);
  for (let row = 0; row < count; row++) {
    const offset = ((row + start) % recordCount) * RECORD_BYTES;
    // X and Y use the bounds of the same 16° × 10° window for every extract.
    positions[row * 2] = (observations.getUint16(offset, true) / 16000) * world.width;
    positions[row * 2 + 1] = (observations.getUint16(offset + 2, true) / 10000) * world.height;
    hours[row] = observations.getUint16(offset + 4, true) / 3600;
    energy[row] = observations.getUint16(offset + 6, true);
    area[row] = observations.getUint16(offset + 8, true);
  }
  return {count, positions, hours, energy, area};
}

/** Counts of observed flashes in longitude bands and equal time bins over the 18-hour window. */
export function createTimeSeries(
  _random: Random,
  panelCount = 32,
  stepCount = 96
): TimeSeriesInput {
  const flashes = sampleFlashes({width: 1, height: 1}, 4096);
  const values = new Float32Array(panelCount * stepCount);
  for (let row = 0; row < flashes.count; row++) {
    const panel = Math.min(panelCount - 1, Math.floor(flashes.positions[row * 2] * panelCount));
    const step = Math.min(stepCount - 1, Math.floor((flashes.hours[row] / 18) * stepCount));
    values[panel * stepCount + step]++;
  }
  return {panelCount, stepCount, values};
}
