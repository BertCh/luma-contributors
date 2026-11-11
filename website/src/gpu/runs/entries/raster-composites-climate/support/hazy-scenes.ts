// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Scenario for the multi-band composites: eight acquisitions of the same ground, built from the six
// Sentinel-2 L2A reflectance bands of the Feather River watershed (July 13, 2021). The ground is
// real; the eight scenes are a synthetic stand-in, because the checked-in imagery has one date with
// all six bands. Each scene gets a seeded smooth field: where it is high the scene is thick cloud
// (bright in every band, and flagged by the QA mask), and below that threshold it carries thin haze
// that the QA mask misses (added reflectance, strongest in the visible bands).

import {sampleReflectance} from '../../../../data/reflectance';
import {createSmoothField, type Random} from '../../../inputs';

export const BAND_NAMES = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'] as const;
export const BAND_COUNT = BAND_NAMES.length;
export const SCENE_COUNT = 8;
const HAZE_SPECTRUM = [0.34, 0.3, 0.26, 0.12, 0.05, 0.03] as const;
const CLOUD_SPECTRUM = [0.62, 0.64, 0.66, 0.68, 0.5, 0.42] as const;
const CLOUD_THRESHOLD = 0.74;
const HAZE_START = 0.45;

export type HazyScenes = {
  columns: number;
  rows: number;
  cellCount: number;
  /** Reflectance of the clear ground, `cell * bandCount + band`. */
  ground: Float32Array;
  /** Smooth field per scene, `cell * SCENE_COUNT + scene`, in [0, 1]. */
  fields: Float32Array;
  /** Per-observation noise, `(cell * SCENE_COUNT + scene) * bandCount + band`. */
  noise: Float32Array;
};

export function createHazyScenes(random: Random, aspect: number, rows: number): HazyScenes {
  const columns = Math.max(8, Math.round(rows * aspect));
  const cellCount = columns * rows;
  const bands = sampleReflectance(columns, rows);
  const ground = new Float32Array(cellCount * BAND_COUNT);
  const names = [
    'blue',
    'green',
    'red',
    'nearInfrared',
    'shortwaveInfrared',
    'shortwaveInfrared2'
  ] as const;
  names.forEach((name, band) => {
    for (let cell = 0; cell < cellCount; cell++)
      ground[cell * BAND_COUNT + band] = bands[name][cell];
  });
  const fields = new Float32Array(cellCount * SCENE_COUNT);
  for (let scene = 0; scene < SCENE_COUNT; scene++) {
    const field = createSmoothField(random, 3, 3);
    for (let row = 0; row < rows; row++) {
      for (let column = 0; column < columns; column++) {
        fields[(row * columns + column) * SCENE_COUNT + scene] = field(
          (column + 0.5) / columns,
          (row + 0.5) / rows
        );
      }
    }
  }
  const noise = Float32Array.from(
    {length: cellCount * SCENE_COUNT * BAND_COUNT},
    () => random() - 0.5
  );
  return {columns, rows, cellCount, ground, fields, noise};
}

export type ComposedScenes = {
  /** `(cell * sliceCount + slice) * bandCount + band`. */
  values: Float32Array;
  /** QA flag per observation: 0 for thick cloud, 1 otherwise (thin haze is not flagged). */
  validity: Uint32Array;
};

/** The first `sliceCount` scenes with the haze amplitude (0 to 1) applied. */
export function composeHazyScenes(
  scenes: HazyScenes,
  sliceCount: number,
  haze: number
): ComposedScenes {
  const values = new Float32Array(scenes.cellCount * sliceCount * BAND_COUNT);
  const validity = new Uint32Array(scenes.cellCount * sliceCount);
  for (let cell = 0; cell < scenes.cellCount; cell++) {
    for (let slice = 0; slice < sliceCount; slice++) {
      const field = scenes.fields[cell * SCENE_COUNT + slice];
      const thick = field > CLOUD_THRESHOLD;
      const thin = thick ? 0 : Math.max(0, (field - HAZE_START) / (CLOUD_THRESHOLD - HAZE_START));
      validity[cell * sliceCount + slice] = thick ? 0 : 1;
      for (let band = 0; band < BAND_COUNT; band++) {
        const jitter = 0.012 * scenes.noise[(cell * SCENE_COUNT + slice) * BAND_COUNT + band];
        values[(cell * sliceCount + slice) * BAND_COUNT + band] =
          (thick ? CLOUD_SPECTRUM[band] : scenes.ground[cell * BAND_COUNT + band]) +
          thin * haze * HAZE_SPECTRUM[band] +
          jitter;
      }
    }
  }
  return {values, validity};
}

/**
 * Chart rows for one cell's observations, y values only (x is the band number). `all` holds every
 * observation; `accepted` holds the QA-accepted ones, with a rejected observation repeating the
 * first accepted value of its band, so drawn over `all` it hides nothing but the accepted dots.
 */
export function getSpectraPoints(composed: ComposedScenes, cell: number, sliceCount: number) {
  const all = new Float32Array(sliceCount * BAND_COUNT);
  const accepted = new Float32Array(sliceCount * BAND_COUNT);
  let firstAccepted = -1;
  for (let slice = 0; slice < sliceCount; slice++) {
    if (composed.validity[cell * sliceCount + slice] === 1) {
      firstAccepted = slice;
      break;
    }
  }
  for (let slice = 0; slice < sliceCount; slice++) {
    const isAccepted = composed.validity[cell * sliceCount + slice] === 1;
    for (let band = 0; band < BAND_COUNT; band++) {
      const own = composed.values[(cell * sliceCount + slice) * BAND_COUNT + band];
      all[slice * BAND_COUNT + band] = own;
      accepted[slice * BAND_COUNT + band] =
        isAccepted || firstAccepted < 0
          ? own
          : composed.values[(cell * sliceCount + firstAccepted) * BAND_COUNT + band];
    }
  }
  return {all, accepted};
}

/** The per-band mean of the observations the QA mask accepted, the naive composite. */
export function getValidMean(
  composed: ComposedScenes,
  cell: number,
  sliceCount: number
): Float32Array {
  const mean = new Float32Array(BAND_COUNT);
  let count = 0;
  for (let slice = 0; slice < sliceCount; slice++) {
    if (!composed.validity[cell * sliceCount + slice]) continue;
    count++;
    for (let band = 0; band < BAND_COUNT; band++) {
      mean[band] += composed.values[(cell * sliceCount + slice) * BAND_COUNT + band];
    }
  }
  return mean.map(value => (count > 0 ? value / count : Number.NaN));
}

/**
 * A cell near the middle of the map where one or two scenes are thick cloud and the others carry
 * the most haze: the cell where a mean and a robust composite disagree most, used as the poster.
 */
export function findShowcaseCell(
  scenes: HazyScenes,
  sliceCount: number
): {column: number; row: number} {
  let best = {column: Math.floor(scenes.columns / 2), row: Math.floor(scenes.rows / 2)};
  let bestScore = -1;
  for (let row = Math.floor(scenes.rows * 0.25); row < Math.floor(scenes.rows * 0.75); row++) {
    for (
      let column = Math.floor(scenes.columns * 0.25);
      column < Math.floor(scenes.columns * 0.75);
      column++
    ) {
      const cell = row * scenes.columns + column;
      let thick = 0;
      let thin = 0;
      for (let slice = 0; slice < sliceCount; slice++) {
        const field = scenes.fields[cell * SCENE_COUNT + slice];
        if (field > CLOUD_THRESHOLD) thick++;
        else thin += Math.max(0, field - HAZE_START);
      }
      const score = thick >= 1 && thick <= 2 ? thin : -1;
      if (score > bestScore) {
        bestScore = score;
        best = {column, row};
      }
    }
  }
  return best;
}
