// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared inputs of the space-time event runs: the 2018 Kīlauea earthquakes (epicenter and days
// since May 1) fitted into the map part of the world with one isotropic scale, so ring radii drawn
// in world units are true kilometers; GOES lightning flashes with their hour; and CSR helpers.

import kilauea from '../../../../data/kilauea-events.json';
import {sampleFlashes} from '../../../../data/lightning';
import turkiye from '../../../../data/recipes/turkiye-quakes.json';
import type {RunBuilder} from '../../../define-run';
import type {CsrWeights} from '../../../inputs';
import {createPointIndex, type PointIndex} from '../../../pick';

export type World = {width: number; height: number};

export type KilaueaEvents = {
  count: number;
  /** World positions, isotropic: what the operations read and the layers draw. */
  positions: Float32Array;
  /** Days since 2018-05-01 00:00 UTC. */
  days: Float32Array;
  /** Kilometers per world unit. */
  kilometersPerUnit: number;
  index: PointIndex;
  focus: {row: number};
};

/** The summit caldera window, [west, south, east, north] degrees: 307 of the 350 events. */
export const SUMMIT_WINDOW = [-155.32, 19.38, -155.235, 19.432] as const;

/**
 * M3.5+ HVO epicenters of the 2018 Kīlauea eruption, May 1 to August 29, inside the summit window
 * (the collapse swarm; the scattered rift events are left out). Positions are local east/north
 * kilometers scaled into `frame` ([left, bottom, right, top] world units) less a margin.
 */
export function createKilaueaEvents(
  frame: readonly [number, number, number, number],
  margin = 0.04
): KilaueaEvents {
  const [west, south, eastEdge, northEdge] = SUMMIT_WINDOW;
  const rows = (kilauea as number[][]).filter(
    row => row[0] >= west && row[0] <= eastEdge && row[1] >= south && row[1] <= northEdge
  );
  const count = rows.length;
  const meanLatitude = rows.reduce((sum, row) => sum + row[1], 0) / count;
  const kilometersPerDegree = 111.32;
  const east = rows.map(
    row => row[0] * Math.cos((meanLatitude * Math.PI) / 180) * kilometersPerDegree
  );
  const north = rows.map(row => row[1] * kilometersPerDegree);
  const [minimumEast, maximumEast] = [Math.min(...east), Math.max(...east)];
  const [minimumNorth, maximumNorth] = [Math.min(...north), Math.max(...north)];
  const [left, bottom, right, top] = frame;
  const width = right - left - 2 * margin;
  const height = top - bottom - 2 * margin;
  const scale = Math.min(
    width / (maximumEast - minimumEast),
    height / (maximumNorth - minimumNorth)
  );
  const offsetX = left + margin + (width - (maximumEast - minimumEast) * scale) / 2;
  const offsetY = bottom + margin + (height - (maximumNorth - minimumNorth) * scale) / 2;
  const positions = new Float32Array(count * 2);
  for (let row = 0; row < count; row++) {
    positions[row * 2] = offsetX + (east[row] - minimumEast) * scale;
    positions[row * 2 + 1] = offsetY + (north[row] - minimumNorth) * scale;
  }
  return {
    count,
    positions,
    days: Float32Array.from(rows, row => row[2]),
    kilometersPerUnit: 1 / scale,
    index: createPointIndex(positions),
    focus: {row: -1}
  };
}

/** Times with the given share kept and the rest exchanged in a fixed cycle (a `Data:` control). */
export function shuffleTimes(times: Float32Array, observedShare: number): Float32Array {
  const shuffled = times.slice();
  const count = times.length;
  const shuffleCount = Math.round(count * (1 - observedShare));
  // A fixed stride visits rows spread over the catalogue, so the shuffled rows are not one burst.
  const order = Array.from({length: count}, (_, index) => (index * 151) % count);
  for (let index = 0; index < shuffleCount; index++) {
    shuffled[order[index]] = times[order[(index + 1) % shuffleCount]];
  }
  return shuffled;
}

/** A CSR the operation writes, with the port `GPUNeighborSearch` takes as `output`. */
export function allocateNeighborWeights(
  builder: RunBuilder,
  id: string,
  rowCount: number,
  capacity: number
) {
  const offsets = builder.allocate(`${id}-offsets`, 'uint32', rowCount + 1);
  const neighbors = builder.allocate(`${id}-neighbors`, 'uint32', capacity);
  const weights = builder.allocate(`${id}-weights`, 'float32', capacity);
  const distances = builder.allocate(`${id}-distances`, 'float32', capacity);
  const overflow = builder.allocate(`${id}-overflow`, 'uint32', 1);
  return {
    offsets,
    neighbors,
    distances,
    port: {
      offsets: offsets.view,
      neighbors: neighbors.view,
      weights: weights.view,
      distances: distances.view,
      status: {overflow: overflow.view}
    },
    input: {
      offsets: offsets.view,
      neighbors: neighbors.view,
      weights: weights.view,
      distances: distances.view
    }
  };
}

/** Row lists of a CSR on the CPU. */
export function getRow(weights: CsrWeights, row: number): Uint32Array {
  return weights.neighbors.subarray(weights.offsets[row], weights.offsets[row + 1]);
}

/** Every row within `radius` of `row` (brute force over a few hundred events). */
export function findWithin(positions: Float32Array, row: number, radius: number): number[] {
  const found: number[] = [];
  const x = positions[row * 2];
  const y = positions[row * 2 + 1];
  for (let other = 0; other < positions.length / 2; other++) {
    if (other === row) continue;
    if (Math.hypot(positions[other * 2] - x, positions[other * 2 + 1] - y) <= radius) {
      found.push(other);
    }
  }
  return found;
}

/** Formats a number that may still be unread. */
export function formatValue(value: number | undefined, digits = 2): string {
  return value === undefined || !Number.isFinite(value) ? '…' : value.toFixed(digits);
}

export type LightningFlashes = {
  count: number;
  /** World positions, isotropic kilometers scaled into the frame. */
  positions: Float32Array;
  /** Hours since 2024-05-21 12:00 UTC, 0 to 18. */
  hours: Float32Array;
  energy: Float32Array;
  area: Float32Array;
  kilometersPerUnit: number;
  index: PointIndex;
  focus: {row: number};
};

/**
 * GOES GLM flashes over 99°–83° W, 36°–46° N with one isotropic scale (local kilometers at 41° N)
 * fitted into `frame` ([left, bottom, right, top] world units).
 */
export function createLightningFlashes(
  frame: readonly [number, number, number, number],
  count: number,
  margin = 0.03
): LightningFlashes {
  const flashes = sampleFlashes({width: 1, height: 1}, count);
  const kilometersPerDegree = 111.32;
  const widthKilometers = 16 * Math.cos((41 * Math.PI) / 180) * kilometersPerDegree;
  const heightKilometers = 10 * kilometersPerDegree;
  const [left, bottom, right, top] = frame;
  const width = right - left - 2 * margin;
  const height = top - bottom - 2 * margin;
  const scale = Math.min(width / widthKilometers, height / heightKilometers);
  const offsetX = left + margin + (width - widthKilometers * scale) / 2;
  const offsetY = bottom + margin + (height - heightKilometers * scale) / 2;
  const positions = new Float32Array(count * 2);
  for (let row = 0; row < count; row++) {
    positions[row * 2] = offsetX + flashes.positions[row * 2] * widthKilometers * scale;
    positions[row * 2 + 1] = offsetY + flashes.positions[row * 2 + 1] * heightKilometers * scale;
  }
  return {
    count,
    positions,
    hours: flashes.hours,
    energy: flashes.energy,
    area: flashes.area,
    kilometersPerUnit: 1 / scale,
    index: createPointIndex(positions),
    focus: {row: -1}
  };
}

/**
 * USGS ComCat M4+ earthquakes of Türkiye and Syria, January 2023 to January 2024 (621 events, the
 * 6 February M7.8 mainshock on day 24), as local kilometers fitted into `frame`. Times are days
 * since the first event.
 */
export function createTurkiyeEvents(
  frame: readonly [number, number, number, number],
  margin = 0.03
) {
  const rows = turkiye as number[][];
  const count = rows.length;
  const meanLatitude = rows.reduce((sum, row) => sum + row[1], 0) / count;
  const kilometersPerDegree = 111.32;
  const east = rows.map(
    row => row[0] * Math.cos((meanLatitude * Math.PI) / 180) * kilometersPerDegree
  );
  const north = rows.map(row => row[1] * kilometersPerDegree);
  const [minimumEast, maximumEast] = [Math.min(...east), Math.max(...east)];
  const [minimumNorth, maximumNorth] = [Math.min(...north), Math.max(...north)];
  const [left, bottom, right, top] = frame;
  const width = right - left - 2 * margin;
  const height = top - bottom - 2 * margin;
  const scale = Math.min(
    width / (maximumEast - minimumEast),
    height / (maximumNorth - minimumNorth)
  );
  const offsetX = left + margin + (width - (maximumEast - minimumEast) * scale) / 2;
  const offsetY = bottom + margin + (height - (maximumNorth - minimumNorth) * scale) / 2;
  const positions = new Float32Array(count * 2);
  for (let row = 0; row < count; row++) {
    positions[row * 2] = offsetX + (east[row] - minimumEast) * scale;
    positions[row * 2 + 1] = offsetY + (north[row] - minimumNorth) * scale;
  }
  const origin = Math.min(...rows.map(row => row[2]));
  return {
    count,
    positions,
    days: Float32Array.from(rows, row => (row[2] - origin) / 86400),
    magnitudes: Float32Array.from(rows, row => row[3]),
    windowArea: (maximumEast - minimumEast) * (maximumNorth - minimumNorth) * scale * scale,
    kilometersPerUnit: 1 / scale,
    index: createPointIndex(positions),
    focus: {row: -1}
  };
}
