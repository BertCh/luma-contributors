// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared pieces of the tract-based recipe runs: mesh upload, islands, and count generators.

import type {RunBuffer, RunBuilder, PolygonsRunLayer} from '../../../define-run';
import {buildRows, type CsrWeights, type TractsInput} from '../../../inputs';

export type TractMesh = {
  vertices: RunBuffer<'float32x2'>;
  indices: RunBuffer<'uint32'>;
  owners: RunBuffer<'uint32'>;
  centers: RunBuffer<'float32x2'>;
  outline: RunBuffer<'float32x4'>;
};

/** Uploads the drawing mesh of the tracts. */
export function uploadTractMesh(builder: RunBuilder, tracts: TractsInput): TractMesh {
  return {
    vertices: builder.upload('vertices', 'float32x2', tracts.vertices),
    indices: builder.upload('indices', 'uint32', tracts.triangleIndices),
    owners: builder.upload('owners', 'uint32', tracts.triangleOwners),
    centers: builder.upload('centers', 'float32x2', tracts.centers),
    outline: builder.upload('outline', 'float32x4', tracts.outline)
  };
}

/** The polygon layer filling tracts by `values`. */
export function getTractFill(
  mesh: TractMesh,
  tracts: TractsInput,
  values: RunBuffer,
  look: Partial<PolygonsRunLayer> = {}
): PolygonsRunLayer {
  return {
    layer: 'polygons',
    positions: mesh.vertices,
    indices: mesh.indices,
    owners: mesh.owners,
    values,
    triangleCount: tracts.triangleCount,
    ...look
  };
}

/** Rook contiguity where the first `islandCount` tracts (in a seeded order) have no neighbors. */
export function isolateTracts(
  contiguity: CsrWeights,
  order: readonly number[],
  islandCount: number
): CsrWeights {
  const islands = new Set(order.slice(0, islandCount));
  return buildRows(contiguity.rowCount, row => {
    if (islands.has(row)) return [];
    const found: [number, number][] = [];
    for (let slot = contiguity.offsets[row]; slot < contiguity.offsets[row + 1]; slot++) {
      const neighbor = contiguity.neighbors[slot];
      if (!islands.has(neighbor)) found.push([neighbor, contiguity.distances[slot]]);
    }
    return found;
  });
}

/** A seeded shuffle of 0..count-1. */
export function createShuffledOrder(random: () => number, count: number): number[] {
  const order = Array.from({length: count}, (_, index) => index);
  for (let index = count - 1; index > 0; index--) {
    const swap = Math.floor(random() * (index + 1));
    [order[index], order[swap]] = [order[swap], order[index]];
  }
  return order;
}

export type RateDraws = {small: Float32Array; size: Float32Array; noise: Float32Array};

/** Seeded draws behind the rate counts, so sliders change the data without reshuffling it. */
export function createRateDraws(random: () => number, count: number): RateDraws {
  const draw = () => Float32Array.from({length: count}, random);
  return {
    small: draw(),
    size: draw(),
    noise: Float32Array.from({length: count}, () => random() + random() + random() - 1.5)
  };
}

/** Populations and event counts whose underlying risk follows the smooth tract values. */
export function getRateCounts(
  tracts: {count: number; values: Float32Array},
  draws: RateDraws,
  contrast: number,
  smallShare: number
): {populations: Float32Array; events: Uint32Array; rates: Float32Array} {
  const populations = new Float32Array(tracts.count);
  const events = new Uint32Array(tracts.count);
  const rates = new Float32Array(tracts.count);
  for (let tract = 0; tract < tracts.count; tract++) {
    const isSmall = draws.small[tract] < smallShare;
    const population = isSmall ? 30 + draws.size[tract] * 80 : 600 + draws.size[tract] * 2400;
    const risk = 0.02 + 0.05 * contrast * tracts.values[tract] ** 2;
    const expected = population * risk;
    const count = Math.max(0, Math.round(expected + draws.noise[tract] * 2 * Math.sqrt(expected)));
    populations[tract] = population;
    events[tract] = count;
    rates[tract] = count / population;
  }
  return {populations, events, rates};
}
