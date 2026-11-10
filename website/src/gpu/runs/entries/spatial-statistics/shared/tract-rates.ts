// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared inputs of the rate and areal runs: tract meshes uploaded for drawing, and observed
// counts of residents age 65+ and total population per tract (with simulated sampling noise).

import type {RunBuilder} from '../../../define-run';
import type {TractsInput} from '../../../inputs';
import type {RunGlyph, RunPointer} from '../../../interaction';
import {findPolygonAt} from '../../../pick';

/** Uploads the tract mesh buffers every polygon layer needs. */
export function uploadTractMesh(builder: RunBuilder, tracts: TractsInput) {
  return {
    vertices: builder.upload('vertices', 'float32x2', tracts.vertices),
    meshIndices: builder.upload('indices', 'uint32', tracts.triangleIndices),
    owners: builder.upload('owners', 'uint32', tracts.triangleOwners),
    centers: builder.upload('centers', 'float32x2', tracts.centers),
    outline: builder.upload('outline', 'float32x4', tracts.outline)
  };
}

/**
 * At unit scale and full contrast these are the Census estimates plus sampling noise: the counts of
 * a tract of `populationScale * population` residents scatter around its rate by the binomial
 * standard deviation, so small populations give noisy raw rates that the smoothers can shrink.
 */
export function getRateCounts(tracts: TractsInput, contrast: number, populationScale: number) {
  const totalOlder = tracts.age65.reduce((sum, value) => sum + value, 0);
  const totalPopulation = tracts.population.reduce((sum, value) => sum + value, 0);
  const average = totalOlder / totalPopulation;
  const populations = Float32Array.from(tracts.population, population =>
    Math.max(1, Math.round(populationScale * population))
  );
  const events = Float32Array.from(populations, (population, index) => {
    const rate = tracts.age65[index] / tracts.population[index];
    const expected = population * (average + (rate - average) * contrast);
    const deviation = Math.sqrt(expected * (1 - expected / population));
    return Math.min(population, Math.max(0, Math.round(expected + deviation * getNormal(index))));
  });
  return {events, populations};
}

/** A fixed pseudo-normal value per tract (sum of three uniform hashes), so the noise does not flicker. */
function getNormal(index: number): number {
  let total = 0;
  for (let draw = 0; draw < 3; draw++) {
    const hash = Math.sin((index * 3 + draw + 1) * 12.9898) * 43758.5453;
    total += hash - Math.floor(hash) - 0.5;
  }
  return total * 2;
}

/** The tract under the pointer; keeps `previous` while the pointer is between tracts. */
export function pickTract(tracts: TractsInput, pointer: RunPointer, previous: number): number {
  const row = findPolygonAt(
    tracts.vertices,
    tracts.triangleIndices,
    tracts.triangleOwners,
    pointer.x,
    pointer.y,
    tracts.triangleCount
  );
  return row >= 0 ? row : previous;
}

/** The outline of one tract as a closed polyline. */
export function createTractOutline(tracts: TractsInput, row: number): RunGlyph {
  return {
    shape: 'polyline',
    points: tracts.vertices.subarray(tracts.ringOffsets[row] * 2, tracts.ringOffsets[row + 1] * 2),
    closed: true
  };
}

/** Rook neighbors of `row` (a view of the contiguity, not a copy). */
export function getTractNeighbors(tracts: TractsInput, row: number): Uint32Array {
  return tracts.contiguity.neighbors.subarray(
    tracts.contiguity.offsets[row],
    tracts.contiguity.offsets[row + 1]
  );
}
