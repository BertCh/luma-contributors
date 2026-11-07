// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared inputs of the rate and areal runs: tract meshes uploaded for drawing, and seeded
// event and population counts per tract.

import type {RunBuilder} from '../../../define-run';
import type {Random, TractsInput} from '../../../inputs';

/** Uploads the tract mesh buffers every polygon layer needs. */
export function uploadTractMesh(builder: RunBuilder, tracts: TractsInput) {
  return {
    vertices: builder.upload('vertices', 'float32x2', tracts.vertices),
    indices: builder.upload('indices', 'uint32', tracts.triangleIndices),
    owners: builder.upload('owners', 'uint32', tracts.triangleOwners),
    centers: builder.upload('centers', 'float32x2', tracts.centers),
    outline: builder.upload('outline', 'float32x4', tracts.outline)
  };
}

export type RateSeeds = {
  /** Population position in [0, 1] per tract. */
  populationUnit: Float32Array;
  /** Standard normal draw per tract (the sampling noise of the event count). */
  gaussian: Float32Array;
};

export function createRateSeeds(random: Random, count: number): RateSeeds {
  return {
    populationUnit: Float32Array.from({length: count}, () => random()),
    gaussian: Float32Array.from({length: count}, () => {
      let sum = 0;
      for (let step = 0; step < 6; step++) sum += random();
      return (sum - 3) * Math.SQRT2;
    })
  };
}

/**
 * Event and population counts per tract. The underlying risk follows `field` (contrast scales its
 * spread); populations span a factor of 40, so small tracts have noisy raw rates.
 */
export function getRateCounts(
  seeds: RateSeeds,
  field: Float32Array,
  riskContrast: number,
  populationScale: number
): {events: Float32Array; populations: Float32Array; rates: Float32Array} {
  const populations = Float32Array.from(seeds.populationUnit, unit =>
    Math.round(populationScale * 100 * 40 ** unit)
  );
  const rates = Float32Array.from(field, value => 0.04 * (1 + riskContrast * (value - 0.5) * 2));
  const events = Float32Array.from(populations, (population, index) => {
    const expected = population * rates[index];
    return Math.max(0, Math.round(expected + Math.sqrt(expected) * seeds.gaussian[index]));
  });
  return {events, populations, rates};
}
