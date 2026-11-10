// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Focal and complement populations per unit for two periods, and the two counterfactuals between
// them, for the segregation runs.

import {getCensusPopulation} from './census-tracts';
import type {QuadZones} from './quads';

export type PopulationScenarios = {
  /** Baseline, composition-only, spatial-only and final populations, (focal, complement) per unit. */
  scenarios: [Float32Array, Float32Array, Float32Array, Float32Array];
  /** Focal share per unit in the baseline and final periods. */
  baselineShare: Float32Array;
  finalShare: Float32Array;
};

/**
 * `sorting` (0–1) amplifies the baseline spatial concentration of the focal group
 * in the modeled period. `growth` changes its share within each tract.
 */
export function getPopulationScenarios(
  zones: QuadZones,
  sorting: number,
  growth: number
): PopulationScenarios {
  const count = zones.count;
  const baseline = new Float32Array(count * 2);
  const final = new Float32Array(count * 2);
  const baselineShare = new Float32Array(count);
  const finalShare = new Float32Array(count);
  for (let unit = 0; unit < count; unit++) {
    const [focal, complement] = getCensusPopulation(unit);
    const total = focal + complement;
    const baseShare = focal / total;
    const finalFocal = Math.max(
      0.01,
      Math.min(0.99, baseShare + growth * 0.1 + sorting * (baseShare - 0.3) * 0.4)
    );
    baseline.set([focal, complement], unit * 2);
    final.set([total * finalFocal, total * (1 - finalFocal)], unit * 2);
    baselineShare[unit] = baseShare;
    finalShare[unit] = finalFocal;
  }
  const sum = (values: Float32Array, offset: number) => {
    let total = 0;
    for (let unit = 0; unit < count; unit++) total += values[unit * 2 + offset];
    return total;
  };
  const focalScale = sum(final, 0) / sum(baseline, 0);
  const compositionOnly = Float32Array.from(baseline, (value, index) =>
    index % 2 === 0 ? value * focalScale : value
  );
  const spatialScale = sum(baseline, 0) / sum(final, 0);
  const spatialOnly = Float32Array.from(final, (value, index) =>
    index % 2 === 0 ? value * spatialScale : value
  );
  return {scenarios: [baseline, compositionOnly, spatialOnly, final], baselineShare, finalShare};
}
