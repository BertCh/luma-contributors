// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import type {HydrologicIndicesOracleResult} from './terrain-hydrologic-indices-oracle';

export const HYDROLOGIC_INDEX_OUTPUTS = [
  'descentTangent',
  'specificCatchmentArea',
  'wetnessIndex',
  'streamPowerIndex'
] as const;
const OUTPUTS = HYDROLOGIC_INDEX_OUTPUTS;
/**
 * Relative bound for slope, catchment and stream power (a few f32 operations plus the GPU
 * `sqrt`, and `cos`/`cosh` of the ground cell width on curved grids) and absolute bound for the
 * wetness index (`log` of those).
 */
const RELATIVE_TOLERANCE = 2e-6;
const WETNESS_TOLERANCE = 4e-6;

export function expectIndicesClose(
  name: string,
  actual: Record<(typeof OUTPUTS)[number], Float32Array>,
  expected: HydrologicIndicesOracleResult
): void {
  for (const output of OUTPUTS) {
    expect(actual[output], `${name} ${output} row count`).toHaveLength(expected[output].length);
    for (const [cell, value] of expected[output].entries()) {
      const label = `${name} ${output} ${cell}`;
      if (!Number.isFinite(value)) {
        expect(actual[output][cell], label).toBe(value);
        continue;
      }
      expect(Number.isFinite(actual[output][cell]), label).toBe(true);
      const tolerance =
        output === 'wetnessIndex' ? WETNESS_TOLERANCE : RELATIVE_TOLERANCE * Math.abs(value);
      expect(Math.abs(actual[output][cell] - value), label).toBeLessThanOrEqual(tolerance);
    }
  }
}
