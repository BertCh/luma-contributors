// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KEPLER_RATE_CAUSES, type KeplerRateCause} from './rate-cause-counts';

export type RateInvalidCause = KeplerRateCause;

export type RateInvalidDiagnostics = {
  causes: Uint8Array;
  counts: Record<RateInvalidCause, number>;
};

/**
 * Attribute each excluded row to exactly one cause, in documented priority: masked, null event,
 * null population, non-finite event, non-finite population, negative event, negative population,
 * zero population. counts sum to the row count.
 */
export function classifyRateInvalidCauses(
  events: ArrayLike<number | null>,
  populations: ArrayLike<number | null>,
  mask?: ArrayLike<number | boolean>
): RateInvalidDiagnostics {
  const causes = new Uint8Array(events.length);
  const counts = Object.fromEntries(KEPLER_RATE_CAUSES.map(cause => [cause, 0])) as Record<
    RateInvalidCause,
    number
  >;
  for (let row = 0; row < events.length; row++) {
    const event = events[row];
    const population = populations[row];
    let cause: RateInvalidCause = 'valid';
    if (mask && !mask[row]) cause = 'masked';
    else if (event === null || event === undefined) cause = 'nullEvent';
    else if (population === null || population === undefined) cause = 'nullPopulation';
    else if (!Number.isFinite(event)) cause = 'nonFiniteEvent';
    else if (!Number.isFinite(population)) cause = 'nonFinitePopulation';
    else if (event < 0) cause = 'negativeEvent';
    else if (population < 0) cause = 'negativePopulation';
    else if (population === 0) cause = 'zeroPopulation';
    causes[row] = KEPLER_RATE_CAUSES.indexOf(cause);
    counts[cause]++;
  }
  return {causes, counts};
}
