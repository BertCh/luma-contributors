// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeAssuncaoReisOracle} from './gpu-rate-standardization-fixture';
import {
  RATE_AUDIT_REFERENCE,
  RATE_AUDIT_MASK,
  RATE_AUDIT_POPULATIONS
} from './rate-standardization-audit-fixture';

it('Assuncao-Reis CPU twin matches exact rational moments and row-specific negative variance fallback', () => {
  const maximumDifferences = {rates: 0, statistics: 0};
  for (const reference of RATE_AUDIT_REFERENCE) {
    // The twin has no mask argument; dropping masked rows as NaN gives the same selected population.
    const events = reference.events.map((value, row) =>
      RATE_AUDIT_MASK[row] ? reference.scale * value : NaN
    );
    const populations = RATE_AUDIT_POPULATIONS.map(value => reference.scale * value);
    for (const reversed of [false, true]) {
      const oracle = computeAssuncaoReisOracle(
        reversed ? events.slice().reverse() : events,
        reversed ? populations.slice().reverse() : populations
      );
      expect(oracle.negativeVarianceRows).toBe(reference.negativeVarianceRows);
      for (const key of ['rates', 'statistics'] as const) {
        const expected =
          key === 'rates'
            ? reversed
              ? reference.rates.slice().reverse()
              : reference.rates
            : reference.statistics;
        const actual = key === 'rates' ? oracle.z : oracle.statistics;
        expect(actual.length).toBe(expected.length);
        for (const [index, value] of expected.entries()) {
          if (Number.isNaN(value)) expect(actual[index]).toBeNaN();
          else {
            expect(Number.isFinite(actual[index])).toBe(true);
            const difference = Math.abs(actual[index] - value);
            maximumDifferences[key] = Math.max(maximumDifferences[key], difference);
            expect(difference, `${reference.name} ${key}[${index}]`).toBeLessThanOrEqual(1e-12);
          }
        }
      }
    }
  }
  console.info('rational Assuncao-Reis maximum absolute differences', maximumDifferences);
});
