// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  applyGeneralizedAuditReviewCase,
  type createGeneralizedAuditFixture
} from './generalized-gwr-audit-fixtures';

export type FinalGeneralizedCase =
  | 'fractional-exposure'
  | 'shifted-fractional-exposure'
  | 'feasible-exposure'
  | 'shifted-feasible-exposure'
  | 'eligible-extrapolation';

export function applyFinalGeneralizedCase(
  fixture: ReturnType<typeof createGeneralizedAuditFixture>,
  scenario?: FinalGeneralizedCase
) {
  if (scenario === 'fractional-exposure' || scenario === 'shifted-fractional-exposure') {
    fixture.positions.fill(0);
    fixture.predictors.splice(0, 6, 0, 0, 0, 1, 1, 1);
    fixture.mask = fixture.mask.map((_, row) => (row < 6 ? 1 : 0));
    fixture.response = fixture.response.map((_, row) => (row < 6 ? Number(row > 0) : NaN));
    fixture.offset = fixture.offset.map(
      (_, row) =>
        ([0, 23, 23, 33, 33, 33][row] ?? 0) +
        (scenario === 'shifted-fractional-exposure' ? 2.004 : 0)
    );
  } else if (scenario === 'eligible-extrapolation') {
    applyGeneralizedAuditReviewCase(fixture, 'extrapolation');
    fixture.mask[12] = 1;
    fixture.response[12] = 1;
  } else if (scenario) {
    fixture.positions.fill(0);
    fixture.response.fill(1);
    fixture.response[0] = 0;
    fixture.offset = fixture.predictors.map(value => (value === 1 ? 10 : 0));
    fixture.offset[0] = -23;
    if (scenario === 'shifted-feasible-exposure')
      fixture.offset = fixture.offset.map(value => value - 4);
  }
}
