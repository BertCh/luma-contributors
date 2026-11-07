// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {INTERPOLATION_COVERAGE_REFERENCE as reference} from './interpolation-coverage-reference';
import {calculateConditionalSimulationReference} from './conditional-simulation-reference';

it('Schur-complement simulation oracle matches independent joint conditioning fixture', () => {
  const actual = calculateConditionalSimulationReference(
    reference.positions,
    reference.values.flatMap((value, row) => [value, reference.errors[row], 0, 1]),
    reference.queries.slice(0, 4),
    reference.covariance,
    1.2
  );
  actual.means.forEach((value, row) =>
    expect(value).toBeCloseTo(reference.simulation.mean[row], 10)
  );
  actual.covariance.forEach((values, row) =>
    values.forEach((value, column) =>
      expect(value).toBeCloseTo(reference.simulation.covariance[row][column], 10)
    )
  );
});
