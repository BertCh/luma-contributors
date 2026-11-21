// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {isCanonicalDasymetricStageMetric} from '../../src/curated/bindings/curated-dasymetric-redistribution-binding';

test('dasymetric evidence publishes completeness, malformed tuples and zero support', () => {
  const validation = {complete: 1, invalidCount: 0};
  const conservation = {complete: 1, zeroEligibleCount: 1};
  const selected = (stage: string, status: Record<string, number>) =>
    Object.fromEntries(
      Object.entries(status).filter(([metric]) => isCanonicalDasymetricStageMetric(stage, metric))
    );
  expect(selected('coverage-validation', validation)).toEqual({complete: 1, invalidCount: 0});
  expect(selected('dasymetric-conservation', conservation)).toEqual({
    complete: 1,
    zeroEligibleCount: 1
  });
  expect(selected('source-totals', {invalidCount: 1})).toEqual({invalidCount: 1});
});
