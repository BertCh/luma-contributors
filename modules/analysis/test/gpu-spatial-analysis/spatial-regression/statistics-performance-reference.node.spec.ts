// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import reference from '../../oracles/fixtures/statistics-coverage.json';
import {calculateSURBenchmarkReference} from './statistics-performance-reference';
import {calculateSURReferenceWithQR} from './sur-qr-reference';

it('SUR benchmark oracle matches the independently generated joint GLS fixture', () => {
  const actual = calculateSURBenchmarkReference(reference.sur.predictors, reference.sur.responses);
  for (const key of ['coefficients', 'covariance', 'residualCovariance', 'residuals'] as const) {
    expect(actual[key]).toHaveLength(reference.sur[key].length);
    actual[key].forEach((value, index) => expect(value).toBeCloseTo(reference.sur[key][index], 8));
  }
});

it('whitened QR agrees with the retained joint GLS oracle without normal equations', () => {
  const actual = calculateSURReferenceWithQR(reference.sur.predictors, reference.sur.responses);
  for (const key of ['coefficients', 'covariance', 'residualCovariance', 'residuals'] as const) {
    actual[key].forEach((value, index) => expect(value).toBeCloseTo(reference.sur[key][index], 8));
  }
});
