// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import type {PointPatternEnvelopeOperationProps} from '../../../src/gpu-spatial-analysis/point-process/gpu-point-pattern-envelope-operation';
import type {OperationGraphRows} from '../../utils/mixed-execution-test-utils';

export function getBoundaryEnvelopeCase(below: boolean) {
  const props: PointPatternEnvelopeOperationProps = {
    curveRowCount: 1,
    simulationCount: 48,
    observed: 'observed',
    simulations: 'simulations',
    statistic: 'extreme-rank',
    significanceLevel: below ? 1 / 49 - 2 ** -58 : 1 / 49,
    nullModel: 'fixed-external',
    fittedParameters: 'none',
    output: {lower: 'lower', upper: 'upper', statistics: 'statistics', status: 'status'}
  };
  return {
    props,
    inputs: {
      observed: {format: 'float32' as const, values: Float32Array.of(0)},
      simulations: {
        format: 'float32' as const,
        values: Float32Array.from([...Array.from({length: 47}, (_, index) => index + 1), 47])
      }
    }
  };
}

export function expectBoundaryEnvelope(result: Record<string, OperationGraphRows>, below: boolean) {
  // Only 0 has extreme rank 1; the upper tie at 47 has midrank 1.5.
  // Removing 0 costs exactly 1/49. Below that level no curve can be removed.
  expect(Array.from(result['status'])).toEqual([0]);
  expect(Array.from(result['lower'])).toEqual([below ? 0 : 1]);
  expect(Array.from(result['upper'])).toEqual([47]);
}

export function getResidualEnvelopeCase(exponent: number) {
  const props: PointPatternEnvelopeOperationProps = {
    curveRowCount: 2,
    simulationCount: 19,
    observed: 'observed',
    simulations: 'simulations',
    reference: 'reference',
    integrationWeights: 'weights',
    statistic: 'integrated-squared-deviation',
    nullModel: 'fixed-external',
    fittedParameters: 'none',
    output: {lower: 'lower', upper: 'upper', statistics: 'statistics', status: 'status'}
  };
  return {
    props,
    inputs: {
      observed: {format: 'float32' as const, values: Float32Array.of(1, 2 ** exponent)},
      simulations: {
        format: 'float32' as const,
        values: Float32Array.from({length: 38}, (_, index) => (index % 2 === 0 ? 1 : 0))
      },
      reference: {format: 'float32' as const, values: Float32Array.of(0, 0)},
      weights: {format: 'float32' as const, values: Float32Array.of(1, 1)}
    }
  };
}

export function expectResidualEnvelope(
  result: Record<string, OperationGraphRows>,
  exponent: number
) {
  // Exact observed sum 1 + 2^(2*exponent) exceeds every simulated sum of 1.
  expect(Array.from(result['status'])).toEqual([0]);
  expect(Array.from(result['statistics'])).toEqual([
    Math.fround(1 + 2 ** (2 * exponent)),
    Math.fround(1 / 20),
    Math.fround(1 / 20),
    19
  ]);
}
