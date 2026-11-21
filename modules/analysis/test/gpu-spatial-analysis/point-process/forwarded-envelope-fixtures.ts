// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {PointPatternEnvelopeOperationProps} from '../../../src/gpu-spatial-analysis/point-process/gpu-point-pattern-envelope-operation';

// Fraction gives |−1−r| = 1+r and |1−r| = 1−r. At r=2^-60 there
// are 0 exceedances of 7, while at r=0 all 7 simulations tie.
export const MAXIMUM_DEVIATION_CASES = [
  {reference: 2 ** -60, probability: 0.125},
  {reference: 0, probability: 1}
];

export function getForwardedEnvelopeCase(integrated: boolean) {
  const simulationCount = integrated ? 3 : 7;
  const props: PointPatternEnvelopeOperationProps & {simulationCount: number} = {
    curveRowCount: integrated ? 2 : 1,
    simulationCount,
    observed: 'observed',
    simulations: 'simulations',
    reference: 'reference',
    ...(integrated ? {integrationWeights: 'weights'} : {}),
    statistic: integrated ? 'integrated-squared-deviation' : 'maximum-deviation',
    nullModel: 'fixed-external',
    fittedParameters: 'none',
    output: {lower: 'lower', upper: 'upper', statistics: 'statistics', status: 'status'}
  };
  return {
    props,
    inputs: {
      observed: {
        format: 'float32' as const,
        values: integrated ? Float32Array.of(1, 2 ** 63) : Float32Array.of(-1)
      },
      simulations: {
        format: 'float32' as const,
        values: integrated ? Float32Array.of(1, 0, 1, 0, 1, 0) : new Float32Array(7).fill(1)
      },
      reference: {
        format: 'float32' as const,
        values: integrated ? Float32Array.of(0, 0) : Float32Array.of(2 ** -26)
      },
      ...(integrated
        ? {weights: {format: 'float32' as const, values: Float32Array.of(1, 2 ** -149)}}
        : {})
    }
  };
}
