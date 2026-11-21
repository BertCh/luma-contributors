// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUInequalityOperation} from '../../../src/dataframe-analysis/composite-indicators/gpu-inequality-operation';
import {
  getGPUInequalityParameterValues,
  GPU_INEQUALITY_GLOBAL_SUMMARY,
  GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH
} from '../../../src/dataframe-analysis/composite-indicators/inequality-parameters';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

// Independent pair-difference expectations pin zone and pooled publication in both modes.
it.each([
  'parallel',
  'reference'
] as const)('NUM-15 %s preserves small weighted Gini and exact equality', async mode => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const fixture of [
    {values: [0, 1], weights: [1, 2 ** 58], expected: 3.469446951953614e-18},
    {values: [1, 2], weights: [2 ** 58, 1], expected: 3.469446951953614e-18},
    {values: [1, 1 + 2 ** -23], weights: [1, 1], expected: 2.9802320611338473e-8},
    {
      values: [2 ** -110, 2 ** -110 * (1 + 2 ** -23)],
      weights: [1, 1],
      expected: 2.9802320611338473e-8
    },
    {values: [0, 2 ** -80], weights: [2 ** -80, 2 ** -80], expected: 0.5},
    {values: [0, 0], weights: [2 ** -80, 2 ** -80], expected: NaN},
    {
      values: [...new Array<number>(256).fill(1), 1 + 2 ** -23],
      weights: new Array<number>(257).fill(1),
      expected: 4.620445137604179e-10
    },
    {values: new Array<number>(12).fill(1), weights: new Array<number>(12).fill(1), expected: 0}
  ]) {
    const rows = await runOperationGraph(device, {
      inputs: {
        values: {format: 'float32', values: Float32Array.from(fixture.values)},
        weights: {format: 'float32', values: Float32Array.from(fixture.weights)},
        zones: {format: 'uint32', values: new Uint32Array(fixture.values.length)},
        parameters: {format: 'float32', values: getGPUInequalityParameterValues()}
      },
      operations: [
        new GPUInequalityOperation({
          mode,
          values: 'values',
          weights: 'weights',
          zoneIds: 'zones',
          zoneCount: 1,
          parameters: 'parameters',
          output: {gini: 'gini', globalSummary: 'summary'}
        })
      ],
      outputs: ['gini', 'summary']
    });
    expect(rows['gini']).toHaveLength(1);
    expect(rows['summary']).toHaveLength(GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH);
    // w0*w1*|x0-x1| / ((w0+w1)*(w0*x0+w1*x1)) rounds to 2^-58
    // for the extreme weights. Unit weights give (2^-23)/(4 + 2^-22).
    // 257 rows cross the prefix tile boundary: 256*2^-23/(257*(257+2^-23)).
    // Constant incomes have zero pairwise differences.
    for (const actual of [rows['gini'][0], rows['summary'][GPU_INEQUALITY_GLOBAL_SUMMARY.GINI]]) {
      if (Number.isNaN(fixture.expected)) {
        expect(actual).toBeNaN();
      } else {
        expect(Number.isFinite(actual)).toBe(true);
        // Relative tolerance does not hide losing the entire small coefficient.
        expect(Math.abs(actual - fixture.expected)).toBeLessThanOrEqual(fixture.expected * 2e-5);
      }
    }
  }
});
