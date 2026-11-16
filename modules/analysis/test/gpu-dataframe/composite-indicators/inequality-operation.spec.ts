// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUInequalityOperation} from '../../../src/dataframe-analysis/composite-indicators/cpu-inequality-operation';
import {
  GPUInequalityOperation,
  type InequalityOperationProps
} from '../../../src/dataframe-analysis/composite-indicators/gpu-inequality-operation';
import {getGPUInequalityParameterValues} from '../../../src/dataframe-analysis/composite-indicators/inequality-parameters';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('the CPU operation defines empty input while the GPU contributor rejects it', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const inputs = {
    values: {format: 'float32' as const, values: new Float32Array()},
    zoneIds: {format: 'uint32' as const, values: new Uint32Array()},
    parameters: {format: 'float32' as const, values: getGPUInequalityParameterValues()}
  };
  const output = {
    gini: 'gini',
    theilT: 'theilT',
    theilL: 'theilL',
    atkinson: 'atkinson',
    hoover: 'hoover',
    palma: 'palma',
    mean: 'mean',
    counts: 'counts',
    lorenzKnots: 'knots',
    globalSummary: 'summary'
  };
  const props = {
    id: 'empty-inequality',
    values: 'values',
    zoneIds: 'zoneIds',
    parameters: 'parameters',
    zoneCount: 1,
    output
  };
  await expect(
    runOperationGraph(device, {
      inputs,
      operations: [new GPUInequalityOperation(props)],
      outputs: Object.values(output)
    })
  ).rejects.toThrow(/^empty-inequality values length must be an integer in \[1, 2147483648\]$/);
  const result = await runOperationGraph(device, {
    inputs,
    operations: [new CPUInequalityOperation(props)],
    outputs: Object.values(output)
  });
  expect(Array.from(result['counts'])).toEqual([0]);
  for (const name of ['gini', 'theilT', 'theilL', 'atkinson', 'hoover', 'palma', 'mean', 'knots'])
    expect(result[name].every(Number.isNaN)).toBe(true);
  expect(Array.from(result['summary'])).toEqual([NaN, NaN, NaN, NaN, 0, NaN, 0, 0]);
});

it('compares every inequality output for weighted zones, zeros, excluded rows and empty zones', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const output = {
    gini: 'gini',
    theilT: 'theilT',
    theilL: 'theilL',
    atkinson: 'atkinson',
    hoover: 'hoover',
    palma: 'palma',
    mean: 'mean',
    counts: 'counts',
    lorenzKnots: 'knots',
    globalSummary: 'summary'
  };
  const props: InequalityOperationProps = {
    values: 'values',
    zoneIds: 'zoneIds',
    weights: 'weights',
    mask: 'mask',
    parameters: 'parameters',
    zoneCount: 4,
    lorenzKnotCount: 5,
    output
  };
  for (const epsilon of [0, 0.5, 1, 2]) {
    for (const entirelyMasked of [false, true]) {
      const inputs = {
        values: {
          format: 'float32' as const,
          values: Float32Array.of(1, 3, 7, 0, 2, 0, 0, -1, NaN, 8)
        },
        zoneIds: {
          format: 'uint32' as const,
          values: Uint32Array.of(0, 0, 1, 1, 1, 2, 2, 0, 0, 0xffffffff)
        },
        weights: {
          format: 'float32' as const,
          values: Float32Array.of(2, 1, 3, 1, 1, 1, 1, 1, 1, 1)
        },
        mask: {format: 'uint32' as const, values: new Uint32Array(10).fill(entirelyMasked ? 0 : 1)},
        parameters: {
          format: 'float32' as const,
          values: getGPUInequalityParameterValues({
            epsilon,
            palmaTopShare: 0.25,
            palmaBottomShare: 0.5
          })
        }
      };
      const results: Awaited<ReturnType<typeof runOperationGraph>>[] = [];
      for (const Operation of [GPUInequalityOperation, CPUInequalityOperation])
        results.push(
          await runOperationGraph(device, {
            inputs,
            operations: [new Operation(props)],
            outputs: Object.values(output)
          })
        );
      for (const name of Object.values(output)) {
        expect(results[0][name].length).toBe(results[1][name].length);
        results[1][name].forEach((expected, row) => {
          if (Number.isNaN(expected)) expect(results[0][name][row]).toBeNaN();
          else if (name === 'counts') expect(results[0][name][row]).toBe(expected);
          // 2e-5 absolute plus relative tolerance for float32 cumulative sums and transcendentals.
          else
            expect(Math.abs(results[0][name][row] - expected)).toBeLessThanOrEqual(
              2e-5 * Math.max(1, Math.abs(expected))
            );
        });
      }
    }
  }
});
