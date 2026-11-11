// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPURasterFractionalZonalStatisticsOperation} from '../../src/raster-analysis/raster-zonal-statistics/cpu-raster-fractional-zonal-statistics-operation';
import {GPURasterFractionalZonalStatisticsOperation} from '../../src/raster-analysis/raster-zonal-statistics/gpu-raster-fractional-zonal-statistics-operation';
import {runOperationGraph} from '../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import {
  createStatisticsInput,
  createStatisticsGraphInputs,
  createSingletonStatisticsProps,
  createSingletonStatisticsInput,
  STATISTICS_PROPS
} from './raster-fractional-zonal-statistics-operation-fixture';

it('compares all zonal outputs with unequal fractions, missing data, repacked area and incomplete coverage', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const quantity of ['extensive', 'density', 'intensive'] as const) {
    for (const scenario of [
      'ordinary',
      'overflow',
      'all-missing',
      'invalid-zone',
      'tiny-coverage'
    ] as const) {
      const input = createStatisticsInput();
      if (scenario === 'overflow') input.coverage.overflow[0] = 1;
      if (scenario === 'all-missing') input.input.validity!.fill(0);
      if (scenario === 'invalid-zone') input.coverage.zoneValidity[1] = 0;
      if (scenario === 'tiny-coverage') {
        input.coverage.count[0] = 1;
        input.coverage.fractions[0] = 1e-32;
      }
      const props = {...STATISTICS_PROPS, quantity};
      const run = (
        operation:
          | CPURasterFractionalZonalStatisticsOperation
          | GPURasterFractionalZonalStatisticsOperation
      ) =>
        runOperationGraph(device, {
          inputs: createStatisticsGraphInputs(input),
          operations: [operation],
          outputs: Object.values(props.output)
        });
      const cpu = await run(new CPURasterFractionalZonalStatisticsOperation(props));
      const gpu = await run(new GPURasterFractionalZonalStatisticsOperation(props));
      if (scenario === 'tiny-coverage') {
        // Normal float32 weight about 1e-32: 5 * weight / 1e-30 is about 0.05.
        for (const result of [cpu, gpu]) {
          expect(Math.abs(result['means'][0] - 0.05)).toBeLessThanOrEqual(2e-7);
          expect(result['validity'][0]).toBe(1);
        }
      }
      expect(Array.from(gpu['validity'])).toEqual(Array.from(cpu['validity']));
      for (const port of ['totals', 'means', 'coveredArea', 'validArea']) {
        for (let zone = 0; zone < props.zoneCount; zone++) {
          const expected = cpu[port][zone];
          if (Number.isNaN(expected)) expect(gpu[port][zone]).toBeNaN();
          else if (expected === 0) expect(gpu[port][zone] === 0).toBe(true);
          // Relative tolerance: a positive result cannot pass as zero.
          else
            expect(Math.abs(gpu[port][zone] - expected)).toBeLessThanOrEqual(
              3e-6 * Math.abs(expected)
            );
        }
      }
    }
  }
});

it('compares all singleton outputs with surplus zone-validity, coverage-column and status rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props = createSingletonStatisticsProps();
  for (const surplusCoverageRows of [false, true]) {
    const input = createSingletonStatisticsInput(surplusCoverageRows);
    const run = (
      operation:
        | CPURasterFractionalZonalStatisticsOperation
        | GPURasterFractionalZonalStatisticsOperation
    ) =>
      runOperationGraph(device, {
        inputs: createStatisticsGraphInputs(input),
        operations: [operation],
        outputs: Object.values(props.output)
      });
    const cpu = await run(new CPURasterFractionalZonalStatisticsOperation(props));
    const gpu = await run(new GPURasterFractionalZonalStatisticsOperation(props));
    for (const [port, expected] of Object.entries({
      totals: 5,
      means: 5,
      coveredArea: 1,
      validArea: 1,
      validity: 1
    })) {
      // All values in this hand-computed case are exactly representable.
      expect(Array.from(cpu[port])).toEqual([expected]);
      expect(Array.from(gpu[port])).toEqual([expected]);
    }
  }
});
