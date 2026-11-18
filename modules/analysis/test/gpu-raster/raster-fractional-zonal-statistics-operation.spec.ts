// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPURasterFractionalZonalStatisticsOperation} from '../../src/raster-analysis/raster-zonal-statistics/cpu-raster-fractional-zonal-statistics-operation';
import {GPURasterFractionalZonalStatisticsOperation} from '../../src/raster-analysis/raster-zonal-statistics/gpu-raster-fractional-zonal-statistics-operation';
import {runOperationGraph} from '../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import {runPreparedCoverageRecovery} from './fractional-final1-fixture';
import {createScalarAreaReviewFixture} from './scalar-area-review-fixture';
import {
  createStatisticsInput,
  createStatisticsGraphInputs,
  createSingletonStatisticsProps,
  createSingletonStatisticsInput,
  STATISTICS_PROPS
} from './raster-fractional-zonal-statistics-operation-fixture';

it('F1 scalar operations suppress independent incomplete evidence and recover on the same graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const base = createSingletonStatisticsProps();
  const props = {
    ...base,
    input: {...base.input, scale: 1, offset: 0},
    coverage: {
      ...base.coverage,
      status: {...base.coverage.status, candidateOverflow: 'candidateOverflow'}
    }
  };
  const input = createSingletonStatisticsInput();
  input.input.values[0] = 7;
  input.input.scale = 1;
  input.input.offset = 0;
  const inputs = {
    ...createStatisticsGraphInputs(input),
    candidateOverflow: {format: 'uint32' as const, values: Uint32Array.of(0)}
  };
  for (const Operation of [
    CPURasterFractionalZonalStatisticsOperation,
    GPURasterFractionalZonalStatisticsOperation
  ]) {
    for (const result of await runPreparedCoverageRecovery(
      device,
      new Operation(props),
      inputs,
      Object.values(props.output)
    )) {
      expect(result.rows['validity'][0]).toBe(Number(!result.incomplete));
      if (result.incomplete)
        for (const port of ['totals', 'means', 'coveredArea', 'validArea'])
          expect(result.rows[port][0]).toBeNaN();
      else {
        expect(result.rows['means'][0]).toBe(7);
        expect(result.rows['coveredArea'][0]).toBe(1);
      }
    }
  }
});

it('F7 both scalar operations preserve one observation of 7 at coverage 1e-32', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const base = createSingletonStatisticsProps();
  const props = {...base, input: {...base.input, scale: 1, offset: 0}};
  const input = createSingletonStatisticsInput();
  input.input.values[0] = 7;
  input.input.scale = 1;
  input.input.offset = 0;
  input.coverage.fractions[0] = 1e-32;
  for (const Operation of [
    CPURasterFractionalZonalStatisticsOperation,
    GPURasterFractionalZonalStatisticsOperation
  ]) {
    const result = await runOperationGraph(device, {
      inputs: createStatisticsGraphInputs(input),
      operations: [new Operation(props)],
      outputs: Object.values(props.output)
    });
    expect(result['validity'][0]).toBe(1);
    expect(Number.isFinite(result['means'][0])).toBe(true);
    expect(result['means'][0]).toBeCloseTo(7, 5);
  }
});

it('R3 R6 R7 preserve offset support, suppress overflow per zone and ignore unused affine area', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scenario of ['offset', 'oversized', 'overflow', 'large-affine'] as const) {
    const {props, graphInputs} = createScalarAreaReviewFixture(scenario);
    for (const Operation of [
      CPURasterFractionalZonalStatisticsOperation,
      GPURasterFractionalZonalStatisticsOperation
    ]) {
      const run = () =>
        runOperationGraph(device, {
          inputs: graphInputs,
          operations: [new Operation(props)],
          outputs: Object.values(props.output)
        });
      if (scenario === 'oversized') {
        await expect(run()).rejects.toThrow(/cellAreas/);
        continue;
      }
      const result = await run();
      const zone = scenario === 'overflow' ? 1 : 0;
      expect(result['coveredArea'][zone]).toBe(3);
      expect(result['validArea'][zone]).toBe(3);
      expect(result['totals'][zone]).toBe(6);
      expect(result['means'][zone]).toBe(2);
      expect(result['validity'][zone]).toBe(1);
      if (scenario === 'overflow') {
        expect(result['validity'][0]).toBe(0);
        for (const port of ['coveredArea', 'validArea', 'totals', 'means'])
          expect(result[port][0]).toBeNaN();
      }
    }
  }
});

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
        input.coverage.requiredCount[0] = 1;
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
        // Normal float32 weight about 1e-32 must preserve the calibrated observation 5.
        for (const result of [cpu, gpu]) {
          expect(Math.abs(result['means'][0] - 5)).toBeLessThanOrEqual(2e-6);
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

it('compares scalar ground-area and explicit-area operation outputs including invalid support', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const mode of ['geographic', 'web-mercator', 'explicit', 'invalid-area'] as const) {
    const input = createStatisticsInput();
    const props = {
      ...STATISTICS_PROPS,
      cellSizeMode: mode === 'explicit' || mode === 'invalid-area' ? ('uniform' as const) : mode,
      cellAreas: mode === 'explicit' || mode === 'invalid-area' ? 'cellAreas' : undefined
    };
    const inputs = createStatisticsGraphInputs(input);
    if (props.cellAreas)
      inputs['cellAreas'] = {
        format: 'float32',
        values: Float32Array.of(mode === 'invalid-area' ? NaN : 3, 7, 5, 2, 9, 4)
      };
    const run = (
      operation:
        | CPURasterFractionalZonalStatisticsOperation
        | GPURasterFractionalZonalStatisticsOperation
    ) =>
      runOperationGraph(device, {
        inputs,
        operations: [operation],
        outputs: Object.values(props.output)
      });
    const cpu = await run(new CPURasterFractionalZonalStatisticsOperation(props));
    const gpu = await run(new GPURasterFractionalZonalStatisticsOperation(props));
    for (const port of Object.keys(props.output)) {
      for (let zone = 0; zone < props.zoneCount; zone++) {
        const expected = cpu[port][zone];
        if (Number.isNaN(expected)) expect(gpu[port][zone]).toBeNaN();
        else {
          expect(Number.isFinite(gpu[port][zone])).toBe(true);
          expect(Math.abs(gpu[port][zone] - expected)).toBeLessThanOrEqual(
            3e-6 * Math.max(1, Math.abs(expected))
          );
        }
      }
    }
  }
});
