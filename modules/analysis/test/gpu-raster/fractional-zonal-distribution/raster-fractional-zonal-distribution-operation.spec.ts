// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {CPURasterFractionalZonalDistributionOperation} from '../../../src/raster-analysis/raster-zonal-statistics/cpu-raster-fractional-zonal-distribution-operation';
import {GPURasterFractionalZonalDistributionOperation} from '../../../src/raster-analysis/raster-zonal-statistics/gpu-raster-fractional-zonal-distribution-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runPreparedCoverageRecovery} from '../fractional-final1-fixture';
import {
  createDistributionGraphInputs,
  createUnitDistributionInput,
  getDistributionGraphInputs,
  DISTRIBUTION_PROPS
} from './raster-fractional-zonal-distribution-operation-fixture';

it('W5-13 GPU operation writes the inherited means destination captured at construction', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  let destination = 'average';
  let reads = 0;
  const output = {...DISTRIBUTION_PROPS.output};
  Reflect.deleteProperty(output, 'means');
  Object.setPrototypeOf(output, {
    get means() {
      reads++;
      return destination;
    }
  });
  const operation = new GPURasterFractionalZonalDistributionOperation({
    ...DISTRIBUTION_PROPS,
    output
  });
  expect(operation.outputs['means']?.value).toBe('average');
  destination = 'replaced';
  const result = await runOperationGraph(device, {
    inputs: createDistributionGraphInputs(),
    operations: [operation],
    outputs: Object.values(operation.outputs).map(descriptor => descriptor.value)
  });
  // Values 0,10,20 have reliability weights 1,2,1; their mean is 10.
  expect(Array.from(result['average'])).toEqual([10]);
  expect(result['replaced']).toBeUndefined();
  expect(reads).toBe(1);
});

it('F1 distribution operations suppress candidate-only coverage and recover after both failure kinds', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props = {
    ...DISTRIBUTION_PROPS,
    grid: makeGPURasterMetadata({width: 1, height: 1}),
    probabilityCount: 1,
    coverage: {
      ...DISTRIBUTION_PROPS.coverage,
      status: {...DISTRIBUTION_PROPS.coverage.status, candidateOverflow: 'candidateOverflow'}
    }
  };
  const input = createUnitDistributionInput(Float32Array.of(7), Float32Array.of(0.5));
  const inputs = {
    ...getDistributionGraphInputs(input),
    candidateOverflow: {format: 'uint32' as const, values: Uint32Array.of(0)}
  };
  for (const Operation of [
    CPURasterFractionalZonalDistributionOperation,
    GPURasterFractionalZonalDistributionOperation
  ]) {
    for (const result of await runPreparedCoverageRecovery(
      device,
      new Operation(props),
      inputs,
      Object.values(props.output)
    )) {
      expect(result.rows['validity'][0]).toBe(Number(!result.incomplete));
      expect(result.rows['status'][0] & 1).toBe(Number(result.incomplete));
      if (result.incomplete) {
        expect(result.rows['means'][0]).toBeNaN();
        expect(result.rows['coveredArea'][0]).toBeNaN();
      } else {
        expect(result.rows['means'][0]).toBe(7);
        expect(result.rows['coveredArea'][0]).toBe(1);
      }
    }
  }
});

it('G5 distribution operations use the first word of padded coverage statuses', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props = {
    ...DISTRIBUTION_PROPS,
    grid: makeGPURasterMetadata({width: 1, height: 1}),
    probabilityCount: 1,
    coverage: {
      ...DISTRIBUTION_PROPS.coverage,
      status: {...DISTRIBUTION_PROPS.coverage.status, candidateOverflow: 'candidateOverflow'}
    }
  };
  for (const loss of [0, 1]) {
    const input = createUnitDistributionInput(Float32Array.of(7), Float32Array.of(0.5));
    input.coverage.count = Uint32Array.of(1, 0xffffffff);
    input.coverage.requiredCount = Uint32Array.of(1, 0xffffffff);
    input.coverage.overflow = Uint32Array.of(0, 1);
    for (const Operation of [
      CPURasterFractionalZonalDistributionOperation,
      GPURasterFractionalZonalDistributionOperation
    ]) {
      const result = await runOperationGraph(device, {
        inputs: {
          ...getDistributionGraphInputs(input),
          candidateOverflow: {format: 'uint32', values: Uint32Array.of(loss, 1 - loss)}
        },
        operations: [new Operation(props)],
        outputs: Object.values(props.output)
      });
      expect(Array.from(result['validity'])).toEqual([1 - loss]);
      expect(Array.from(result['status'])).toEqual([8 | loss]);
      if (loss) {
        expect(result['means'][0]).toBeNaN();
        expect(result['coveredArea'][0]).toBeNaN();
      } else {
        expect(result['means'][0]).toBe(7);
        expect(result['coveredArea'][0]).toBe(1);
      }
    }
  }
});

for (const quantileMethod of ['exactextract', 'inverse-cdf'] as const) {
  for (const empty of [false, true]) {
    it(`compares every ${quantileMethod} distribution output with empty=${empty}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const outputs = Object.values(DISTRIBUTION_PROPS.output);
      const results = [];
      for (const Operation of [
        GPURasterFractionalZonalDistributionOperation,
        CPURasterFractionalZonalDistributionOperation
      ])
        results.push(
          await runOperationGraph(device, {
            inputs: createDistributionGraphInputs(empty),
            operations: [
              new Operation({...DISTRIBUTION_PROPS, quantileMethod, cellSizeMode: 'uniform'})
            ],
            outputs
          })
        );
      for (const output of outputs) {
        const expected = results[1][output];
        const actual = results[0][output];
        expect(actual.length).toBe(expected.length);
        for (let index = 0; index < expected.length; index++) {
          // Float outputs: 2e-5 relative/absolute tolerance. Flags and NaNs: exact.
          if (expected instanceof Uint32Array || Number.isNaN(expected[index]))
            expect(actual[index]).toBe(expected[index]);
          else
            expect(Math.abs(actual[index] - expected[index])).toBeLessThanOrEqual(
              2e-5 * Math.max(1, Math.abs(expected[index]))
            );
        }
      }
    });
  }
}

for (const scenario of ['rounded-nodata', 'asymmetric-weights'] as const) {
  it(`compares every distribution output for ${scenario}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input =
      scenario === 'rounded-nodata'
        ? createUnitDistributionInput(new Float32Array([0.1, 2]), new Float32Array([0, 0.5, 1]))
        : createUnitDistributionInput(new Float32Array([0, 10, 20]), new Float32Array([0.5]));
    if (scenario === 'rounded-nodata') input.input.noDataValue = 0.1;
    else input.observationWeights = new Float32Array([1, 1, 2]);
    const props = {
      ...DISTRIBUTION_PROPS,
      grid: makeGPURasterMetadata({width: input.input.values.length, height: 1}),
      probabilityCount: input.probabilities.length,
      input: {...DISTRIBUTION_PROPS.input, noDataValue: input.input.noDataValue}
    };
    const outputs = Object.values(props.output);
    const results = [];
    for (const Operation of [
      GPURasterFractionalZonalDistributionOperation,
      CPURasterFractionalZonalDistributionOperation
    ]) {
      results.push(
        await runOperationGraph(device, {
          inputs: getDistributionGraphInputs(input),
          operations: [new Operation(props)],
          outputs
        })
      );
    }
    for (const output of outputs) {
      const expected = results[1][output];
      const actual = results[0][output];
      expect(actual.length).toBe(expected.length);
      for (let index = 0; index < expected.length; index++) {
        // Same tolerance as the original comparison: 2e-5 relative/absolute for floats;
        // status/validity words and NaNs compare exactly.
        if (expected instanceof Uint32Array || Number.isNaN(expected[index]))
          expect(actual[index]).toBe(expected[index]);
        else
          expect(Math.abs(actual[index] - expected[index])).toBeLessThanOrEqual(
            2e-5 * Math.max(1, Math.abs(expected[index]))
          );
      }
    }
    for (const result of results) {
      if (scenario === 'rounded-nodata') {
        expect(Array.from(result['means'])).toEqual([2]);
        expect(Array.from(result['effectiveSampleSizes'])).toEqual([1]);
        expect(Array.from(result['validArea'])).toEqual([1]);
        expect(Array.from(result['status'])).toEqual([8]);
        expect(Array.from(result['quantiles'])).toEqual([2, 2, 2]);
      } else expect(Math.abs(result['quantiles'][0] - 12)).toBeLessThanOrEqual(2e-5);
    }
  });
}

for (const oversized of ['values', 'validity'] as const) {
  it(`distribution operations both reject oversized ${oversized}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = createUnitDistributionInput(new Float32Array([3]), new Float32Array([0.5]));
    if (oversized === 'values') input.input.values = new Float32Array([3, 100]);
    else input.input.validity = new Uint32Array([1, 1]);
    const props = {
      ...DISTRIBUTION_PROPS,
      grid: makeGPURasterMetadata({width: 1, height: 1}),
      probabilityCount: 1
    };
    for (const Operation of [
      GPURasterFractionalZonalDistributionOperation,
      CPURasterFractionalZonalDistributionOperation
    ]) {
      await expect(
        runOperationGraph(device, {
          inputs: getDistributionGraphInputs(input),
          operations: [new Operation(props)],
          outputs: Object.values(props.output)
        })
      ).rejects.toThrow(
        oversized === 'values' ? /exactly one sample per pixel/ : /exactly one flag per pixel/
      );
    }
  });
}

for (const cellSizeMode of ['geographic', 'web-mercator'] as const) {
  it(`compares every distribution output with ${cellSizeMode} row areas`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = createUnitDistributionInput(new Float32Array([3]), new Float32Array([0.5]));
    const props = {
      ...DISTRIBUTION_PROPS,
      grid: {
        ...DISTRIBUTION_PROPS.grid,
        width: 1,
        height: 1,
        affine: [1, 0, 0, 0, -1, 0.5] as const
      },
      cellAreas: undefined,
      cellSizeMode,
      probabilityCount: 1
    };
    const outputs = Object.values(props.output);
    const results = [];
    for (const Operation of [
      GPURasterFractionalZonalDistributionOperation,
      CPURasterFractionalZonalDistributionOperation
    ]) {
      results.push(
        await runOperationGraph(device, {
          inputs: getDistributionGraphInputs(input),
          operations: [new Operation(props)],
          outputs
        })
      );
    }
    for (const output of outputs) {
      const actual = results[0][output];
      const expected = results[1][output];
      expect(actual.length).toBe(expected.length);
      for (let index = 0; index < expected.length; index++) {
        // Float64 ground scales versus WGSL: 2e-5 relative/absolute, flags and NaNs exact.
        if (expected instanceof Uint32Array || Number.isNaN(expected[index]))
          expect(actual[index]).toBe(expected[index]);
        else
          expect(Math.abs(actual[index] - expected[index])).toBeLessThanOrEqual(
            2e-5 * Math.max(1, Math.abs(expected[index]))
          );
      }
    }
    for (const result of results) {
      expect(result['coveredArea'][0]).toBeGreaterThan(0);
      expect(result['validArea'][0]).toBe(result['coveredArea'][0]);
      expect(result['means'][0]).toBe(3);
      expect(result['status'][0]).toBe(8);
    }
  });
}

for (const cellSizeMode of ['uniform', 'geographic', 'web-mercator'] as const) {
  it(`compares every distribution output over three rows with ${cellSizeMode} areas, with and without parameters`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // Three rows of one column far from the equator (60 degrees north, or 8,000 km north in
    // Mercator meters). Zone 0 covers rows 0 and 1, zone 1 covers row 2, so a wrong row index
    // changes the zone areas. The parameters describe a grid 30 rows further south with another
    // cell area, so a twin that read the affine instead of the parameters (or the reverse) differs.
    const step = cellSizeMode === 'web-mercator' ? 500000 : 1;
    const originY = cellSizeMode === 'web-mercator' ? 8000000 : 60;
    const input = createUnitDistributionInput(new Float32Array([3, 4, 5]), new Float32Array([0.5]));
    input.coverage.zoneIds = new Uint32Array([0, 0, 1]);
    input.coverage.zoneValidity = new Uint32Array([1, 1]);
    input.cellAreas = undefined;
    input.parameters = new Float32Array([
      0,
      originY - 30 * step,
      step,
      0,
      0,
      -step,
      -1.5 * step * step,
      0
    ]);
    const coveredAreas: Float32Array[] = [];
    for (const withParameters of [true, false]) {
      const props = {
        ...DISTRIBUTION_PROPS,
        grid: {
          ...DISTRIBUTION_PROPS.grid,
          width: 1,
          height: 3,
          affine: [step, 0, 0, 0, -step, originY] as const
        },
        zoneCount: 2,
        probabilityCount: 1,
        cellAreas: undefined,
        parameters: withParameters ? DISTRIBUTION_PROPS.parameters : undefined,
        cellSizeMode
      };
      const {cellAreas: _cellAreas, parameters, ...graphInputs} = getDistributionGraphInputs(input);
      const outputs = Object.values(props.output);
      const results = [];
      for (const Operation of [
        GPURasterFractionalZonalDistributionOperation,
        CPURasterFractionalZonalDistributionOperation
      ]) {
        results.push(
          await runOperationGraph(device, {
            inputs: withParameters ? {...graphInputs, parameters} : graphInputs,
            operations: [new Operation(props)],
            outputs
          })
        );
      }
      for (const output of outputs) {
        const actual = results[0][output];
        const expected = results[1][output];
        expect(actual.length).toBe(expected.length);
        for (let index = 0; index < expected.length; index++) {
          // Float32 WGSL versus float64 ground scales: 2e-5 relative/absolute (measured at most
          // 3.2e-7 on the test GPU); flags and NaNs exact.
          if (expected instanceof Uint32Array || Number.isNaN(expected[index]))
            expect(actual[index], `${output}[${index}]`).toBe(expected[index]);
          else
            expect(
              Math.abs(actual[index] - expected[index]),
              `${output}[${index}]`
            ).toBeLessThanOrEqual(2e-5 * Math.max(1, Math.abs(expected[index])));
        }
      }
      expect(Array.from(results[1]['means'])).toEqual([3.5, 5]);
      coveredAreas.push(results[1]['coveredArea'] as Float32Array);
    }
    // The parameters, not the affine, decide the areas when they are given.
    for (const zone of [0, 1]) {
      expect(
        Math.abs(coveredAreas[0][zone] - coveredAreas[1][zone]) / coveredAreas[1][zone]
      ).toBeGreaterThan(0.01);
    }
  });
}
