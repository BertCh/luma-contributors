// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUTerrainCatchmentSummaryOperation} from '../../../src/gpu-terrain/terrain-processing/cpu-terrain-catchment-summary-operation';
import {GPUTerrainCatchmentSummaryOperation} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  makeCatchmentInput,
  makeCatchmentProps
} from './terrain-catchment-summary-operation-fixture';

it('compares every summary/profile output on multiple basins, bounds, cycles and missing inputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scene of [
    'ordinary',
    'bounded',
    'cycle',
    'missing',
    'geographic',
    'mercator',
    'cancellation',
    'cancellation-permuted',
    'close-extrema'
  ] as const) {
    const input = makeCatchmentInput(),
      props = makeCatchmentProps();
    if (scene === 'bounded') props.maximumSteps = 2;
    if (scene === 'cycle') input.flowDirections[1] = 16;
    if (scene === 'missing') {
      input.validity[2] = 0;
      input.streams[6] = 0xffffffff;
      input.basinIds[1] = 0xffffffff;
    }
    if (scene === 'geographic') {
      props.cellSizeMode = 'geographic';
      props.grid = {...props.grid, affine: [0.01, 0, 0, 0, -0.01, 60]};
    }
    if (scene === 'mercator') {
      props.cellSizeMode = 'web-mercator';
      props.grid = {...props.grid, affine: [20, 0, 0, 0, -30, 4000000]};
    }
    if (
      scene === 'cancellation' ||
      scene === 'cancellation-permuted' ||
      scene === 'close-extrema'
    ) {
      const magnitude = 2 ** 60;
      props.grid = {...props.grid, width: 3, height: 1, affine: [1, 0, 0, 0, -1, 0]};
      props.basinCount = 1;
      props.histogramBins = 2;
      input.elevation =
        scene === 'close-extrema'
          ? Float32Array.of(8, 8.0001, 8)
          : scene === 'cancellation'
            ? Float32Array.of(magnitude, 1, -magnitude)
            : Float32Array.of(magnitude, -magnitude, 1);
      input.basinIds = new Uint32Array(3);
      input.streams = new Uint32Array(3);
      input.flowDirections = new Uint32Array(3);
      input.validity = new Uint32Array(3).fill(1);
    }
    const results = [];
    for (const Operation of [
      GPUTerrainCatchmentSummaryOperation,
      CPUTerrainCatchmentSummaryOperation
    ])
      results.push(
        await runOperationGraph(device, {
          inputs: {
            elevation: {format: 'float32', values: input.elevation},
            validity: {format: 'uint32', values: input.validity},
            basinIds: {format: 'uint32', values: input.basinIds},
            streams: {format: 'uint32', values: input.streams},
            flowDirections: {format: 'uint32', values: input.flowDirections}
          },
          operations: [new Operation(props)],
          outputs: Object.values(props.output)
        })
      );
    expect(Array.from(results[0]['profileStatus'])).toEqual(
      Array.from(results[1]['profileStatus'])
    );
    for (const port of ['statistics', 'hypsometry', 'profiles']) {
      expect(results[0][port].length).toBe(results[1][port].length);
      for (let index = 0; index < results[1][port].length; index++) {
        const actual = results[0][port][index],
          expected = results[1][port][index];
        if (Number.isNaN(expected)) expect(actual).toBeNaN();
        else if (
          expected === 0 ||
          (port === 'profiles' && index % 3 === 1) ||
          (port === 'statistics' && (index % 6 === 2 || index % 6 === 3))
        )
          expect(actual === expected).toBe(true);
        // Relative tolerance covers WGS84/edge arithmetic and fixed-point float32 terms.
        else expect(Math.abs(actual - expected)).toBeLessThanOrEqual(Math.abs(expected) * 2e-5);
      }
    }
    for (const result of results) {
      if (scene === 'cancellation' || scene === 'cancellation-permuted') {
        // Kernel-derived GPU hypothesis: all three integer terms survive the fixed-point sum.
        const expectedMean = Math.fround(1 / 3);
        expect(Math.abs(result['statistics'][1] - expectedMean)).toBeLessThanOrEqual(
          expectedMean * 2e-5
        );
      }
      if (scene === 'ordinary' || scene === 'close-extrema')
        expect(result['statistics'][2]).toBe(8);
      if (scene === 'close-extrema') expect(result['statistics'][3]).toBe(Math.fround(8.0001));
    }
  }
});
