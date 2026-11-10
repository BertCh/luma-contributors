// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPURasterRegistration} from '../../../src/raster-analysis/remote-sensing/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRemoteSensingHarness} from './remote-sensing-test-utils';

/** Deterministic pseudo-random texture, so the correlation peak is unique and unambiguous. */
function makeTexture(width: number, height: number): number[] {
  let state = 12345;
  return Array.from({length: width * height}, () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0xffffffff;
  });
}

// These grids exceed one 256-candidate workgroup, so the peak search crosses the group boundary
// and exercises the second (group folding) pass of the parallel argmax.
it('RS07 periodic peak search spans several workgroups (64 x 64 = 16 groups)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRemoteSensingHarness(device);
  const width = 64;
  const height = 64;
  const referenceValues = makeTexture(width, height);
  const [shiftColumns, shiftRows] = [7, -5];
  const moving = referenceValues.map((_, index) => {
    const column = index % width;
    const row = Math.floor(index / width);
    return referenceValues[
      ((row + shiftRows + height) % height) * width + ((column + shiftColumns + width) % width)
    ];
  });
  const result = await harness.run(
    new GPURasterRegistration({
      grid: makeGPURasterMetadata({width, height}),
      reference: harness.input(referenceValues),
      moving: harness.input(moving),
      upsampleFactor: 4,
      output: {
        shift: harness.output('shift', 'float32x2', 1),
        peak: harness.output('peak', 'float32', 1),
        status: harness.output('status', 'uint32', 1)
      }
    })
  );
  expect(result['status']).toEqual([0]);
  expect(result['shift'][0]).toBeCloseTo(shiftColumns, 1);
  expect(result['shift'][1]).toBeCloseTo(shiftRows, 1);
});

it('RS07 masked peak search spans several workgroups (25 x 25 search cells)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRemoteSensingHarness(device);
  const width = 40;
  const height = 40;
  const referenceValues = makeTexture(width, height);
  const moving = referenceValues.map((_, index) => {
    const column = index % width;
    const row = Math.floor(index / width);
    const sourceColumn = Math.min(width - 1, Math.max(0, column + 3));
    const sourceRow = Math.min(height - 1, Math.max(0, row - 2));
    return referenceValues[sourceRow * width + sourceColumn];
  });
  const result = await harness.run(
    new GPURasterRegistration({
      mode: 'masked-linear',
      grid: makeGPURasterMetadata({width, height}),
      reference: harness.input(referenceValues),
      moving: harness.input(moving),
      referenceValidity: harness.mask(new Array(width * height).fill(1)),
      movingValidity: harness.mask(new Array(width * height).fill(1)),
      maximumShift: [12, 12],
      upsampleFactor: 1,
      output: {
        shift: harness.output('shift', 'float32x2', 1),
        peak: harness.output('peak', 'float32', 1),
        status: harness.output('status', 'uint32', 1)
      }
    })
  );
  expect(result['status']).toEqual([0]);
  expect(Math.abs(result['shift'][0])).toBe(3);
  expect(Math.abs(result['shift'][1])).toBe(2);
  expect(result['peak'][0]).toBeGreaterThan(0.9);
});
