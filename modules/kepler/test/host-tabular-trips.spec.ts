// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {type GPUVector} from '@luma-contributors/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {TripRuntime} from '../../../examples/deck/kepler-compute/src/trip-runtime';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';
import {createTabularTripRuntimeFixture} from './utils/tabular-trip-runtime-fixture';

async function readWords(vector: GPUVector): Promise<number[]> {
  const result: number[] = [];
  for (const chunk of vector.data) {
    if (!chunk.length) continue;
    const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.length * 4);
    result.push(...new Uint32Array(bytes.buffer, bytes.byteOffset, chunk.length));
  }
  return result;
}

it('tabular trips are visible when any member row is selected, across source batches', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Trip a = rows 0,4 and trip b = rows 1,2,5; row 3 is an invalid single-vertex trip.
  const cases: [number[], number[]][] = [
    [
      [0, 0, 0, 1, 0, 0],
      [0, 0]
    ],
    [
      [0, 0, 0, 0, 1, 0],
      [1, 0]
    ],
    [
      [0, 0, 1, 0, 0, 0],
      [0, 1]
    ],
    [
      [1, 0, 0, 0, 0, 1],
      [1, 1]
    ],
    [
      [0, 0, 0, 0, 0, 0],
      [0, 0]
    ]
  ];
  for (const [mask, expected] of cases) {
    const fixture = createTabularTripRuntimeFixture(device, mask);
    const runtime = new TripRuntime({
      device,
      column: 'id',
      trips: fixture.trips,
      selectionMask: fixture.selectionMask,
      workingMemoryBudgetBytes: 1024 * 1024
    });
    try {
      const encoder = device.createCommandEncoder();
      runtime.encode(encoder);
      device.submit(encoder.finish());
      expect(await readWords(runtime.binding.selectionMask)).toEqual(expected);
    } finally {
      await runtime.destroy(Promise.resolve());
      fixture.destroy();
    }
  }
});

it('tabular trips drop unselected vertices like native by repeating the nearest selected vertex', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Trip b is rows 1 (t=0), 2 (t=100), 5 (t=200); trip a is rows 0 (t=0), 4 (t=100).
  const cases: [number[], number[]][] = [
    [
      [0, 0, 1, 0, 0, 1],
      [0, 100, 100, 100, 200]
    ],
    [
      [0, 1, 0, 0, 0, 0],
      [0, 100, 0, 0, 0]
    ],
    [
      [0, 0, 0, 0, 0, 1],
      [0, 100, 200, 200, 200]
    ]
  ];
  for (const [mask, expectedTimes] of cases) {
    const fixture = createTabularTripRuntimeFixture(device, mask);
    const runtime = new TripRuntime({
      device,
      column: 'id',
      trips: fixture.trips,
      selectionMask: fixture.selectionMask,
      workingMemoryBudgetBytes: 1024 * 1024
    });
    try {
      const encoder = device.createCommandEncoder();
      runtime.encode(encoder);
      device.submit(encoder.finish());
      const words: number[] = [];
      for (const chunk of runtime.binding.getTimestamps.data) {
        if (!chunk.valueLength) continue;
        const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.valueLength * 4);
        words.push(...new Uint32Array(bytes.buffer, bytes.byteOffset, chunk.valueLength));
      }
      expect(Array.from(new Float32Array(Uint32Array.from(words).buffer))).toEqual(expectedTimes);
    } finally {
      await runtime.destroy(Promise.resolve());
      fixture.destroy();
    }
  }
});
