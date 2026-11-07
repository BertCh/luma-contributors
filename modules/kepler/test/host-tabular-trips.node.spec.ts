// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createNullWebGPUDevice} from '../../analysis/test/utils/gpu-contributor-test-utils';
import {expect, test} from 'vitest';
import {TripRuntime} from '../../../examples/deck/kepler-compute/src/trip-runtime';
import {createTabularTripRuntimeFixture} from './utils/tabular-trip-runtime-fixture';

test('tabular trip masks use member rows and account the intermediate member flags', async () => {
  const device = createNullWebGPUDevice();
  const fixture = createTabularTripRuntimeFixture(device, [0, 1, 0, 0, 0, 0]);
  try {
    expect(fixture.trips.source.tripCount).toBe(2);
    const runtime = new TripRuntime({
      device,
      column: 'id',
      trips: fixture.trips,
      selectionMask: fixture.selectionMask,
      workingMemoryBudgetBytes: 1024
    });
    // Trip flags and member flags, plus same-layout filtered paths and times (88 bytes).
    expect(runtime.accountedResidentByteLength).toBe(124);
    expect(runtime.binding.getPath).not.toBe(fixture.trips.getPath);
    expect(runtime.binding.getPath.data[0].valueOffsets).toBe(
      fixture.trips.getPath.data[0].valueOffsets
    );
    expect(runtime.binding.selectionMask.data.map(chunk => chunk.length)).toEqual([2, 0]);
    expect(runtime.binding.resolveSourceRow(1)).toBe(1);
    expect(
      () =>
        new TripRuntime({
          device,
          column: 'id',
          trips: fixture.trips,
          selectionMask: fixture.selectionMask,
          workingMemoryBudgetBytes: 123
        })
    ).toThrow('outputs exceed');
    await runtime.destroy(Promise.resolve());
  } finally {
    fixture.destroy();
    device.destroy();
  }
});
