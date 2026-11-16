// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';

// A synthetic 64 KiB binding ceiling makes a 10,001-row uint64 input exceed one binding while
// its separate key chunks and every row-sized scalar scratch binding still fit, cheaply.
export const GROUP_STATISTICS_TEST_STORAGE_BINDING_BYTE_LENGTH = 65536;

/**
 * Runs the synthetic binding-limit fixture on the original luma.gl Device. A device proxy would
 * differ from buffer.device and fail GPUCommandGraph's imported-buffer ownership check before
 * the contributor runs. Only the limits property is replaced, and its exact descriptor is
 * restored after the callback completes or throws. Use in sequential tests on the shared device.
 */
export async function withGroupStatisticsTestBindingLimit<Result>(
  device: Device,
  run: (device: Device) => Result | Promise<Result>
): Promise<Result> {
  const originalDescriptor = Object.getOwnPropertyDescriptor(device, 'limits');
  if (!originalDescriptor) {
    throw new Error('group-statistics test device must expose its own limits property');
  }
  const limits = new Proxy(device.limits, {
    get(deviceLimits, property) {
      return property === 'maxStorageBufferBindingSize'
        ? GROUP_STATISTICS_TEST_STORAGE_BINDING_BYTE_LENGTH
        : Reflect.get(deviceLimits, property, deviceLimits);
    }
  });
  Object.defineProperty(device, 'limits', {value: limits});
  try {
    return await run(device);
  } finally {
    Object.defineProperty(device, 'limits', originalDescriptor);
  }
}
