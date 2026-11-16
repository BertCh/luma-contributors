// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';

/** Native supported limits are read-only. Override the wrapper property for this scope only. */
export function withDensityTestBindingLimit<Result>(
  device: Device,
  byteLength: number | undefined,
  run: () => Result
): Result {
  if (byteLength === undefined) return run();
  const descriptor = Object.getOwnPropertyDescriptor(device, 'limits');
  if (!descriptor) throw new Error('density test device must expose its own limits property');
  const limits = new Proxy(device.limits, {
    get(deviceLimits, property) {
      return property === 'maxStorageBufferBindingSize' || property === 'maxBufferSize'
        ? byteLength
        : Reflect.get(deviceLimits, property, deviceLimits);
    }
  });
  Object.defineProperty(device, 'limits', {value: limits});
  try {
    return run();
  } finally {
    Object.defineProperty(device, 'limits', descriptor);
  }
}
