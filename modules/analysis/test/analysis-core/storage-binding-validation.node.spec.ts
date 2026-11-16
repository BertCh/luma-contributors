// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {validateGPUStorageBindingByteLength} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {createNullWebGPUDevice} from '../../src/testing/gpu-contributor-test-utils';

it('rejects malformed byte lengths before they can bypass the binding limit', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const byteLength of [NaN, Infinity, -Infinity, -1, 0.5, Number.MAX_SAFE_INTEGER + 1]) {
      expect(() =>
        validateGPUStorageBindingByteLength(device, 'binding', 'scratch', byteLength, 'reduce rows')
      ).toThrow(/^binding scratch byteLength must be a nonnegative safe integer/);
    }
    expect(() =>
      validateGPUStorageBindingByteLength(device, 'binding', 'scratch', 0, 'reduce rows')
    ).not.toThrow();
    expect(() =>
      validateGPUStorageBindingByteLength(
        device,
        'binding',
        'scratch',
        device.limits.maxStorageBufferBindingSize,
        'reduce rows'
      )
    ).not.toThrow();
    expect(() =>
      validateGPUStorageBindingByteLength(
        device,
        'binding',
        'scratch',
        device.limits.maxStorageBufferBindingSize + 4,
        'reduce rows'
      )
    ).toThrow(/binding scratch needs/);
  } finally {
    device.destroy();
  }
});
