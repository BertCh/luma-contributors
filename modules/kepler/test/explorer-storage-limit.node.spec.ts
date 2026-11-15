// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {NullDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {GPUResidentExplorer} from '../src/exploration/gpu-resident-explorer';

function createDevice(maxStorageBuffersPerShaderStage: number): NullDevice {
  const device = new NullDevice({});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxStorageBuffersPerShaderStage}
  });
  return device;
}

function createExternalDimensions(count: number) {
  return Array.from({length: count}, (_, index) => ({
    kind: 'external' as const,
    id: `dimension-${index}`,
    mask: undefined as never
  }));
}

test('explorer rejects more crossfilter dimensions than one compose pass can bind', () => {
  // Eight selection masks plus the output need nine storage buffers in the compose pass.
  expect(
    () =>
      new GPUResidentExplorer({
        device: createDevice(8),
        columns: {},
        sourceChunkLengths: [1],
        dimensions: createExternalDimensions(8)
      })
  ).toThrow(/Explorer dimensions: 8 selection masks need 9 storage buffers.*is 8/);
});

test('explorer counts the liveness mask as a composed selection', () => {
  expect(
    () =>
      new GPUResidentExplorer({
        device: createDevice(8),
        columns: {},
        sourceChunkLengths: [1],
        dimensions: createExternalDimensions(7),
        liveMask: {} as never
      })
  ).toThrow(/8 selection masks need 9 storage buffers/);
});
