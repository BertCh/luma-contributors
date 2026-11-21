// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {expect, it} from 'vitest';
import {
  getGPUCommandNodeKernelSources,
  getPrivateMemoryBytes,
  getWorkgroupMemoryBytes
} from '../../src/testing/gpu-contributor-test-utils';

const SOURCE = `override N: u32 = 1u;
var<workgroup> scratch: array<u32, N>;
@compute @workgroup_size(1) fn main() {
  var values: array<u32, 4>;
  values[0] = 7u;
  scratch[0] = values[0];
}`;

it('measures the complete recorded override configuration (TOOL-118)', () => {
  const configurations = getGPUCommandNodeKernelSources(
    [4, 8192].map(length => ({
      compile: ({device}: {device: Device}) => {
        const shader = device.createShader({stage: 'compute', source: SOURCE});
        device.createComputePipeline({id: 'same-kernel', shader, constants: {N: length}});
        return {destroy: () => shader.destroy()};
      }
    }))
  );
  expect(configurations).toHaveLength(2);
  expect(configurations[0].source).toBe(configurations[1].source);
  // u32 has four bytes: 4 * 4 = 16; 8192 * 4 = 32768. Both are already 16-byte aligned.
  expect(configurations.map(configuration => getWorkgroupMemoryBytes(configuration))).toEqual([
    16, 32768
  ]);
  expect(configurations.map(configuration => getPrivateMemoryBytes(configuration))).toEqual([
    16, 16
  ]);
});

it('preserves raw-source estimation with default or explicit override constants (TOOL-118)', () => {
  expect(getWorkgroupMemoryBytes(SOURCE)).toBe(16);
  expect(getWorkgroupMemoryBytes(SOURCE, {N: 8192})).toBe(32768);
  // Standalone shader fixture: four u32 elements occupy 16 bytes of function-scope memory.
  expect(getPrivateMemoryBytes('fn probe() { var values: array<u32, 4>; values[0] = 1u; }')).toBe(
    16
  );
});
