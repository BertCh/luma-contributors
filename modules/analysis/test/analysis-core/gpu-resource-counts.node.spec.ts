// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, luma} from '@luma.gl/core';
import {expect, it} from 'vitest';
import {
  getGPUResourceCounts,
  getGPUResourceLeaks,
  validateNoGPUResourceLeaks
} from '../../src/testing/gpu-resource-counts';
import {createNullWebGPUDevice} from '../../src/testing/gpu-contributor-test-utils';

it('reads the maintained texture counter and takes independent snapshots', () => {
  const statistics = luma.stats.get('GPU Resource Counts');
  const baseline = getGPUResourceCounts();
  statistics.get('Textures Active').incrementCount();
  try {
    expect(getGPUResourceLeaks(baseline)).toEqual({buffers: 0, textures: 1});
    expect(() => validateNoGPUResourceLeaks('textures', baseline)).toThrow(
      'textures leaked GPU resources: 0 buffers, 1 textures'
    );
  } finally {
    statistics.get('Textures Active').decrementCount();
  }
  validateNoGPUResourceLeaks('textures', baseline);
});

it('detects a live buffer and clears the finding after destruction', () => {
  const device = createNullWebGPUDevice();
  const baseline = getGPUResourceCounts();
  const buffer = device.createBuffer({byteLength: 4, usage: Buffer.STORAGE});
  try {
    expect(getGPUResourceLeaks(baseline)).toEqual({buffers: 1, textures: 0});
  } finally {
    buffer.destroy();
    device.destroy();
  }
  validateNoGPUResourceLeaks('buffer', baseline);
});

it('keeps resource types separate when pre-existing buffers are freed', () => {
  expect(getGPUResourceLeaks({buffers: 5, textures: 0}, {buffers: 4, textures: 1})).toEqual({
    buffers: 0,
    textures: 1
  });
});

it('recording helpers destroy compiled node resources, including when a later compile fails', async () => {
  const {getGPUCommandNodeKernelSources, recordComputePipelines} = await import(
    '../../src/testing/gpu-contributor-test-utils'
  );
  for (const record of [getGPUCommandNodeKernelSources, recordComputePipelines]) {
    const baseline = getGPUResourceCounts();
    const node = {
      compile: ({device}: {device: import('@luma.gl/core').Device}) => {
        const buffer = device.createBuffer({byteLength: 256, usage: Buffer.UNIFORM});
        return {destroy: () => buffer.destroy()};
      }
    };
    record([node]);
    validateNoGPUResourceLeaks('recorded node', baseline);
    expect(() =>
      record([
        node,
        {
          compile: () => {
            throw new Error('compile failed');
          }
        }
      ])
    ).toThrow('compile failed');
    validateNoGPUResourceLeaks('failed recording', baseline);
  }
});

it('constructs the exact headless lifecycle fixture without leaking its buffers', async () => {
  const {createResourceLifecycleFixture} = await import('./gpu-resource-lifecycle.test-utils');
  const device = createNullWebGPUDevice();
  const baseline = getGPUResourceCounts();
  try {
    const fixture = createResourceLifecycleFixture(device);
    try {
      expect(fixture.rowCount).toBe(513);
    } finally {
      fixture.test.destroy();
    }
    validateNoGPUResourceLeaks('lifecycle fixture', baseline);
  } finally {
    device.destroy();
  }
});
