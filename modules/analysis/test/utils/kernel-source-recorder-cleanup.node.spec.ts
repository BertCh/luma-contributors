// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {expect, it} from 'vitest';
import {getGPUCommandNodeKernelSources} from '../../src/testing/gpu-contributor-test-utils';

// TOOL-121: a node whose compile allocates and then throws returns no compiled object, so the
// recorder itself must release what the node created.
it('getGPUCommandNodeKernelSources destroys resources a throwing compile allocated', () => {
  const allocated: Buffer[] = [];
  const failingNode = {
    workload: {operation: 'FailingFixture'},
    compile: ({device}: {device: Device}) => {
      allocated.push(device.createBuffer({byteLength: 16, usage: Buffer.STORAGE}));
      throw new Error('fixture compile failure');
    }
  };
  expect(() => getGPUCommandNodeKernelSources([failingNode])).toThrow('fixture compile failure');
  expect(allocated).toHaveLength(1);
  expect(allocated[0].destroyed).toBe(true);
});

it('getGPUCommandNodeKernelSources leaves resource release to a successful compile', () => {
  const allocated: Buffer[] = [];
  const node = {
    workload: {operation: 'OwningFixture'},
    compile: ({device}: {device: Device}) => {
      const buffer = device.createBuffer({byteLength: 16, usage: Buffer.STORAGE});
      allocated.push(buffer);
      return {destroy: () => buffer.destroy()};
    }
  };
  expect(getGPUCommandNodeKernelSources([node])).toEqual([]);
  expect(allocated[0].destroyed).toBe(true);
  // A compiled object that forgets to release its buffer is not hidden by the recorder.
  const leakingNode = {
    compile: ({device}: {device: Device}) => {
      allocated.push(device.createBuffer({byteLength: 16, usage: Buffer.STORAGE}));
      return {destroy: () => {}};
    }
  };
  getGPUCommandNodeKernelSources([leakingNode]);
  expect(allocated[1].destroyed).toBe(false);
  allocated[1].destroy();
});
