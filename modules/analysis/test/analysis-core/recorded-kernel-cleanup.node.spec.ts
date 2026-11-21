// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device, type Sampler} from '@luma.gl/core';
import {expect, it} from 'vitest';
import {getGPUCommandNodeKernelSources} from '../../src/testing/gpu-contributor-test-utils';

it('destroys samplers and buffers allocated by a failed node compile (review cleanup note)', () => {
  const samplers: Sampler[] = [];
  const buffers: Buffer[] = [];
  expect(() =>
    getGPUCommandNodeKernelSources([
      {
        compile: ({device}: {device: Device}) => {
          samplers.push(device.createSampler({}));
          buffers.push(device.createBuffer({byteLength: 16, usage: Buffer.STORAGE}));
          throw new Error('failed sampler fixture');
        }
      }
    ])
  ).toThrow('failed sampler fixture');
  expect(buffers).toHaveLength(1);
  expect(samplers).toHaveLength(1);
  expect(buffers[0].destroyed).toBe(true);
  expect(samplers[0].destroyed).toBe(true);
});
