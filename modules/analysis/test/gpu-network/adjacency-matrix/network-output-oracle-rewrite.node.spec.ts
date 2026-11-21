// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {expect, it, vi} from 'vitest';
import {createOracleContext} from './network-output-oracle-test-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('W5-03 writes float32 vector replays as float bytes and preserves uint32', () => {
  const device = createNullWebGPUDevice();
  const created = vi.spyOn(device, 'createBuffer');
  const context = createOracleContext(device);
  try {
    for (const format of ['float32x2', 'float32', 'uint32x2', 'uint32'] as const) {
      const values = format.startsWith('float32') ? [0.25, -1.5] : [7, 0xffffffff];
      const view = context.input(format, format, [0, 0]);
      const buffer = created.mock.results.at(-1)!.value as Buffer;
      const writes = vi.spyOn(buffer, 'write');
      context.write(view, values);
      const data = writes.mock.calls[0][0];
      expect(data).toBeInstanceOf(format.startsWith('float32') ? Float32Array : Uint32Array);
      expect(Array.from(data as Float32Array | Uint32Array)).toEqual(values);
    }
  } finally {
    context.destroy();
    created.mockRestore();
    device.destroy();
  }
});
