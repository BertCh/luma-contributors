// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../src/testing/require-test-device';

it('requires a working WebGPU device for local headless verification', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const buffer = device.createBuffer({
    data: new Uint32Array([17]),
    usage: Buffer.COPY_SRC | Buffer.COPY_DST
  });
  try {
    const bytes = await buffer.readAsync();
    // readAsync returns bytes; reinterpret their storage instead of widening each byte.
    const values = new Uint32Array(
      bytes.buffer,
      bytes.byteOffset,
      bytes.byteLength / Uint32Array.BYTES_PER_ELEMENT
    );
    expect(values).toEqual(new Uint32Array([17]));
  } finally {
    buffer.destroy();
  }
});
