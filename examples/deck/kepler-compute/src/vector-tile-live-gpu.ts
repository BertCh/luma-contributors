// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {VectorTileGpuBuffer} from './vector-tile-live-controller';

/** Buffer factory for decoded tile positions (vertex) and filter masks (storage). */
export function makeVectorTileBufferFactory(
  device: Device
): (data: Float32Array | Uint32Array) => VectorTileGpuBuffer {
  return data => {
    const buffer = device.createBuffer({
      id: 'kepler-vector-tile-live',
      data: data.byteLength ? data : new Uint32Array(1),
      usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    return {
      byteLength: buffer.byteLength,
      write: values => buffer.write(values),
      destroy: () => buffer.destroy()
    };
  };
}
