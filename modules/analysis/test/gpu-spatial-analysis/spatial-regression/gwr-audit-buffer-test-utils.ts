// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';

/** Initialized fixture storage that tests subsequently read back, including in-place outputs. */
export function createReadableGWRAuditBuffer(
  device: Device,
  values: Float32Array | Uint32Array
): Buffer {
  return device.createBuffer({
    data: values,
    usage: Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC
  });
}
