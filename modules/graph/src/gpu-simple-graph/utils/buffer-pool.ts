// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Device, Buffer} from '@luma.gl/core';

/** A pooled buffer may be at most this many times larger than the requested size. */
const MAXIMUM_REUSE_RATIO = 2;
/** Requests this small may reuse any pooled buffer up to this size regardless of the ratio. */
const MINIMUM_REUSE_BYTE_LENGTH = 256;

export class BufferPool {
  poolSize: number = 20;
  /** Maximum bytes retained per device pool. */
  maxPooledByteLength: number = 256 * 1024 * 1024;

  private bufferPools: Map<Device, Buffer[]>;

  constructor() {
    this.bufferPools = new Map();
  }

  createOrReuse(device: Device, byteLength: number): Buffer {
    if (byteLength > device.limits.maxBufferSize) {
      throw new Error(
        `Buffer pool cannot allocate ${byteLength} bytes: device.limits.maxBufferSize is ${device.limits.maxBufferSize}`
      );
    }

    const pool = this.bufferPools.get(device);
    // The pool is sorted by increasing size, so the first fit is the smallest fit.
    const maximumByteLength = Math.max(byteLength * MAXIMUM_REUSE_RATIO, MINIMUM_REUSE_BYTE_LENGTH);
    const i = pool
      ? pool.findIndex(b => b.byteLength >= byteLength && b.byteLength <= maximumByteLength)
      : -1;
    if (i < 0) {
      return device.createBuffer({
        usage: Buffer.VERTEX | Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC,
        byteLength
      });
    }
    const [result] = pool!.splice(i, 1);
    return result;
  }

  recycle(buffer: Buffer) {
    const device = buffer.device;
    if (!this.bufferPools.has(device)) {
      this.bufferPools.set(device, []);
    }
    const pool = this.bufferPools.get(device)!;
    // Sort buffers by increasing size
    const i = pool.findIndex(b => b.byteLength > buffer.byteLength);
    if (i < 0) {
      pool.push(buffer);
    } else {
      pool.splice(i, 0, buffer);
    }
    // Bound every pool and discard entries for devices that became unusable since the last recycle.
    this.purge();
  }

  /** Destroys the largest cached buffers until every device pool satisfies its count and byte budget. */
  purge(): void {
    for (const [device, pool] of this.bufferPools) {
      const targetSize = device.isLost ? 0 : this.poolSize;
      let pooledByteLength = pool.reduce((total, buffer) => total + buffer.byteLength, 0);
      while (
        pool.length > targetSize ||
        (pool.length > 0 && pooledByteLength > this.maxPooledByteLength)
      ) {
        const evicted = pool.pop()!;
        pooledByteLength -= evicted.byteLength;
        evicted.destroy();
      }
      if (pool.length === 0) {
        this.bufferPools.delete(device);
      }
    }
  }
}

export const bufferPool = new BufferPool();
