// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {NullDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {BufferPool} from '../../src/gpu-simple-graph/utils/buffer-pool';

test('BufferPool does not serve small requests from much larger buffers', () => {
  const device = new NullDevice({});
  const pool = new BufferPool();
  const large = pool.createOrReuse(device, 1024 * 1024);
  pool.recycle(large);

  const small = pool.createOrReuse(device, 4096);
  expect(small).not.toBe(large);
  expect(pool.createOrReuse(device, 700 * 1024)).toBe(large);
  device.destroy();
});

test('BufferPool evicts the largest buffers first', () => {
  const device = new NullDevice({});
  const pool = new BufferPool();
  pool.poolSize = 2;
  const buffers = [4096, 8192, 1024 * 1024].map(byteLength =>
    pool.createOrReuse(device, byteLength)
  );
  for (const buffer of buffers) pool.recycle(buffer);

  expect(buffers[2].destroyed).toBe(true);
  expect(buffers[0].destroyed).toBe(false);
  expect(buffers[1].destroyed).toBe(false);
  pool.poolSize = 0;
  pool.purge();
  device.destroy();
});

test('BufferPool enforces a byte budget', () => {
  const device = new NullDevice({});
  const pool = new BufferPool();
  pool.maxPooledByteLength = 10000;
  const small = pool.createOrReuse(device, 4096);
  const large = pool.createOrReuse(device, 8192);
  pool.recycle(small);
  pool.recycle(large);
  expect(large.destroyed).toBe(true);
  expect(small.destroyed).toBe(false);
  pool.poolSize = 0;
  pool.purge();
  device.destroy();
});
