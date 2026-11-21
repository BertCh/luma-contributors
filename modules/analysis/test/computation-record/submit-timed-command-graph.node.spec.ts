// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {expect, test} from 'vitest';
import {submitGPUCommandGraphWithTimings} from '../../src/computation-record/index';

test('DPC-127: the timestamp query set is destroyed when encoder creation throws', () => {
  let destroyCount = 0;
  const device = {
    features: new Set(['timestamp-query']),
    createQuerySet: () => ({
      destroy: () => {
        destroyCount++;
      }
    }),
    createCommandEncoder: () => {
      throw new Error('encoder creation failed');
    }
  } as unknown as Device;
  const compiled = {stats: {nodeOrder: ['a']}} as never;
  expect(() => submitGPUCommandGraphWithTimings(device, compiled, {} as never)).toThrow(
    'encoder creation failed'
  );
  expect(destroyCount).toBe(1);
});
