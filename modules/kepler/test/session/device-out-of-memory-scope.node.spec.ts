// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  KeplerDeviceOutOfMemoryError,
  runWithOutOfMemoryScope
} from '../../src/session/device-out-of-memory-scope';
import {isKeplerAllocationError} from '../../src/runtime/dataset-registry';

test('scope catches device-wide OOM while the step awaits unrelated work', async () => {
  let releaseStep!: () => void;
  const stepFinished = new Promise<void>(resolve => {
    releaseStep = resolve;
  });
  let reportedError: {message: string} | null = null;
  const calls: string[] = [];
  const pending = runWithOutOfMemoryScope(
    {
      handle: {
        pushErrorScope(filter: string) {
          calls.push(filter);
        },
        async popErrorScope() {
          calls.push('pop');
          return reportedError;
        }
      }
    },
    async () => {
      await stepFinished;
      return 7;
    }
  );
  expect(calls).toEqual(['out-of-memory']);
  // The raw device scope observes every allocation, regardless of who initiated it.
  reportedError = {message: 'unrelated texture allocation'};
  releaseStep();
  const error = await pending.catch((caught: unknown) => caught);
  expect(isKeplerAllocationError(error)).toBe(true);
  expect(error).toHaveProperty('cause', expect.any(KeplerDeviceOutOfMemoryError));
  expect(calls).toEqual(['out-of-memory', 'pop']);
});

test('the original step error wins even when popping the scope also rejects', async () => {
  const original = new Error('source upload failed');
  let popped = false;
  await expect(
    runWithOutOfMemoryScope(
      {
        handle: {
          pushErrorScope() {},
          async popErrorScope() {
            popped = true;
            throw new Error('device lost');
          }
        }
      },
      () => {
        throw original;
      }
    )
  ).rejects.toBe(original);
  expect(popped).toBe(true);
});
