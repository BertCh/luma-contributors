// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Vitest scans its test entries before running them. Make the iframe's public dependencies
// visible to that scan so opening the iframe cannot trigger dependency optimization/reloads.
// These imports create no device. The example itself runs only in the independent iframe,
// which still verifies that its adapter registry starts empty.
import '@luma.gl/core';
import '@luma.gl/core/webgpu';
import '@luma.gl/graph';
import '@luma-contributors/analysis/gpu-spatial-analysis';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../src/testing/require-test-device';

type FirstRunMessage = {
  type: 'first-density-result';
  baselineAdapterMissing: boolean;
  missingAdapterRejected: boolean;
  displayed: [number[], number][];
  error?: string;
};

it('executes both displayed fences in a fresh browser realm without a prepared device', async () => {
  requireTestDevice(navigator.gpu);
  const frame = document.createElement('iframe');
  frame.src = '/modules/analysis/test/analysis-core/point-density-first-run.html';
  let timeout: ReturnType<typeof setTimeout> | undefined;
  let receive: (event: MessageEvent) => void = () => {};
  try {
    const result = new Promise<FirstRunMessage>((resolve, reject) => {
      receive = event => {
        if (event.source === frame.contentWindow && event.data?.type === 'first-density-result') {
          resolve(event.data);
        }
      };
      window.addEventListener('message', receive);
      timeout = setTimeout(() => reject(new Error('Fresh first-run frame did not finish')), 30_000);
    });
    document.body.append(frame);
    const message = await result;
    expect(message.error).toBeUndefined();
    expect(message.baselineAdapterMissing).toBe(true);
    expect(message.missingAdapterRejected).toBe(true);
    expect(message.displayed).toEqual([[[2, 0, 1, 1], 0]]);
  } finally {
    clearTimeout(timeout);
    window.removeEventListener('message', receive);
    frame.remove();
  }
});
