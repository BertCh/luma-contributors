// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {GPUCommandGraphEncoding} from '../../src/gpu-core/gpu-command-graph';

function createTimedNode(id: string, readResults: () => Promise<BigInt64Array>) {
  return {
    stats: {id, type: 'compute'},
    timestamp: {
      querySet: {device: {type: 'webgpu'}, readResults},
      beginIndex: 0,
      endIndex: 1
    }
  };
}

describe('GPUCommandGraphEncoding.readTimings', () => {
  it('reports per-node errors without discarding successful timings', async () => {
    const good = createTimedNode('good', async () => BigInt64Array.from([1000000n, 3000000n]));
    const bad = createTimedNode('bad', async () => {
      throw new Error('device lost');
    });
    const encoding = new GPUCommandGraphEncoding([good, bad] as never, 0);
    const report = await encoding.readTimings();
    expect(report.nodes[0].gpuTimeMilliseconds).toBe(2);
    expect(report.nodes[1].timingError).toBe('device lost');
    expect(report.gpuTimeMilliseconds).toBe(2);
  });
});
