// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPURasterEmbeddingNearestPixels} from '../../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-nearest-pixels';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';

it('R5 nearest pixels counts eligible support below k and clears it on replay', async () => {
  const device = await getWebGPUTestDevice('core');
  requireTestDevice(device);
  const harness = createParityGraph(device);
  const mask = harness.input('mask', [1, 0, 0], 'uint32');
  const counts = harness.output('counts', 1, 'uint32');
  const rows = harness.output('rows', 2, 'uint32');
  const scores = harness.output('scores', 2, 'float32');
  harness.graph.add(
    new GPURasterEmbeddingNearestPixels({
      pixelCount: 3,
      dimensions: 2,
      queryCount: 1,
      resultCount: 2,
      vectors: harness.input('vectors', [1, 0, 0, 1, -1, 0], 'float32').view,
      queries: harness.input('query', [1, 0], 'float32').view,
      mask: mask.view,
      output: {pixelRows: rows.view, scores: scores.view, resultCounts: counts.view}
    })
  );
  const compiled = harness.graph.compile();
  try {
    // Only pixel 0 is eligible, so min(k=2, support=1)=1; its unit-vector cosine is 1.
    // Masking that last pixel makes support zero, even though the requested k remains two.
    counts.buffer.write(Uint32Array.of(12345));
    submitGraph(device, compiled, undefined);
    expect(await readUint32(counts.buffer, 1)).toEqual([1]);
    expect((await readUint32(rows.buffer, 1))[0]).toBe(0);
    expect((await readFloat32(scores.buffer, 1))[0]).toBe(1);
    mask.buffer.write(Uint32Array.of(0, 0, 0));
    submitGraph(device, compiled, undefined);
    expect(await readUint32(counts.buffer, 1)).toEqual([0]);
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});
