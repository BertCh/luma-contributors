// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterEach, expect, test, vi} from 'vitest';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../analysis/test/utils/gpu-contributor-test-utils';
import {CatalogGpuSession} from '../src/analysis/catalog-gpu-session';
import {runKeplerRasterEmbedding} from '../src/analysis/raster-embeddings';

afterEach(() => vi.restoreAllMocks());

test.each([
  false,
  true
])('K-means publishes labels only when GPU convergence is %s', async converged => {
  const device = createNullWebGPUDevice();
  const status = Uint32Array.of(32, converged ? 0 : 10, converged ? 1 : 0);
  const add = vi.spyOn(GPUCommandGraph.prototype, 'add').mockImplementation(() => {});
  vi.spyOn(CatalogGpuSession.prototype, 'run').mockImplementation(() => {});
  const read = vi
    .spyOn(CatalogGpuSession.prototype, 'readUint32')
    .mockImplementation(async output =>
      output.buffer.id.endsWith('-kmeans-status') ? status : new Uint32Array(output.length)
    );
  try {
    const result = runKeplerRasterEmbedding(
      device,
      {
        width: 256,
        height: 1,
        dimensions: 1,
        vectors: Float32Array.from({length: 256}, (_, index) => index)
      },
      {operation: 'kmeans', clusterCount: 8}
    );
    if (converged) expect((await result).columns['label']).toHaveLength(256);
    else
      await expect(result).rejects.toThrow(
        /K-means did not converge after 32 iterations \(10 changed labels\)/
      );
    expect(add.mock.calls[0][0]).toMatchObject({props: {output: {status: expect.anything()}}});
    expect(read.mock.calls.some(([output]) => output.buffer.id.endsWith('-kmeans-status'))).toBe(
      true
    );
  } finally {
    device.destroy();
  }
});
