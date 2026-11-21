// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {getRasterEmbeddingStretchRanges} from '@luma-contributors/analysis/raster-analysis';
import {createNullWebGPUDevice} from '../../analysis/test/utils/gpu-contributor-test-utils';
import {CatalogGpuSession} from '../src/analysis/catalog-gpu-session';
import {runKeplerRasterEmbedding} from '../src/analysis/raster-embeddings';

test.each([
  1, 0
])('PCA adapter packs public stretch ranges for eigenvalues [%s,0,0]', async firstEigenvalue => {
  // Three observations (-1,0,0), (0,0,0), (1,0,0) have unbiased covariance
  // diag(1,0,0). Two-sigma spread is 2; zero components borrow that spread.
  // An all-zero covariance has no spread to borrow and uses radius 1.
  const radius = firstEigenvalue ? 2 : 1;
  const expected = [
    [-radius, radius],
    [-radius, radius],
    [-radius, radius]
  ];
  expect(getRasterEmbeddingStretchRanges([firstEigenvalue, 0, 0])).toEqual(expected);
  const device = createNullWebGPUDevice();
  const parameters: number[][] = [];
  const allocate = device.createBuffer.bind(device);
  vi.spyOn(device, 'createBuffer').mockImplementation(properties => {
    if (
      !(properties instanceof ArrayBuffer) &&
      !ArrayBuffer.isView(properties) &&
      properties.id === 'kepler-embedding-projection-parameters' &&
      properties.data instanceof Float32Array
    )
      parameters.push(Array.from(properties.data));
    return allocate(properties);
  });
  vi.spyOn(CatalogGpuSession.prototype, 'run').mockImplementation(() => {});
  vi.spyOn(CatalogGpuSession.prototype, 'readScalar').mockResolvedValue(3);
  vi.spyOn(CatalogGpuSession.prototype, 'readFloat32').mockImplementation(async output => {
    const values = new Float32Array(output.length);
    if (output.buffer.id.endsWith('-covariance')) values[0] = firstEigenvalue;
    return values;
  });
  try {
    await runKeplerRasterEmbedding(
      device,
      {
        width: 3,
        height: 1,
        dimensions: 3,
        vectors: new Float32Array([-firstEigenvalue, 0, 0, 0, 0, 0, firstEigenvalue, 0, 0])
      },
      {operation: 'pca'}
    );
    expect(parameters).toEqual([expected.flat()]);
    const packed = parameters[0];
    for (let channel = 0; channel < 3; channel++) {
      const low = packed[channel * 2];
      const high = packed[channel * 2 + 1];
      const grey = (0 - low) / (high - low);
      expect(Number.isFinite(grey)).toBe(true);
      expect(grey).toBe(0.5);
    }
  } finally {
    vi.restoreAllMocks();
    device.destroy();
  }
});
