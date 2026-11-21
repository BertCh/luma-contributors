// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {createNullWebGPUDevice} from '../../analysis/test/utils/gpu-contributor-test-utils';
import {makeCatalogRasterInput} from '../src/analysis/catalog-raster-input';
import {runCatalogOperationAsync} from '../src/analysis/catalog-bindings';
import {CatalogGpuSession} from '../src/analysis/catalog-gpu-session';
import {CURATED_DRAINAGE_NETWORK_BINDING} from '../src/curated/bindings/curated-drainage-network-binding';

function makeRaster(shear: number) {
  return makeCatalogRasterInput(
    {
      id: 'metric-control',
      crs: 'planar',
      cellSizeMode: 'uniform',
      grid: {width: 3, height: 3, affine: [1, shear, 0, 0, 1, 0], pixelInterpretation: 'area'},
      band: {
        index: 1,
        values: new Float32Array([9, 8, 7, 6, 5, 4, 3, 2, 1]),
        validity: new Uint8Array(9).fill(1),
        noDataValue: null
      }
    },
    9
  );
}

test('planar raster adapter preserves shear; the terrain binding rejects it before GPU allocation', async () => {
  // Pixel axes (1,0) and (0.5,1) have dot product 0.5, so the orthogonal D8
  // diagonal sqrt(dx^2 + dy^2) is unsupported. The generic raster adapter accepts it.
  const elevation = makeRaster(0.5);
  expect(elevation.grid.affine).toEqual([1, 0.5, 0, 0, 1, 0]);
  const device = createNullWebGPUDevice();
  const allocate = vi.spyOn(device, 'createBuffer');
  try {
    await expect(
      runCatalogOperationAsync(CURATED_DRAINAGE_NETWORK_BINDING, {}, {elevation}, {device})
    ).rejects.toThrow(/elevation.grid.affine must be axis aligned/);
    expect(allocate).not.toHaveBeenCalled();
    // This terrain binding is GPU-only: neither an absent device nor forced CPU can
    // publish the sheared raster through a fallback implementation.
    expect(CURATED_DRAINAGE_NETWORK_BINDING.runData).toBeUndefined();
    expect(CURATED_DRAINAGE_NETWORK_BINDING.run).toBeUndefined();
    for (const options of [{}, {engine: 'cpu' as const}]) {
      await expect(
        runCatalogOperationAsync(CURATED_DRAINAGE_NETWORK_BINDING, {}, {elevation}, options)
      ).rejects.toThrow();
    }
  } finally {
    allocate.mockRestore();
    device.destroy();
  }
});

test('orthogonal terrain control reaches graph submission and does not hide a GPU failure in CPU fallback', async () => {
  const device = createNullWebGPUDevice();
  const submission = vi.spyOn(CatalogGpuSession.prototype, 'run').mockImplementation(() => {
    throw new Error('orthogonal control reached submission');
  });
  try {
    await expect(
      runCatalogOperationAsync(
        CURATED_DRAINAGE_NETWORK_BINDING,
        {},
        {elevation: makeRaster(0)},
        {device}
      )
    ).rejects.toThrow('orthogonal control reached submission');
    expect(submission).toHaveBeenCalledOnce();
  } finally {
    submission.mockRestore();
    device.destroy();
  }
});
