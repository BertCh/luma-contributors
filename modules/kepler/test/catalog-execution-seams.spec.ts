// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../analysis/src/testing/require-test-device';
import {getCatalogBinding, runCatalogOperationAsync} from '../src/analysis/catalog-bindings';

test('PCA and projection report the CPU eigensolver separately from GPU dispatches', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const operation of ['pca', 'projection']) {
    const result = await runCatalogOperationAsync(
      getCatalogBinding(`raster-embedding-${operation}`)!,
      {},
      {},
      {device}
    );
    expect(result.dispatchedContributors).toEqual([
      'GPURasterEmbeddingCovariance',
      'GPURasterEmbeddingProjection'
    ]);
    expect(result.stages?.map(stage => [stage.identity.exportName, stage.execution])).toEqual([
      ['GPURasterEmbeddingCovariance', 'gpu'],
      ['getRasterEmbeddingPrincipalComponents', 'cpu'],
      ['GPURasterEmbeddingProjection', 'gpu']
    ]);
    expect(result.tables[0].rowCount).toBe(64);
  }
});

test('ICP maximumIterations sets the executed history length', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Identical tetrahedra have the identity source-to-target transform.
  const cloud = {
    type: 'FeatureCollection' as const,
    features: [
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1]
    ].map(coordinates => ({geometry: {type: 'Point' as const, coordinates}, properties: {}}))
  };
  const result = await runCatalogOperationAsync(
    getCatalogBinding('pointcloud-icp')!,
    {coordinateSystem: 'planar', maximumDistance: 2, maximumIterations: 3},
    {source: cloud, target: cloud},
    {device}
  );
  expect(result.dispatchedContributors).toEqual(['GPUPointCloudICP']);
  const history = result.tables.find(table => table.id === 'registration')!;
  expect(history.rowCount).toBe(3);
  expect(history.columns['iteration']).toEqual([0, 1, 2]);
  // Every GPU-written history row must retain all four matches, including frozen rows.
  // A host-generated length or an identity transform alone cannot prove the budget ran.
  expect(history.columns['matched_count']).toEqual([4, 4, 4]);
  expect(result.summary['registrationStatus']).toBe(0); // Public ICP code 0 means converged.
  expect(history.columns['registration_status']).toEqual([0, 0, 0]);
  for (let component = 0; component < 16; component++) {
    const value = history.columns[`transform_${component}`][0];
    expect(Number.isFinite(value)).toBe(true);
    expect(value).toBeCloseTo(component % 5 === 0 ? 1 : 0, 5);
  }
});
