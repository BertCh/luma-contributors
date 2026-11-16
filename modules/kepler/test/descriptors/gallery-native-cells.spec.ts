// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {test} from 'vitest';
import {requireTestDevice} from '../../../analysis/test/utils/require-test-device';
import {
  getCatalogBinding,
  KEPLER_CATALOG_BINDINGS,
  runCatalogOperationAsync
} from '../../src/analysis/catalog-bindings';
import {suggestKeplerAnalyses} from '../../src/descriptors/applicability';
import {profileGallerySample, readGalleryFeatures} from './gallery-sample-profiles';

test('actual H3 boundaries buffer on the GPU with dissolve off and retain source rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const suggestion = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, [
    profileGallerySample('h3-cells')
  ])
    .suggestions.slice(0, 3)
    .find(entry => entry.bindingId === 'geometry-polygon-buffer');
  if (!suggestion) throw new Error('H3 gallery: polygon buffer missing from top three');
  const result = await runCatalogOperationAsync(
    getCatalogBinding(suggestion.bindingId)!,
    {...suggestion.applicability.parameters, dissolve: false},
    {polygons: readGalleryFeatures('h3-cells')},
    {device}
  );
  const buffers = result.tables.find(table => table.id === 'buffers');
  const sourceRows = [...new Set(buffers?.columns['source_index'] ?? [])].sort(
    (left, right) => Number(left) - Number(right)
  );
  if (
    result.engine !== 'gpu' ||
    !result.dispatchedContributors.includes('GPUBufferSurface') ||
    !buffers ||
    buffers.rowCount < 4 ||
    JSON.stringify(sourceRows) !== JSON.stringify([1, 2, 4, 5]) ||
    !buffers.columns['area']?.every(
      area => typeof area === 'number' && Number.isFinite(area) && area > 0
    ) ||
    result.diagnostics.some(diagnostic => diagnostic.severity === 'error')
  ) {
    throw new Error(
      `H3 buffer mismatch: expected GPUBufferSurface, positive-area buffers and source rows [1,2,4,5]; got ${JSON.stringify({engine: result.engine, dispatched: result.dispatchedContributors, rows: buffers?.rowCount, sourceRows, areas: buffers?.columns['area'], diagnostics: result.diagnostics})}`
    );
  }
});
