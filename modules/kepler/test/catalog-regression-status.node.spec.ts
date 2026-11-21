// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {createNullWebGPUDevice} from '../../analysis/test/utils/gpu-contributor-test-utils';
import {getCatalogBinding, normalizeCatalogParameters} from '../src/analysis/catalog-bindings';
import {CatalogGpuSession} from '../src/analysis/catalog-gpu-session';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';

test.each([
  'regression-gtwr',
  'regression-mgtwr'
])('%s counts singular and excluded rows with unknown status ignored', async id => {
  // Producer census: GTWR and MGTWR both emit GWR's 0=OK, 1=SINGULAR, 2=EXCLUDED.
  // Two singular rows and three excluded rows; unknown 99 contributes to neither count.
  const words = new Uint32Array([0, 1, 1, 2, 2, 2, 99]);
  const points: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: Array.from(words, (_, index) => ({
      id: `observation-${index}`,
      properties: {observed: index, predictor: index, time: index},
      geometry: {type: 'Point', coordinates: [index, index % 2]}
    }))
  };
  const device = createNullWebGPUDevice();
  vi.spyOn(CatalogGpuSession.prototype, 'run').mockImplementation(() => {});
  vi.spyOn(CatalogGpuSession.prototype, 'readUint32').mockResolvedValue(words);
  vi.spyOn(CatalogGpuSession.prototype, 'readFloat32').mockImplementation(
    async output => new Float32Array(output.length)
  );
  try {
    const binding = getCatalogBinding(id)!;
    const output = await binding.runGpu!(
      {device},
      {points},
      normalizeCatalogParameters(binding, {coordinateSystem: 'planar'})
    );
    const diagnostics = output.tables.find(table => table.id === `${id}-diagnostics`)!;
    const readMetric = (name: string) =>
      diagnostics.columns['value'][Array.from(diagnostics.columns['metric']).indexOf(name)];
    expect(readMetric('singular_rows')).toBe(2);
    expect(readMetric('excluded_rows')).toBe(3);
    expect(Array.from(output.tables[0].columns['status'])).toEqual(Array.from(words));
  } finally {
    vi.restoreAllMocks();
    device.destroy();
  }
});
