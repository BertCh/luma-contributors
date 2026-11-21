// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../analysis/src/testing/require-test-device';
import {getCatalogBinding, getCatalogDefaultParameters} from '../src/analysis/catalog-bindings';

it('the isopleth binding preserves a singleton bridge zero raster and empty evidence (W5-14)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const binding = getCatalogBinding('trajectory-isopleths')!;
  const result = await binding.runGpu!(
    {device},
    {
      tracks: {
        type: 'FeatureCollection',
        features: [
          {
            geometry: {type: 'Point', coordinates: [7, 9]},
            properties: {track_id: 'one', timestamp: 0}
          }
        ]
      }
    },
    {
      ...getCatalogDefaultParameters(binding),
      coordinateSystem: 'planar',
      gridWidth: 2,
      gridHeight: 3,
      padding: 1
    }
  );
  // No consecutive fixes means zero bridge mass; no isopleth cell can be inside.
  expect(result.tables[0].rowCount).toBe(0);
  expect(result.summary['empty']).toBe(1);
  expect(result.summary['gridWidth']).toBe(2);
  expect(result.summary['gridHeight']).toBe(3);
  expect(result.summary['cellCount']).toBe(0);
});
