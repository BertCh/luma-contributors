// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  findPointColumns,
  getViewStateForBounds
} from '../../../examples/deck/kepler-compute/src/standalone-viewer-model';

test('findPointColumns prefers Kepler layer config then column names', () => {
  const config = {
    visState: {layers: [{config: {columns: {lat: {value: 'a'}, lng: {value: 'b'}}}}]}
  };
  expect(findPointColumns(['a', 'b', 'lat', 'lng'], config)).toEqual({
    longitude: 'b',
    latitude: 'a'
  });
  expect(findPointColumns(['Longitude', 'Latitude'], null)).toEqual({
    longitude: 'Longitude',
    latitude: 'Latitude'
  });
  expect(findPointColumns(['q'], null)).toBeNull();
});

test('getViewStateForBounds centers and clamps zoom', () => {
  const view = getViewStateForBounds([-10, 0, 10, 20], 800, 600);
  expect(view.longitude).toBe(0);
  expect(view.latitude).toBe(10);
  expect(view.zoom).toBeGreaterThan(0);
  expect(getViewStateForBounds([0, 0, 0, 0], 800, 600).zoom).toBe(20);
});
