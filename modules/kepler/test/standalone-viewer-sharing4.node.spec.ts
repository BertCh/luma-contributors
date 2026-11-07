// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {readViewerConfig} from '../../../examples/deck/kepler-compute/src/standalone-viewer-config';
import {
  createCustomColorScale,
  createCustomOrdinalColorScale
} from '../../../examples/deck/kepler-compute/src/standalone-viewer-color-map';
import {
  applyPolygonFilters,
  getPolygonFiltersForLayer
} from '../../../examples/deck/kepler-compute/src/standalone-viewer-polygon';
import {decodeCellFeatures} from '../../../examples/deck/kepler-compute/src/standalone-viewer-cells';

test('custom color maps', () => {
  const threshold = createCustomColorScale([
    [10, '#ff0000'],
    [20, '#00ff00'],
    [null, '#0000ff']
  ]);
  expect(threshold.getColor(5)).toEqual([255, 0, 0]);
  expect(threshold.getColor(15)).toEqual([0, 255, 0]);
  expect(threshold.getColor(99)).toEqual([0, 0, 255]);
  const ordinal = createCustomOrdinalColorScale([
    [['a', 'b'], '#ff0000'],
    [null, '#0000ff']
  ]);
  expect(ordinal.getCategoryColor('b')).toEqual([255, 0, 0]);
  expect(ordinal.getCategoryColor('z')).toEqual([0, 0, 255]);
});

test('polygon filters and split/basemap config', () => {
  const square = {
    type: 'Feature',
    geometry: {
      type: 'Polygon',
      coordinates: [
        [
          [0, 0],
          [10, 0],
          [10, 10],
          [0, 10],
          [0, 0]
        ]
      ]
    }
  };
  const config = readViewerConfig({
    config: {
      visState: {
        filters: [
          {id: 'f', dataId: ['d'], name: [], type: 'polygon', layerId: ['L'], value: square}
        ],
        splitMaps: [{layers: {L: true}}, {layers: {L: false}}]
      },
      mapStyle: {styleType: 'x', mapStyles: {x: {url: 'https://example.com/style.json'}}}
    }
  });
  expect(config.splitMapLayers).toEqual([{L: true}, {L: false}]);
  expect(config.basemapStyleUrl).toBe('https://example.com/style.json');
  const filters = getPolygonFiltersForLayer(config.filters, 'L', 'd');
  expect(filters.length).toBe(1);
  const points: [number, number][][] = [[[5, 5]], [[50, 5]]];
  expect([...applyPolygonFilters(Uint32Array.of(0, 1), filters, row => points[row])]).toEqual([0]);
});

test('geohash cells', () => {
  const dataset: any = {readStrings: () => ['u4pruyd', 'bad!']};
  const features = decodeCellFeatures('geohash', dataset, 'token', Uint32Array.of(0, 1));
  expect(features.length).toBe(1);
  expect(features[0].geometry.coordinates[0].length).toBe(5);
});
