// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {readViewerConfig} from '../../../examples/deck/kepler-compute/src/standalone-viewer-config';
import {selectFilteredRows} from '../../../examples/deck/kepler-compute/src/standalone-viewer-filters';
import {
  createContinuousColorScale,
  createOrdinalColorScale,
  createSizeScale
} from '../../../examples/deck/kepler-compute/src/standalone-viewer-scales';

const saved = {
  version: 'v1',
  config: {
    visState: {
      layers: [
        {id: 'b', type: 'arc', config: {dataId: 'd', columns: {lat0: {value: 'a'}}, visConfig: {}}},
        {
          id: 'a',
          type: 'point',
          config: {
            dataId: 'd',
            isVisible: false,
            columns: {lat: {value: 'lat'}, lng: {value: 'lng'}},
            visConfig: {}
          },
          visualChannels: {colorField: {name: 'v', type: 'real'}, colorScale: 'quantile'}
        }
      ],
      filters: [{id: 'f', dataId: ['d'], name: ['v'], type: 'range', value: [1, 2]}],
      interactionConfig: {tooltip: {fieldsToShow: {d: [{name: 'v'}]}}}
    },
    mapState: {zoom: 3}
  }
};

test('readViewerConfig reverses draw order and reads channels, filters, tooltips', () => {
  const config = readViewerConfig(saved);
  expect(config.layers.map(layer => layer.id)).toEqual(['a', 'b']);
  expect(config.layers[0].isVisible).toBe(false);
  expect(config.layers[0].channels['color'].field?.name).toBe('v');
  expect(config.filters[0].datasetIds).toEqual(['d']);
  expect(config.tooltipFields['d']).toEqual(['v']);
  expect(config.mapState?.zoom).toBe(3);
});

test('selectFilteredRows applies range and multiSelect filters', () => {
  const readers = {
    rowCount: 4,
    readNumbers: () => Float64Array.from([0, 1, 2, 3]),
    readStrings: () => ['x', 'y', 'x', null]
  };
  const range = {id: 'r', datasetIds: ['d'], fieldNames: ['v'], type: 'range', value: [1, 2]};
  const multi = {id: 'm', datasetIds: ['d'], fieldNames: ['s'], type: 'multiSelect', value: ['x']};
  expect(Array.from(selectFilteredRows('d', [range], readers))).toEqual([1, 2]);
  expect(Array.from(selectFilteredRows('d', [range, multi], readers))).toEqual([2]);
  expect(Array.from(selectFilteredRows('other', [range], readers))).toEqual([0, 1, 2, 3]);
});

test('color and size scales', () => {
  const colors: [number, number, number][] = [
    [0, 0, 0],
    [100, 100, 100],
    [200, 200, 200]
  ];
  const quantize = createContinuousColorScale('quantize', colors, [0, 3, 6, 9]);
  expect(quantize.getColor(0)).toEqual([0, 0, 0]);
  expect(quantize.getColor(9)).toEqual([200, 200, 200]);
  const linear = createContinuousColorScale('linear', colors, [0, 10]);
  expect(linear.getColor(5)).toEqual([100, 100, 100]);
  const ordinal = createOrdinalColorScale(['b', 'a', null], colors);
  expect(ordinal.categories).toEqual(['a', 'b']);
  expect(ordinal.getCategoryColor('b')).toEqual([100, 100, 100]);
  expect(createSizeScale('linear', [0, 10], [2, 12]).getSize(5)).toBe(7);
});
