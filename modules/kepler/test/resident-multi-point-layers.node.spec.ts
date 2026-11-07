// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  prepareNativePolygonFilter,
  prepareResidentPolygonFilter
} from '../../../examples/deck/kepler-compute/src/resident-table-binding';
import {
  listPointSelectorLayerIds,
  planPointLayerSelector,
  type PointSelectorLayer
} from '../../../examples/deck/kepler-compute/src/resident-point-layer-selectors';
import type {KeplerComputeFilter} from '../../../examples/deck/kepler-compute/src/kepler-table-bridge';

function makeLayer(
  id: string,
  longitude: string,
  latitude: string
): PointSelectorLayer & {
  config: {dataId: string};
} {
  return {
    id,
    type: 'point',
    config: {
      dataId: 'observations',
      isVisible: true,
      columns: {lng: {value: longitude}, lat: {value: latitude}}
    }
  };
}

const ring = [
  [0, 0],
  [1, 0],
  [1, 1],
  [0, 0]
];
function makeFilter(layerIds: string[]): KeplerComputeFilter {
  return {
    id: 'polygon',
    type: 'polygon',
    dataId: ['observations'],
    layerId: layerIds,
    value: {type: 'Feature', geometry: {type: 'Polygon', coordinates: [ring]}, properties: {}}
  };
}

const source = {
  longitude: 'lng',
  latitude: 'lat',
  positions: {} as never,
  revision: 3
};

describe('multiple point layers per resident dataset', () => {
  test('each duplicated layer id gets its own selector over shared positions', () => {
    const first = planPointLayerSelector(makeLayer('a', 'lng', 'lat'), source);
    const second = planPointLayerSelector(makeLayer('a-copy', 'lng', 'lat'), source);
    expect(first.status === 'selector' && first.selector.id).toBe('a');
    expect(second.status === 'selector' && second.selector.id).toBe('a-copy');
    if (first.status === 'selector' && second.status === 'selector')
      expect(first.selector.geometry).toEqual(second.selector.geometry);
  });

  test('a layer with another lng/lat mapping is reported instead of borrowing positions', () => {
    const plan = planPointLayerSelector(makeLayer('other', 'lng2', 'lat2'), source);
    expect(plan.status).toBe('mismatch');
    expect(
      listPointSelectorLayerIds(
        [makeLayer('a', 'lng', 'lat'), makeLayer('other', 'lng2', 'lat2')],
        'observations',
        source
      )
    ).toEqual(['a']);
  });

  test('native polygon targets resolve per layer id and ignore untargeted duplicates', () => {
    const layers = [makeLayer('a', 'lng', 'lat'), makeLayer('a-copy', 'lng', 'lat')];
    const prepared = prepareNativePolygonFilter(makeFilter(['a-copy']), 'observations', layers, [
      'a',
      'a-copy'
    ]);
    expect(prepared?.layerIds).toEqual(['a-copy']);
  });

  test('the single-mask binding accepts duplicates only when all share the filter and mapping', () => {
    const layers = [makeLayer('a', 'lng', 'lat'), makeLayer('a-copy', 'lng', 'lat')];
    expect(
      prepareResidentPolygonFilter(makeFilter(['a', 'a-copy']), 'observations', 'a', layers)
        .layerIds
    ).toEqual(['a', 'a-copy']);
    expect(() =>
      prepareResidentPolygonFilter(makeFilter(['a']), 'observations', 'a', layers)
    ).toThrow('separate per-layer');
    expect(() =>
      prepareResidentPolygonFilter(makeFilter(['a', 'other']), 'observations', 'a', [
        layers[0],
        makeLayer('other', 'lng2', 'lat2')
      ])
    ).toThrow('different lng/lat');
  });
});
