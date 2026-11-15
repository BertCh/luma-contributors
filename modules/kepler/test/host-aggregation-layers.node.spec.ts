// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterAll, expect, test, vi} from 'vitest';

// Node uses isolate:false: discard previously evaluated consumers before applying this file's mocks.
vi.hoisted(() => vi.resetModules());
import {createResidentAggregationLayerClasses} from '../../../examples/deck/kepler-compute/src/resident-aggregation-layers';

// Exercise the adapter's native render gate without loading Kepler's WebGL bootstrap in Node.
vi.mock('@kepler.gl/utils', () => ({
  DataRow: class {
    constructor(
      readonly _dataContainer: unknown,
      readonly _rowIndex: number
    ) {}
  }
}));

vi.mock('@kepler.gl/layers', () => {
  class NativeLayer {
    id = 'aggregate';
    config = {dataId: 'resident', isVisible: true, visConfig: {opacity: 1}};
    meta = {};
    hasLayerData(layerData: {data?: unknown[]}) {
      return Boolean(layerData?.data?.length);
    }
    formatLayerData() {
      return {data: ['native']};
    }
    updateLayerMeta() {}
    getHoverData() {
      return 'native';
    }
    renderLayer() {
      return ['native'];
    }
  }
  return {LayerClasses: {grid: NativeLayer, hexagon: NativeLayer, cluster: NativeLayer}};
});

test('native aggregate overrides admit only the resident sentinel and honor split-map visibility', () => {
  const resolveAggregation = vi.fn(() => ({reason: 'Unsupported native control'}));
  const onUnsupported = vi.fn();
  const classes = createResidentAggregationLayerClasses({
    isResidentDataset: datasetId => datasetId === 'resident',
    resolveAggregation,
    onUnsupported,
    getBounds: () => [-75, 39, -73, 41]
  });
  for (const Constructor of [classes.grid, classes.hexagon]) {
    const layer = new Constructor({id: 'aggregate'});
    const sentinel = layer.formatLayerData({}, {}, undefined);
    expect(layer.hasLayerData(sentinel)).toBe(true);
    expect(layer.hasLayerData({residentAggregation: true})).toBe(false);
    expect(layer.hasLayerData({data: [1]})).toBe(false);
    expect(layer.renderLayer({visible: false})).toEqual([]);
    expect(resolveAggregation).not.toHaveBeenCalled();
    expect(layer.renderLayer({visible: true})).toEqual([]);
    expect(onUnsupported).toHaveBeenCalledWith('aggregate', 'Unsupported native control');
    resolveAggregation.mockClear();
    layer.config.dataId = 'ordinary';
    expect(layer.formatLayerData({}, {}, undefined)).toEqual({data: ['native']});
    expect(layer.hasLayerData({data: [1]})).toBe(true);
    expect(layer.renderLayer()).toEqual(['native']);
  }
});

// Remove mocked dependencies and their consumers before the next shared-worker spec.
afterAll(() => {
  vi.doUnmock('@kepler.gl/utils');
  vi.doUnmock('@kepler.gl/layers');
  vi.resetModules();
});
