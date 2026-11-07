// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test, vi} from 'vitest';
import {GPUVector} from '@luma-contributors/graph/gpu-data';
import {createResidentHeatmapLayerClasses} from '../../../examples/deck/kepler-compute/src/resident-heatmap-layers';
import {makeResidentHeatmapLayer} from '../../../examples/deck/kepler-compute/src/resident-heatmap-layer';

const calls = vi.hoisted(() => ({render: vi.fn(), format: vi.fn(), domain: vi.fn()}));
vi.mock('../../../examples/deck/kepler-compute/src/resident-heatmap-layer', () => ({
  makeResidentHeatmapLayer: vi.fn((layer, inputs) => ({props: {...layer.props, inputs}}))
}));
vi.mock('@kepler.gl/layers', () => {
  class Heatmap {
    id = 'heatmap';
    meta = {};
    config = {
      dataId: 'resident',
      isVisible: true,
      columnMode: 'points',
      columns: {lat: {value: 'north'}, lng: {value: 'east'}},
      weightField: null as {name: string} | null
    };
    static findDefaultLayerProps() {
      throw new Error('Native geometry discovery must not scan resident sources');
    }
    hasLayerData() {
      return true;
    }
    formatLayerData(...parameters: unknown[]) {
      calls.format(...parameters);
      throw new Error('Native formatting must not scan resident sources');
    }
    updateLayerMeta() {
      throw new Error('Native bounds must not scan resident sources');
    }
    updateLayerDomain(...parameters: unknown[]) {
      calls.domain(...parameters);
      return this;
    }
    renderLayer(options: unknown) {
      calls.render(options);
      return [
        {
          props: {radiusPixels: 17, intensity: 2, threshold: 0.05},
          clone(extra: object) {
            return {props: {...this.props, ...extra}};
          }
        }
      ];
    }
  }
  return {LayerClasses: {heatmap: Heatmap}};
});

function setup() {
  vi.clearAllMocks();
  const binding = {
    positions: new GPUVector({type: 'data', name: 'positions', format: 'float32x2', data: []}),
    selectionMask: new GPUVector({type: 'data', name: 'selection', format: 'uint32', data: []}),
    columns: {lat: 'north', lng: 'east'},
    revision: 3
  };
  const onUnsupported = vi.fn();
  const onRenderError = vi.fn();
  const classes = createResidentHeatmapLayerClasses({
    isResidentDataset: datasetId => datasetId === 'resident',
    resolveHeatmap: () => ({binding}),
    getBounds: () => [-2, -1, 2, 1],
    onUnsupported,
    onRenderError
  });
  const layer = new classes.heatmap({id: 'heatmap', dataId: 'resident'});
  return {layer, classes, binding, onUnsupported, onRenderError};
}

describe('resident heatmap native boundary', () => {
  test('bypasses CPU consumers and retains native configuration before borrowing GPU vectors', () => {
    const {layer, classes, binding} = setup();
    const inaccessible = new Proxy(
      {},
      {
        get: () => {
          throw new Error('Unexpected source access');
        }
      }
    );
    expect(classes.heatmap.findDefaultLayerProps({id: 'resident'} as never)).toEqual({props: []});
    const data = layer.formatLayerData(inaccessible as never, undefined);
    expect(layer.hasLayerData(data as never)).toBe(true);
    layer.updateLayerMeta(inaccessible as never);
    expect(layer.meta['bounds']).toEqual([-2, -1, 2, 1]);
    layer.updateLayerDomain(inaccessible as never);
    expect(calls.format).not.toHaveBeenCalled();
    expect(calls.domain).not.toHaveBeenCalled();
    const rendered = layer.renderLayer({mapState: {zoom: 7}, visible: true} as never)[0];
    expect(calls.render).toHaveBeenCalledWith(
      expect.objectContaining({
        data,
        gpuFilter: null,
        visible: true,
        mapState: {zoom: 7}
      })
    );
    expect(makeResidentHeatmapLayer).toHaveBeenCalledWith(expect.anything(), binding);
    expect(rendered.props).toMatchObject({
      id: 'heatmap-resident-heatmap',
      radiusPixels: 17,
      intensity: 2,
      threshold: 0.05
    });
  });

  test('rejects changed bindings before native rendering', () => {
    const {layer, onUnsupported} = setup();
    layer.config.columns.lat!.value = 'unbound';
    expect(layer.renderLayer({} as never)).toEqual([]);
    expect(onUnsupported).toHaveBeenLastCalledWith('heatmap', expect.stringContaining('binding'));
    layer.config.columns.lat!.value = 'north';
    layer.config.weightField = {name: 'unbound'} as never;
    expect(layer.renderLayer({} as never)).toEqual([]);
    expect(calls.render).not.toHaveBeenCalled();
  });

  test('routes renderer failures to the captured native layer', () => {
    const {layer, onRenderError} = setup();
    const rendered = layer.renderLayer({} as never)[0];
    const error = new Error('Injected heatmap failure');
    expect(rendered.props.onError?.(error)).toBe(true);
    expect(onRenderError).toHaveBeenCalledWith(layer, error);
  });
});
