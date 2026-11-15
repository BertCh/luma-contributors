// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import type {PickingInfo} from '@deck.gl/core';

// Exercise the installed DeckPicker, including its real async depth loop and draw/readback
// ordering. Only GPU operations are replaced. The app does not import this private module.
const pickerUrl = new URL('./lib/deck-picker.js', import.meta.resolve('@deck.gl/core'));
const {default: DeckPicker} = await import(pickerUrl.href);

function createPickingFixture() {
  function createLayer(id: string) {
    const layer = {
      id,
      props: {id, data: [{}]},
      current: true,
      getCurrentLayer() {
        return this.current ? this : null;
      },
      getPickingInfo: ({info}: {info: PickingInfo}) => ({...info, object: {id}}),
      disablePickingIndex: vi.fn(),
      restorePickingColors: vi.fn(),
      writeUniforms: vi.fn(() => {
        // WebGPUBuffer.destroyHandle clears its handle. A subsequent UniformStore upload
        // therefore fails in queue.writeBuffer, exactly as the driver's guide-layer toast.
        if (!layer.current)
          throw new TypeError('writeBuffer: parameter 1 is not of type GPUBuffer');
      })
    };
    return layer;
  }
  const stroke = createLayer('kepler_editor_layer-geojson-polygons-stroke');
  const guide = createLayer('kepler_editor_layer-guides-points-circle');
  const layers = [stroke, guide];
  const viewport = {x: 0, y: 0, unproject: (point: number[]) => point};
  const canvasContext = {
    cssToDeviceRatio: () => 1,
    cssToDevicePixels: () => ({x: 0, y: 0, width: 1, height: 1})
  };
  const render = vi.fn(({layers: drawnLayers}: {layers: typeof layers}) => {
    for (const layer of drawnLayers) layer.writeUniforms();
    return {
      stats: {},
      decodePickingColor: () => ({pickedLayer: stroke, pickedObjectIndex: 0})
    };
  });
  let reads = 0;
  const read = vi.fn(async () => {
    // Simulate the guides disappearing during the first GPU readback. Deck's query still
    // holds its original array while the layer manager has finalized this primitive.
    guide.current = false;
    return ++reads === 1 ? new Uint8Array([1, 0, 0, 1]) : new Uint8Array(4);
  });
  const picker = Object.create(DeckPicker.prototype);
  Object.assign(picker, {
    device: {getDefaultCanvasContext: () => canvasContext},
    pickingFBO: {width: 1, height: 1, colorAttachments: [{texture: {}}]},
    lastPickedInfo: {layerId: null, index: -1, info: null},
    pickLayersPass: {render},
    _getPickable: () => layers,
    _resizeBuffer() {},
    _updateStats() {},
    _readTextureDataAsync: read
  });
  return {
    picker,
    stroke,
    guide,
    render,
    read,
    options: {layers, viewports: [viewport], views: {}, effects: [], x: 0, y: 0, depth: 2}
  };
}

test('async editor multi-picking never redraws guides finalized during the first readback', async () => {
  const fixture = createPickingFixture();
  const result = await fixture.picker.pickObjectAsync(fixture.options);
  expect(result.result).toMatchObject([{index: 0, object: {id: fixture.stroke.id}}]);
  expect(fixture.render).toHaveBeenCalledTimes(2);
  expect(fixture.guide.writeUniforms).toHaveBeenCalledOnce();
  expect(fixture.stroke.writeUniforms).toHaveBeenCalledTimes(2);
  expect(fixture.stroke.disablePickingIndex).toHaveBeenCalledWith(0);
  expect(fixture.stroke.restorePickingColors).toHaveBeenCalledOnce();
});

test('async picking drops a picked guide finalized during readback before changing its attributes', async () => {
  const fixture = createPickingFixture();
  fixture.render.mockImplementation(({layers}) => {
    for (const layer of layers) layer.writeUniforms();
    return {
      stats: {},
      decodePickingColor: () => ({pickedLayer: fixture.guide, pickedObjectIndex: 0})
    };
  });
  const result = await fixture.picker.pickObjectAsync(fixture.options);
  expect(result.result).toEqual([]);
  expect(fixture.render).toHaveBeenCalledOnce();
  expect(fixture.guide.disablePickingIndex).not.toHaveBeenCalled();
  expect(fixture.guide.restorePickingColors).not.toHaveBeenCalled();
});

test('async picking resolves matched guide instances before the next draw', async () => {
  const fixture = createPickingFixture();
  const replacement = {...fixture.guide, current: true, writeUniforms: vi.fn()};
  replacement.getCurrentLayer = () => replacement;
  fixture.guide.getCurrentLayer = () => (fixture.guide.current ? fixture.guide : replacement);
  await fixture.picker.pickObjectAsync(fixture.options);
  expect(fixture.render.mock.calls[1][0].layers).toEqual([fixture.stroke, replacement]);
  expect(fixture.guide.writeUniforms).toHaveBeenCalledOnce();
  expect(replacement.writeUniforms).toHaveBeenCalledOnce();
});

test('async picking does not restore attributes on a previously picked guide finalized during a later readback', async () => {
  const fixture = createPickingFixture();
  fixture.render.mockImplementation(({layers}) => {
    for (const layer of layers) layer.writeUniforms();
    return {
      stats: {},
      decodePickingColor: () => ({pickedLayer: fixture.guide, pickedObjectIndex: 0})
    };
  });
  let reads = 0;
  fixture.read.mockImplementation(async () => {
    if (++reads === 1) return new Uint8Array([1, 0, 0, 1]);
    fixture.guide.current = false;
    return new Uint8Array(4);
  });
  await fixture.picker.pickObjectAsync(fixture.options);
  expect(fixture.guide.disablePickingIndex).toHaveBeenCalledWith(0);
  expect(fixture.guide.restorePickingColors).not.toHaveBeenCalled();
});

test('async picking rechecks a hit after the separate depth readback', async () => {
  const fixture = createPickingFixture();
  fixture.picker.depthFBO = fixture.picker.pickingFBO;
  let reads = 0;
  fixture.read.mockImplementation(async () => {
    if (++reads === 2) fixture.stroke.current = false;
    return new Uint8Array([1, 0, 0, 1]);
  });
  const result = await fixture.picker.pickObjectAsync({...fixture.options, unproject3D: true});
  expect(result.result).toEqual([]);
  expect(fixture.render).toHaveBeenCalledTimes(2);
  expect(fixture.stroke.disablePickingIndex).not.toHaveBeenCalled();
  expect(fixture.stroke.restorePickingColors).not.toHaveBeenCalled();
});
