// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Deck, type PickingInfo} from '@deck.gl/core';
import {expect, test, vi} from 'vitest';

// This is a contract test of the pinned yarn patch, deliberately loading its installed dist
// file (not a replacement picker). DeckPicker has no public export. The app does not import it.
const pickerUrl = new URL('./lib/deck-picker.js', import.meta.resolve('@deck.gl/core'));
const {default: DeckPicker} = await import(pickerUrl.href);

for (const method of ['pickObject', 'pickObjectAsync']) {
  test(`${method} notifies the previous layer when canvas leave supplies no viewport`, async () => {
    const layerHover = vi.fn(() => false);
    const globalHover = vi.fn();
    const layer = {
      id: 'resident-points',
      props: {id: 'resident-points'},
      getPickingInfo: ({info}: {info: PickingInfo}) => info,
      updateAutoHighlight: vi.fn(),
      onHover: layerHover
    };
    // No GPU resources are touched by the no-viewport branch. Keep every other picker method
    // real; attempting to draw or allocate would fail with this intentionally minimal device.
    const picker = Object.create(DeckPicker.prototype);
    Object.assign(picker, {
      device: {getDefaultCanvasContext: () => ({cssToDeviceRatio: () => 1})},
      _getPickable: () => [layer],
      lastPickedInfo: {layerId: layer.id, index: 7, info: {layer, index: 7, picked: true}}
    });
    const result = await picker[method]({
      layers: [layer],
      viewports: [],
      views: {},
      x: -1,
      y: -1,
      mode: 'hover'
    });
    expect(result.result).toHaveLength(1);
    expect(result.result[0]).toMatchObject({layer, picked: false, index: -1, x: -1, y: -1});
    expect(picker.lastPickedInfo).toEqual({layerId: null, index: -1, info: null});
    const deck = {
      widgetManager: {onHover: vi.fn()},
      cursorState: {isHovering: true},
      props: {onHover: globalHover}
    };
    const applyHover = Reflect.get(Deck.prototype, '_applyHoverCallbacks');
    const event = {type: 'pointerleave'};
    applyHover.call(deck, result, event);
    expect(layerHover).toHaveBeenCalledWith(result.result[0], event);
    expect(globalHover).toHaveBeenCalledWith(result.result[0], event);
    expect(deck.cursorState.isHovering).toBe(false);
  });
}
