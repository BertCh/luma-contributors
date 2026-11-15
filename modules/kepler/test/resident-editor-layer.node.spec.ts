// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {expect, test, vi} from 'vitest';
import {makeResidentEditorLayer} from '../../../examples/deck/kepler-compute/src/resident-editor-layer';

type GestureEvent = {
  type: string;
  offsetCenter: {x: number; y: number};
  srcEvent: object;
  stopImmediatePropagation(): void;
};
type Pointer = {picks: {index: number}[]; screenCoords: number[]; mapCoords: number[]};

function fixture(cancelPan = false) {
  let current: NativeEditor;
  const calls: {type: string; pointer: Pointer | null; pointerDown: unknown}[] = [];
  const propagated: string[] = [];
  const mapGuards: string[] = [];
  const errors: Error[] = [];
  const deckClicks: Promise<unknown>[] = [];
  const deckClickHandlers = new Set<(event: GestureEvent) => void>();
  let resolvePick: ((value: {result: {index: number}[]}) => void) | undefined;
  let delay = false;
  const viewport = {
    revision: 1,
    x: 0,
    y: 0,
    containsPixel: () => true,
    unproject: (point: number[]) => point.map(value => value / 10),
    equals(other: {revision: number}) {
      return this.revision === other.revision;
    }
  };
  const pick = vi.fn(async (_method: string, _statistics: string, _options: {x: number}) =>
    delay
      ? new Promise<{result: {index: number}[]}>(resolve => {
          resolvePick = resolve;
        })
      : {result: [{index: 8}, {index: 3}]}
  );
  const pickTerrain = vi.fn(async () => ({coordinate: [12, 34, 56]}));
  const terrainLayers: {id: string; props: {pickable: string}}[] = [];
  const manager = {
    on(_type: string, handler: (event: GestureEvent) => void) {
      deckClickHandlers.add(handler);
    },
    off(_type: string, handler: (event: GestureEvent) => void) {
      deckClickHandlers.delete(handler);
    },
    emit(event: GestureEvent) {
      let stopped = false;
      const dispatched = {
        ...event,
        stopImmediatePropagation: () => {
          stopped = true;
        }
      };
      if (event.type === 'click') {
        for (const handler of deckClickHandlers) handler(dispatched);
      }
      if (!stopped) current._forwardEventToCurrentLayer(dispatched);
      if (!stopped) {
        propagated.push(event.type);
        if (event.type === 'click') {
          const selected = Reflect.get(context.deck, '_lastPointerDownInfoPromise');
          if (selected) deckClicks.push(selected);
        }
      }
    }
  };
  const context: {deck: object; viewport: typeof viewport; layerManager: object} = {
    deck: {
      eventManager: manager,
      _pickAsync: pick,
      pickObjectAsync: pickTerrain,
      getViewports: () => [context.viewport],
      _lastPointerDownInfoPromise: Promise.resolve({index: -1, layer: null})
    },
    viewport,
    layerManager: {getLayers: () => terrainLayers}
  };
  class NativeEditor {
    props: {
      id: string;
      data: object;
      mode: object;
      modeConfig: object;
      extensions: unknown[];
      pickingDepth: number;
      pickingRadius: number;
    };
    id: string;
    state: {
      _editableLayerState?: {pointerDownPicks: unknown; pressScreenCoords?: number[]};
      residentEditorQueue?: {events: (GestureEvent | MouseEvent)[]};
    } = {};
    context = context;
    finalized = false;
    constructor(...properties: Partial<NativeEditor['props']>[]) {
      this.props = Object.assign(
        {
          id: 'kepler_editor_layer',
          data: {},
          mode: {},
          modeConfig: {screenSpace: true},
          extensions: ['native-dash'],
          pickingDepth: 5,
          pickingRadius: 10
        },
        ...properties
      );
      this.id = this.props.id;
    }
    initializeState() {
      this.state._editableLayerState = {pointerDownPicks: null};
    }
    finalizeState() {
      this.finalized = true;
    }
    getCurrentLayer() {
      return current;
    }
    setState(value: object) {
      Object.assign(this.state, value);
    }
    raiseError(error: Error) {
      errors.push(error);
    }
    clone(properties: object) {
      return new NativeEditor(this.props, properties);
    }
    getPicks(): {index: number}[] {
      throw new Error('Synchronous editor picking must not run');
    }
    getScreenCoords(event: MouseEvent) {
      return [event.clientX - 100, event.clientY - 200];
    }
    getMapCoords(_position: number[]): number[] | null {
      throw new Error('Synchronous terrain picking must not run');
    }
    toBasePointerEvent(_event: GestureEvent | MouseEvent): Pointer | null {
      return {picks: this.getPicks(), screenCoords: [], mapCoords: []};
    }
    _ownsPointerDown(event: MouseEvent): boolean {
      // The v10 ownership hook calls pointer conversion outside recognized event dispatch.
      return Boolean(current.toBasePointerEvent(event));
    }
    _onNativeMapInteraction(event: MouseEvent) {
      if (event.type === 'mousedown') current._ownsPointerDown(event);
      mapGuards.push(event.type);
    }
    _forwardEventToCurrentLayer(event: GestureEvent | MouseEvent) {
      // Match editable-layers 10: native clicks are not subscribed through Mjolnir, and
      // DOM event fields live on the prototype. Pointer-down/map guards run immediately.
      if (!('srcEvent' in event)) {
        if (event.type === 'pointerdown') {
          current.state._editableLayerState!.pressScreenCoords = this.getScreenCoords(event);
          return;
        }
        if (event.type === 'mousedown') {
          current._onNativeMapInteraction(event);
          return;
        }
        if (event.type !== 'click' && event.type !== 'dblclick') return;
        if (event.button !== 0 || (event.type === 'click' && event.detail > 1)) return;
      }
      const pointer = event.type === 'keyup' ? null : current.toBasePointerEvent(event);
      if (event.type === 'panstart' && pointer) {
        // v10 resolves press picks/coordinates separately from the moved pointer.
        current.getMapCoords(
          current.state._editableLayerState!.pressScreenCoords ?? pointer.screenCoords
        );
        current.state._editableLayerState!.pointerDownPicks = current.getPicks();
      }
      calls.push({
        type: event.type,
        pointer,
        pointerDown: current.state._editableLayerState!.pointerDownPicks
      });
      if (cancelPan && ['panstart', 'panmove'].includes(event.type))
        event.stopImmediatePropagation();
      if (event.type === 'panend') current.state._editableLayerState!.pointerDownPicks = null;
    }
  }
  const original = new NativeEditor();
  current = makeResidentEditorLayer(original as unknown as Layer) as unknown as NativeEditor;
  current.initializeState();
  return {
    original,
    layer: current,
    calls,
    propagated,
    mapGuards,
    deckClicks,
    deckClickHandlers,
    context,
    errors,
    pick,
    pickTerrain,
    terrainLayers,
    viewport,
    event(type: string, x = 10) {
      const stop = vi.fn();
      const event = {type, offsetCenter: {x, y: 20}, srcEvent: {}, stopImmediatePropagation: stop};
      current._forwardEventToCurrentLayer(event);
      expect(stop).toHaveBeenCalledOnce();
    },
    recognizedClick() {
      manager.emit({
        type: 'click',
        offsetCenter: {x: 10, y: 20},
        srcEvent: {},
        stopImmediatePropagation() {}
      });
    },
    nativeEvent(type: string, button = 0, detail = 1, x = 110) {
      const event = Object.create({
        get type() {
          return type;
        },
        get clientX() {
          return x;
        },
        get clientY() {
          return 220;
        },
        get button() {
          return button;
        },
        get detail() {
          return detail;
        }
      }) as MouseEvent;
      current._forwardEventToCurrentLayer(event);
      return event;
    },
    delay() {
      delay = true;
    },
    finish() {
      delay = false;
      resolvePick!({result: [{index: 1}, {index: 2}]});
    },
    replace() {
      const next = makeResidentEditorLayer(
        new NativeEditor({
          ...current.props,
          modeConfig: {...current.props.modeConfig}
        }) as unknown as Layer
      ) as unknown as NativeEditor;
      next.state = current.state;
      current = next;
      return next;
    }
  };
}

async function settled() {
  for (let index = 0; index < 12; index++) await Promise.resolve();
}

test('native editor keeps constructor and props, resolves all depth hits, and never synchronously picks', async () => {
  const context = fixture();
  expect(context.layer).toBeInstanceOf(context.original.constructor);
  expect(context.layer.props.mode).toBe(context.original.props.mode);
  expect(context.layer.props.data).toBe(context.original.props.data);
  // deck.gl 10 ships WGSL for PathStyleExtension, so the editor dash is that extension as is.
  expect(
    context.layer.props.extensions.map(extension => (extension as object).constructor.name)
  ).toEqual(['PathStyleExtension', 'ResidentEditorPickingExtension']);
  context.event('click');
  expect(context.calls).toEqual([]);
  expect(context.propagated).toEqual([]);
  await settled();
  expect(context.pick).toHaveBeenCalledWith('pickObjectAsync', 'pickMultipleObjects Time', {
    x: 10,
    y: 20,
    layerIds: ['kepler_editor_layer'],
    radius: 10,
    depth: 5
  });
  expect(context.calls[0].pointer).toMatchObject({
    picks: [{index: 8}, {index: 3}],
    screenCoords: [10, 20],
    mapCoords: [1, 2]
  });
  expect(context.propagated).toEqual(['click']);
  expect(context.errors).toEqual([]);
});

for (const cancelPan of [true, false]) {
  test(`native drag propagation and pointer-down handles survive async replay (cancel=${cancelPan})`, async () => {
    const context = fixture(cancelPan);
    context.event('panstart');
    context.event('panmove', 30);
    context.event('panend', 40);
    await settled();
    expect(context.calls.map(call => call.type)).toEqual(['panstart', 'panmove', 'panend']);
    expect(context.calls[1].pointerDown).toEqual([{index: 8}, {index: 3}]);
    expect(context.layer.state._editableLayerState!.pointerDownPicks).toBeNull();
    expect(context.propagated).toEqual(cancelPan ? ['panend'] : ['panstart', 'panmove', 'panend']);
  });
}

test('pending picks survive equivalent native clones but never publish after source, mode or viewport changes', async () => {
  const clone = fixture();
  clone.delay();
  clone.event('click');
  const current = clone.replace();
  clone.finish();
  await settled();
  expect(clone.calls).toHaveLength(1);
  expect(current.state).toBe(clone.layer.state);
  for (const change of ['source', 'mode', 'viewport']) {
    const context = fixture();
    context.delay();
    context.event('click');
    if (change === 'source') context.layer.props.data = {};
    if (change === 'mode') context.layer.props.mode = {};
    if (change === 'viewport') context.layer.context.viewport = {...context.viewport, revision: 2};
    context.finish();
    await settled();
    expect(context.calls).toEqual([]);
    expect(context.propagated).toEqual([]);
  }
});

test('queued pointer moves coalesce, discrete events retain order, and finalization cancels pending publication', async () => {
  const context = fixture();
  context.delay();
  context.event('pointermove', 1);
  for (let index = 2; index < 100; index++) context.event('pointermove', index);
  context.event('click', 100);
  context.event('keyup');
  context.finish();
  await settled();
  expect(context.calls.map(call => [call.type, call.pointer?.screenCoords[0]])).toEqual([
    ['pointermove', 1],
    ['pointermove', 99],
    ['click', 100],
    ['keyup', undefined]
  ]);
  expect(context.errors).toEqual([]);
  const disposed = fixture();
  disposed.delay();
  disposed.event('click');
  disposed.layer.finalizeState();
  disposed.finish();
  await settled();
  expect(disposed.layer.finalized).toBe(true);
  expect(disposed.calls).toEqual([]);
});

test('terrain coordinate lookup uses public asynchronous picking and keeps editable coordinates 2D', async () => {
  const context = fixture();
  context.terrainLayers.push({id: 'terrain', props: {pickable: '3d'}});
  context.event('click');
  await settled();
  expect(context.pickTerrain).toHaveBeenCalledWith({
    x: 10,
    y: 20,
    layerIds: ['terrain'],
    unproject3D: true
  });
  expect(context.calls[0].pointer?.mapCoords).toEqual([12, 34]);
});

test('missing pinned async seams and bounded discrete queue pressure produce explicit errors', async () => {
  const context = fixture();
  context.delay();
  context.event('click');
  for (let index = 0; index < 70; index++) context.event('click');
  expect(context.layer.state.residentEditorQueue!.events).toHaveLength(64);
  expect(context.errors[0].message).toMatch(/bounded capacity/);
  context.layer.finalizeState();
  context.finish();
  await settled();
  expect(() => makeResidentEditorLayer({constructor: Object} as unknown as Layer)).toThrow(
    /editable-layer event seam/
  );
});

test('failed final drag pick clears native handles and lets controllers end their pan', async () => {
  const context = fixture();
  context.event('panstart');
  await settled();
  context.pick.mockRejectedValueOnce(new Error('readback failed'));
  context.event('panend');
  await settled();
  expect(context.calls.at(-1)?.type).toBe('panend');
  expect(context.calls.at(-1)?.pointer).toBeNull();
  expect(context.layer.state._editableLayerState!.pointerDownPicks).toBeNull();
  expect(context.propagated).toEqual(['panstart', 'panend']);
  expect(context.errors[0].message).toBe('readback failed');
});

test('queue overflow retains bounded terminal drag cleanup without a final pick or stale edit', async () => {
  const context = fixture();
  context.event('panstart');
  await settled();
  expect(context.layer.state._editableLayerState!.pointerDownPicks).not.toBeNull();
  context.delay();
  context.event('panmove');
  for (let index = 0; index < 64; index++) context.event('click');
  context.event('panend');
  expect(context.layer.state.residentEditorQueue!.events).toHaveLength(64);
  expect(context.layer.state.residentEditorQueue!.events.at(-1)?.type).toBe('panend');
  expect(context.errors.at(-1)?.message).toMatch(/bounded capacity/);
  context.finish();
  await vi.waitFor(() => expect(context.calls.at(-1)?.type).toBe('panend'));
  expect(context.calls.at(-1)?.pointer).toBeNull();
  expect(context.layer.state._editableLayerState!.pointerDownPicks).toBeNull();
  expect(context.propagated.at(-1)).toBe('panend');
  expect(context.pick).toHaveBeenCalledTimes(65);
});

test('v10 DOM clicks retain prototype coordinates and dispatch through native handlers exactly once', async () => {
  const context = fixture();
  context.nativeEvent('pointerdown');
  expect(context.layer.state._editableLayerState!.pressScreenCoords).toEqual([10, 20]);
  context.nativeEvent('mousedown');
  expect(context.pick).not.toHaveBeenCalled();
  context.nativeEvent('click');
  await settled();
  expect(context.calls).toHaveLength(1);
  expect(context.calls[0]).toMatchObject({
    type: 'click',
    pointer: {screenCoords: [10, 20], mapCoords: [1, 2]}
  });
  expect(context.propagated).toEqual([]);
  context.nativeEvent('click', 0, 2);
  context.nativeEvent('dblclick', 0, 2);
  expect(context.mapGuards).toEqual(['mousedown', 'dblclick']);
  context.nativeEvent('click', 2);
  await settled();
  expect(context.calls.map(call => call.type)).toEqual(['click', 'dblclick']);
  expect(context.errors).toEqual([]);
});

test('v10 panstart uses async picks at the original native press rather than a synchronous picker', async () => {
  const context = fixture();
  context.nativeEvent('pointerdown', 0, 1, 105);
  context.event('panstart', 30);
  await settled();
  expect(context.pick.mock.calls.map(call => call[2].x)).toEqual([30, 5]);
  expect(context.calls[0]).toMatchObject({
    type: 'panstart',
    pointer: {screenCoords: [30, 20]},
    pointerDown: [{index: 8}, {index: 3}]
  });
  expect(context.errors).toEqual([]);
});

test('native mousedown ownership avoids synchronous pointer conversion, while async pan ownership still reaches Deck', async () => {
  for (const ownsPan of [true, false]) {
    const context = fixture(ownsPan);
    expect(() => context.nativeEvent('mousedown')).not.toThrow();
    expect(context.mapGuards).toEqual(['mousedown']);
    expect(context.pick).not.toHaveBeenCalled();
    context.nativeEvent('pointerdown');
    context.event('panstart');
    await settled();
    expect(context.propagated).toEqual(ownsPan ? [] : ['panstart']);
    expect(context.errors).toEqual([]);
  }
});

test('recognized selection clicks use the editor multi-pick, skip duplicate native edits, and remove their listener on finalize', async () => {
  const context = fixture();
  const editorPick = {index: 0, layer: {id: 'kepler_editor_layer'}, object: {id: 'polygon'}};
  context.pick.mockResolvedValue({result: [editorPick]});
  const originalPromise = Reflect.get(context.context.deck, '_lastPointerDownInfoPromise');
  context.recognizedClick();
  expect(context.deckClicks).toEqual([]);
  await settled();
  expect(context.deckClicks).toHaveLength(1);
  expect(await context.deckClicks[0]).toEqual(editorPick);
  expect(Reflect.get(context.context.deck, '_lastPointerDownInfoPromise')).toBe(originalPromise);
  expect(context.calls).toEqual([]);
  expect(context.deckClickHandlers.size).toBe(1);
  context.nativeEvent('click');
  await settled();
  expect(context.calls.map(call => call.type)).toEqual(['click']);
  context.layer.finalizeState();
  expect(context.deckClickHandlers.size).toBe(0);
  expect(context.errors).toEqual([]);
});
