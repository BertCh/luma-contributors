// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {expect, test, vi} from 'vitest';
import {makeResidentEditorLayer} from '../../../examples/deck/kepler-compute/src/resident-editor-layer';

type Event = {
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
  const errors: Error[] = [];
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
  const pick = vi.fn(async () =>
    delay
      ? new Promise<{result: {index: number}[]}>(resolve => {
          resolvePick = resolve;
        })
      : {result: [{index: 8}, {index: 3}]}
  );
  const pickTerrain = vi.fn(async () => ({coordinate: [12, 34, 56]}));
  const terrainLayers: {id: string; props: {pickable: string}}[] = [];
  const manager = {
    emit(event: Event) {
      let stopped = false;
      current._forwardEventToCurrentLayer({
        ...event,
        stopImmediatePropagation: () => {
          stopped = true;
        }
      });
      if (!stopped) propagated.push(event.type);
    }
  };
  const context: {deck: object; viewport: typeof viewport; layerManager: object} = {
    deck: {
      eventManager: manager,
      _pickAsync: pick,
      pickObjectAsync: pickTerrain,
      getViewports: () => [context.viewport]
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
      _editableLayerState?: {pointerDownPicks: unknown};
      residentEditorQueue?: {events: Event[]};
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
    getPicks() {
      throw new Error('Synchronous editor picking must not run');
    }
    toBasePointerEvent(_event: Event): Pointer | null {
      return {picks: this.getPicks(), screenCoords: [], mapCoords: []};
    }
    _forwardEventToCurrentLayer(event: Event) {
      const pointer = event.type === 'keyup' ? null : current.toBasePointerEvent(event);
      if (event.type === 'panstart' && pointer)
        current.state._editableLayerState!.pointerDownPicks = pointer.picks;
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
  // The GLSL-only native dash is replaced by the WGSL-capable editor dash extension.
  expect(context.layer.props.extensions.map(extension => extension.constructor.name)).toEqual([
    'EditorWGSLDashExtension'
  ]);
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
