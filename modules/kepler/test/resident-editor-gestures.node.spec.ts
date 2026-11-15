// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {EditableGeoJsonLayer} from '@deck.gl-community/editable-layers';
import {expect, test, vi} from 'vitest';
import {makeResidentEditorLayer} from '../../../examples/deck/kepler-compute/src/resident-editor-layer';

function createViewport(id: string, x: number) {
  return {
    id,
    x,
    y: 0,
    revision: 1,
    containsPixel: (point: {x: number}) => point.x >= x && point.x < x + 500,
    unproject: (point: number[]) => [point[0] + x * 2, point[1]],
    equals(other: {id: string; revision: number}) {
      return this.id === other.id && this.revision === other.revision;
    }
  };
}

function createGestureProbe() {
  type Gesture = {
    type: string;
    offsetCenter: {x: number; y: number};
    srcEvent: object;
    deltaX: number;
    stopImmediatePropagation(): void;
  };
  const events: ({type: string} & Record<string, unknown>)[] = [];
  const errors: Error[] = [];
  const listeners = new Map<string, Set<(event: Gesture) => void>>();
  const eventRoot = {addEventListener() {}, removeEventListener() {}};
  const eventManager = {
    getElement: () => eventRoot,
    on(type: string, handler: (event: Gesture) => void) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type)!.add(handler);
    },
    off(type: string, handler: (event: Gesture) => void) {
      listeners.get(type)?.delete(handler);
    },
    emit(event: Gesture) {
      let stopped = false;
      const dispatched = {...event, stopImmediatePropagation: () => (stopped = true)};
      for (const handler of listeners.get(event.type) ?? []) {
        handler(dispatched);
        if (stopped) break;
      }
    }
  };
  let viewports = [createViewport('left', 0), createViewport('right', 500)];
  let delay = false;
  let releaseRead: (() => void) | undefined;
  const pick = vi.fn(async () => {
    if (delay) {
      delay = false;
      await new Promise<void>(resolve => (releaseRead = resolve));
    }
    return {result: []};
  });
  const context = {
    deck: {
      eventManager,
      getCanvas: () => eventRoot,
      _pickAsync: pick,
      getViewports: () => viewports
    },
    viewport: viewports[0],
    layerManager: {getLayers: () => []}
  };
  // Keep the published native click, pan, press/reset and event-forwarding handlers. Only
  // layer/GPU plumbing and the final edit-mode callbacks are replaced by this probe.
  class NativeEditorProbe extends EditableGeoJsonLayer {
    static override layerName = 'NativeEditorProbe';
    override context = context as unknown as EditableGeoJsonLayer['context'];
    override getCurrentLayer(): this {
      return this;
    }
    override setState(update: Record<string, unknown>): void {
      this.state = Object.assign(this.state ?? {}, update);
    }
    override getScreenCoords(event: MouseEvent): number[] {
      return [event.clientX, event.clientY];
    }
    override raiseError(error: Error): void {
      errors.push(error);
    }
    override onStartDragging(event: Parameters<EditableGeoJsonLayer['onStartDragging']>[0]): void {
      events.push({type: 'panstart', ...event});
    }
    override onDragging(event: Parameters<EditableGeoJsonLayer['onDragging']>[0]): void {
      events.push({type: 'panmove', ...event});
    }
    override onStopDragging(event: Parameters<EditableGeoJsonLayer['onStopDragging']>[0]): void {
      events.push({type: 'panend', ...event});
    }
    override onLayerClick(event: Parameters<EditableGeoJsonLayer['onLayerClick']>[0]): void {
      events.push({type: 'click', ...event});
    }
  }
  const layer = makeResidentEditorLayer(
    new NativeEditorProbe({
      id: 'kepler_editor_layer',
      data: {type: 'FeatureCollection', features: []},
      pickingRadius: 10,
      pickingDepth: 5
    })
  ) as NativeEditorProbe;
  layer.initializeState();
  return {
    events,
    errors,
    pick,
    layer,
    native(type: string, x: number) {
      layer._forwardEventToCurrentLayer({
        type,
        clientX: x,
        clientY: 20,
        button: 0,
        detail: 1,
        isPrimary: true
      } as unknown as MouseEvent);
    },
    gesture(type: string, x: number, deltaX = 0) {
      eventManager.emit({
        type,
        offsetCenter: {x, y: 20},
        srcEvent: {},
        deltaX,
        stopImmediatePropagation() {}
      });
    },
    delayRead() {
      delay = true;
    },
    releaseRead() {
      if (!releaseRead) throw new Error('Probe has no pending readback');
      releaseRead();
    },
    replacePressViewport() {
      viewports = [{...viewports[0], revision: 2}, viewports[1]];
    },
    async settle() {
      await vi.waitFor(() => {
        const queue = Reflect.get(layer.state, 'residentEditorQueue') as {draining: boolean};
        expect(queue.draining).toBe(false);
      });
      expect(errors).toEqual([]);
    }
  };
}

test('published handlers isolate a delayed drag from a later native press and click', async () => {
  const probe = createGestureProbe();
  probe.native('pointerdown', 5);
  probe.delayRead();
  probe.gesture('panstart', 30, 25);
  probe.gesture('panmove', 35);
  probe.gesture('panend', 40);
  probe.native('click', 40); // A click from the dragged press stays ineligible before replay.
  probe.native('pointerdown', 100);
  probe.native('click', 100);
  probe.releaseRead();
  await probe.settle();
  expect(probe.events.map(event => event.type)).toEqual(['panstart', 'panmove', 'panend', 'click']);
  for (const event of probe.events.slice(0, 3)) {
    expect(event).toMatchObject({pointerDownScreenCoords: [5, 20], pointerDownMapCoords: [5, 20]});
  }
  expect(probe.events[3]).toMatchObject({screenCoords: [100, 20]});
});

test('a queued eligible click survives a later press whose drag has already been recognized', async () => {
  const probe = createGestureProbe();
  probe.native('pointerdown', 5);
  probe.delayRead();
  probe.native('click', 5);
  probe.native('pointerdown', 100);
  probe.gesture('panstart', 130, 30);
  probe.gesture('panend', 140);
  probe.releaseRead();
  await probe.settle();
  expect(probe.events.map(event => event.type)).toEqual(['click', 'panstart', 'panend']);
  expect(probe.events[0]).toMatchObject({screenCoords: [5, 20]});
  expect(probe.events[1]).toMatchObject({
    pointerDownScreenCoords: [100, 20],
    pointerDownMapCoords: [100, 20]
  });
});

test('successive recognizer-only drags each retain their own delta-derived press', async () => {
  const probe = createGestureProbe();
  probe.delayRead();
  probe.gesture('panstart', 30, 25);
  probe.gesture('panend', 40);
  probe.gesture('panstart', 130, 30);
  probe.gesture('panend', 140);
  probe.releaseRead();
  await probe.settle();
  expect(probe.events.map(event => event.type)).toEqual([
    'panstart',
    'panend',
    'panstart',
    'panend'
  ]);
  expect(probe.events[0]).toMatchObject({pointerDownScreenCoords: [5, 20]});
  expect(probe.events[2]).toMatchObject({pointerDownScreenCoords: [100, 20]});
});

test('published panstart unprojects its original press in the left view after crossing into the right view', async () => {
  const probe = createGestureProbe();
  probe.native('pointerdown', 499);
  probe.gesture('panstart', 504, 5);
  await probe.settle();
  expect(probe.events).toMatchObject([
    {
      type: 'panstart',
      screenCoords: [504, 20],
      mapCoords: [1004, 20],
      pointerDownScreenCoords: [499, 20],
      pointerDownMapCoords: [499, 20]
    }
  ]);
});

test('press-view changes during readback cancel a crossing drag even when its moved-pointer view is unchanged', async () => {
  const probe = createGestureProbe();
  probe.native('pointerdown', 499);
  probe.delayRead();
  probe.gesture('panstart', 504, 5);
  probe.replacePressViewport();
  probe.releaseRead();
  await probe.settle();
  expect(probe.events).toEqual([]);
});

test('press-view changes during the original-press readback also cancel a crossing drag', async () => {
  const probe = createGestureProbe();
  let releasePressRead: (() => void) | undefined;
  probe.pick.mockResolvedValueOnce({result: []}).mockImplementationOnce(async () => {
    await new Promise<void>(resolve => (releasePressRead = resolve));
    return {result: []};
  });
  probe.native('pointerdown', 499);
  probe.gesture('panstart', 504, 5);
  await vi.waitFor(() => expect(probe.pick).toHaveBeenCalledTimes(2));
  probe.replacePressViewport();
  if (!releasePressRead) throw new Error('Probe has no pending original-press readback');
  releasePressRead();
  await probe.settle();
  expect(probe.events).toEqual([]);
});
