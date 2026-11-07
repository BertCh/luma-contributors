// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Deck, Layer, PickingInfo, Viewport} from '@deck.gl/core';
import type {EditableGeoJsonLayer} from '@deck.gl-community/editable-layers';
import type {EventManager, MjolnirEvent, MjolnirGestureEvent} from 'mjolnir.js';
import {disableEditorDashExtension, makeEditorDashExtensions} from './editor-dash-extension';

type PointerEvent = ReturnType<EditableGeoJsonLayer['toBasePointerEvent']>;
type EditorQueue = {
  events: MjolnirEvent[];
  draining: boolean;
  disposed: boolean;
  invalidDrag: boolean;
};
type Replay = {queue: EditorQueue; pointer: PointerEvent};
const REPLAY = Symbol('resident-editor-replay');
const CANCEL_DRAG = Symbol('resident-editor-overflow-panend');
const MAXIMUM_PENDING_EVENTS = 64;
const factories = new Map<typeof EditableGeoJsonLayer, (layer: Layer) => Layer>();
const adaptedConstructors = new Set<Function>();

/**
 * Pinned EditableLayer 9.3.8 performs synchronous multi-picking before every pointer handler.
 * Retain its actual constructor/modes but suspend event dispatch until WebGPU picking completes.
 * Mjolnir's public emit() creates a fresh propagation scope: native cancelPan still runs before
 * lower-priority map controllers, rather than attempting to stop an already-dispatched event.
 */
export function makeResidentEditorLayer(nativeLayer: Layer): Layer {
  if (adaptedConstructors.has(nativeLayer.constructor))
    return nativeLayer.clone({extensions: makeEditorDashExtensions()});
  for (const method of ['_forwardEventToCurrentLayer', 'toBasePointerEvent', 'getPicks']) {
    if (typeof Reflect.get(nativeLayer, method) !== 'function')
      throw new Error('Unsupported native editable-layer event seam');
  }
  // Kepler's CJS editor may have a different identity from the ESM export. Keep that exact class.
  const NativeEditor = nativeLayer.constructor as typeof EditableGeoJsonLayer;
  let factory = factories.get(NativeEditor);
  if (!factory) {
    class ResidentEditorLayer extends NativeEditor {
      static override layerName = 'ResidentAsyncEditorLayer';
      declare state: EditableGeoJsonLayer['state'] & {residentEditorQueue?: EditorQueue};

      override initializeState(): void {
        getEventManager(this.context.deck);
        requireAsyncPicker(this.context.deck);
        super.initializeState();
        this.setState({
          residentEditorQueue: {events: [], draining: false, disposed: false, invalidDrag: false}
        });
      }

      override raiseError(error: Error, message?: string): void {
        // A dash shader that fails to build must not take the editor down: later layers use solid strokes.
        if (/shader|pipeline|wgsl|dash/i.test(`${message ?? ''} ${error.message}`)) {
          disableEditorDashExtension(error.message);
        }
        super.raiseError(error, message ?? '');
      }

      override finalizeState(): void {
        const queue = this.state.residentEditorQueue;
        if (queue) {
          queue.disposed = true;
          queue.events.length = 0;
        }
        super.finalizeState();
      }

      override _forwardEventToCurrentLayer(event: MjolnirEvent): void {
        const current = this.getCurrentLayer() as ResidentEditorLayer | null;
        const queue = current?.state.residentEditorQueue;
        if (!current || !queue || queue.disposed) return;
        const replay = readReplay(event);
        if (replay) {
          if (replay.queue === queue) super._forwardEventToCurrentLayer(event);
          return;
        }
        // Stop now; fresh propagation functions are supplied during the asynchronous replay.
        event.stopImmediatePropagation();
        const captured = {
          ...event,
          ...('offsetCenter' in event ? {offsetCenter: {...event.offsetCenter}} : {})
        };
        const previous = queue.events.at(-1);
        // Moves sample the newest pointer location while a read is pending; start/end/click order
        // is retained. This bounds pointer backlog without replaying obsolete hover positions.
        if (
          (event.type === 'pointermove' || event.type === 'panmove') &&
          previous?.type === event.type
        ) {
          queue.events[queue.events.length - 1] = captured;
        } else if (queue.events.length < MAXIMUM_PENDING_EVENTS) {
          queue.events.push(captured);
        } else {
          if (event.type === 'panend') {
            // A terminal event must not be lost behind input backlog. Replace the last queued
            // event without growing the bound; null-pointer replay clears handles/controllers
            // without committing a final edit from an overloaded gesture.
            Reflect.set(captured, CANCEL_DRAG, true);
            queue.events[queue.events.length - 1] = captured;
          }
          current.raiseError(
            new Error('Native editor input queue exceeded its bounded capacity'),
            'resident editor events'
          );
          return;
        }
        if (!queue.draining) void current.drainResidentEditorEvents(queue);
      }

      override toBasePointerEvent(event: MjolnirGestureEvent): PointerEvent {
        const replay = readReplay(event);
        if (!replay || replay.queue !== this.state.residentEditorQueue)
          throw new Error('Native editor pointer events require asynchronous dispatch');
        return replay.pointer;
      }

      private async drainResidentEditorEvents(queue: EditorQueue): Promise<void> {
        queue.draining = true;
        try {
          while (queue.events.length && !queue.disposed) {
            const event = queue.events.shift()!;
            const current = this.getCurrentLayer() as ResidentEditorLayer | null;
            if (!current || current.state.residentEditorQueue !== queue) break;
            let replayStarted = false;
            try {
              if (event.type === 'panstart') queue.invalidDrag = false;
              const viewport = getEventViewport(current, event);
              const source = current.props.data;
              const mode = current.props.mode;
              const modeConfig = current.props.modeConfig;
              const pointer =
                event.type === 'keyup' ||
                Reflect.get(event, CANCEL_DRAG) === true ||
                (queue.invalidDrag && event.type.startsWith('pan'))
                  ? null
                  : await preparePointerEvent(current, event);
              const latest = this.getCurrentLayer() as ResidentEditorLayer | null;
              if (queue.disposed || !latest || latest.state.residentEditorQueue !== queue) continue;
              if (
                latest.props.data !== source ||
                latest.props.mode !== mode ||
                !sameModeConfig(latest.props.modeConfig, modeConfig) ||
                !getEventViewport(latest, event).equals(viewport)
              ) {
                if (event.type.startsWith('pan')) queue.invalidDrag = true;
                if (event.type === 'panend') {
                  replayStarted = true;
                  replayEditorEvent(latest, event, queue, null);
                  queue.invalidDrag = false;
                }
                continue;
              }
              // Do not retain picked objects across source/mode/viewport changes. The native
              // layer supplies its own pointerDownPicks and edit-handle lifecycle during replay.
              replayStarted = true;
              replayEditorEvent(latest, event, queue, pointer);
              if (event.type === 'panend') queue.invalidDrag = false;
            } catch (error) {
              if (event.type.startsWith('pan')) queue.invalidDrag = true;
              if (!queue.disposed && !replayStarted && event.type === 'panend') {
                // Finish an already-started map pan and clear native pointerDown handles even
                // when the final pick fails. A null pointer cannot commit an obsolete edit.
                const latest = this.getCurrentLayer() as ResidentEditorLayer | null;
                if (latest?.state.residentEditorQueue === queue) {
                  replayEditorEvent(latest, event, queue, null);
                  queue.invalidDrag = false;
                }
              }
              if (!queue.disposed)
                current.raiseError(
                  error instanceof Error ? error : new Error(String(error)),
                  'resident asynchronous editor picking'
                );
            }
          }
        } finally {
          queue.draining = false;
        }
      }
    }
    adaptedConstructors.add(ResidentEditorLayer);
    factory = layer =>
      new ResidentEditorLayer((layer as EditableGeoJsonLayer).props, {
        extensions: makeEditorDashExtensions()
      });
    factories.set(NativeEditor, factory);
  }
  return factory(nativeLayer);
}

function replayEditorEvent(
  layer: EditableGeoJsonLayer,
  event: MjolnirEvent,
  queue: EditorQueue,
  pointer: PointerEvent
): void {
  const replayEvent = {...event, [REPLAY]: {queue, pointer}};
  const deck = layer.context.deck;
  const pointerDownHolder = deck as unknown as
    | {_lastPointerDownInfoPromise?: Promise<PickingInfo> | null}
    | undefined;
  const originalPromise = pointerDownHolder?._lastPointerDownInfoPromise;
  const editorPick =
    event.type === 'click' && pointer
      ? (pointer.picks as unknown as PickingInfo[]).find(
          pick => pick.layer?.id === layer.id && pick.object
        )
      : undefined;
  if (pointerDownHolder && editorPick) {
    // Deck dispatches a click with the object picked at pointer-down, which may be a layer above
    // the editor in WebGPU draw order. Native selection needs the editor feature that the
    // multi-pick already found under the pointer, unless pointer-down already found one.
    pointerDownHolder._lastPointerDownInfoPromise = Promise.resolve(
      originalPromise ?? editorPick
    ).then(info => (info?.layer?.id === layer.id && info.object ? info : editorPick));
  }
  try {
    getEventManager(deck).emit(replayEvent);
  } finally {
    if (pointerDownHolder && editorPick)
      pointerDownHolder._lastPointerDownInfoPromise = originalPromise;
  }
}

function readReplay(event: MjolnirEvent): Replay | undefined {
  return Reflect.get(event, REPLAY) as Replay | undefined;
}

function getEventManager(deck: Deck | undefined): EventManager {
  if (!deck) throw new Error('Native editor requires an attached Deck instance');
  const manager = Reflect.get(deck, 'eventManager') as EventManager | undefined;
  if (!manager || typeof manager.emit !== 'function')
    throw new Error('Native asynchronous editor requires Mjolnir event replay');
  return manager;
}

type MultiPickOptions = {
  x: number;
  y: number;
  layerIds: string[];
  radius?: number;
  depth?: number;
};
type AsyncPicker = (
  method: 'pickObjectAsync',
  statistics: string,
  options: MultiPickOptions
) => Promise<{result: PickingInfo[]}>;

function requireAsyncPicker(deck: Deck | undefined): AsyncPicker {
  if (!deck) throw new Error('Native editor requires an attached Deck instance');
  // Deck 9.4 exposes no public point multi-pick async API. Its pinned _pickAsync forwards depth
  // to the async DeckPicker; public pickObjectAsync discards all but the first returned hit.
  const picker: unknown = Reflect.get(deck, '_pickAsync');
  if (typeof picker !== 'function')
    throw new Error('Pinned Deck asynchronous multi-picking seam is unavailable');
  return picker as AsyncPicker;
}

async function preparePointerEvent(
  layer: EditableGeoJsonLayer,
  event: MjolnirEvent
): Promise<PointerEvent> {
  if (!('offsetCenter' in event)) return null;
  const screenCoords: [number, number] = [event.offsetCenter.x, event.offsetCenter.y];
  const deck = layer.context.deck;
  if (!deck) throw new Error('Native editor requires an attached Deck instance');
  const terrainLayerIds = layer.context.layerManager
    .getLayers()
    .filter(candidate => Reflect.get(candidate.props, 'pickable') === '3d')
    .map(candidate => candidate.id);
  let mapCoords: number[];
  if (terrainLayerIds.length) {
    const terrain = await deck.pickObjectAsync({
      x: screenCoords[0],
      y: screenCoords[1],
      layerIds: terrainLayerIds,
      unproject3D: true
    });
    if (!terrain?.coordinate) return null;
    mapCoords = [terrain.coordinate[0], terrain.coordinate[1]];
  } else {
    // Split maps share one Deck; unproject through the view that contains the pointer, relative to
    // that view's rectangle, instead of whichever viewport last drew.
    const pointerViewport = getEventViewport(layer, event);
    mapCoords = pointerViewport.unproject([
      screenCoords[0] - pointerViewport.x,
      screenCoords[1] - pointerViewport.y
    ]);
  }
  const picks = await requireAsyncPicker(deck).call(
    deck,
    'pickObjectAsync',
    'pickMultipleObjects Time',
    {
      x: screenCoords[0],
      y: screenCoords[1],
      layerIds: [layer.id],
      radius: layer.props.pickingRadius,
      depth: layer.props.pickingDepth
    }
  );
  // A guide pick without its object cannot describe an edit handle; native handle lookup reads it.
  return {
    screenCoords,
    mapCoords,
    picks: picks.result.filter(pick => !Reflect.get(pick, 'isGuide') || pick.object),
    sourceEvent: event.srcEvent
  };
}

/**
 * The view under the pointer. A layer's `context.viewport` is whichever view drew or picked last,
 * which differs between split views and must not decide whether an event is still current.
 */
function getEventViewport(layer: EditableGeoJsonLayer, event: MjolnirEvent): Viewport {
  const deck = layer.context.deck;
  if (deck && 'offsetCenter' in event) {
    const {x, y} = event.offsetCenter;
    const containing = deck.getViewports().find(candidate => candidate.containsPixel({x, y}));
    if (containing) return containing;
  }
  return layer.context.viewport;
}

function sameModeConfig(first: unknown, second: unknown): boolean {
  if (first === second) return true;
  if (!first || !second || typeof first !== 'object' || typeof second !== 'object') return false;
  // Pinned Kepler recreates {viewport, screenSpace, lockRectangles, steps} on every render.
  // The live Deck viewport is compared separately; equal mode settings must survive layer clones.
  const keys = Object.keys(first).filter(key => key !== 'viewport');
  return (
    keys.length === Object.keys(second).filter(key => key !== 'viewport').length &&
    keys.every(key => Reflect.get(first, key) === Reflect.get(second, key))
  );
}
