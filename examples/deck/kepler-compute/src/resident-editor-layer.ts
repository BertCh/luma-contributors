// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Deck, Layer, PickingInfo, Viewport} from '@deck.gl/core';
import type {EditableGeoJsonLayer} from '@deck.gl-community/editable-layers';
import type {EventManager, MjolnirEvent, MjolnirGestureEvent} from 'mjolnir.js';
import {disableEditorDashExtension, makeEditorDashExtensions} from './editor-dash-extension';
import {ResidentEditorPickingExtension} from './resident-editor-picking-extension';

type PointerEvent =
  | (NonNullable<ReturnType<EditableGeoJsonLayer['toBasePointerEvent']>> & {picks: PickingInfo[]})
  | null;
type EditorEvent = MjolnirEvent | Event;
type NativePressState = {
  pressScreenCoords: number[] | null;
  pointerDownScreenCoords: number[] | null;
  pointerDownMapCoords: number[] | null;
  pointerDownPicks: PickingInfo[] | null;
};
type EditorPress = {
  screenCoords: number[];
  viewport: Viewport;
  didDrag: boolean;
  ended: boolean;
  nativeState: NativePressState;
};
type CapturedInput = {
  press?: EditorPress;
  didDrag: boolean;
  nativeState: NativePressState;
};
type EditorQueue = {
  events: EditorEvent[];
  draining: boolean;
  disposed: boolean;
  invalidDrag: boolean;
  press?: EditorPress;
  deckClickHandler: (event: MjolnirEvent) => void;
};
type Replay = {queue: EditorQueue; pointer: PointerEvent};
const REPLAY = Symbol('resident-editor-replay');
const nativeReplays = new WeakMap<Event, Replay>();
const capturedInputs = new WeakMap<EditorEvent, CapturedInput>();
const DECK_CLICK = Symbol('resident-editor-deck-click');
const CANCEL_DRAG = Symbol('resident-editor-overflow-panend');
const MAXIMUM_PENDING_EVENTS = 64;
const factories = new Map<typeof EditableGeoJsonLayer, (layer: Layer) => Layer>();
const adaptedConstructors = new Set<Function>();

/**
 * EditableLayer 10 performs synchronous multi-picking before pointer handlers. Native DOM
 * clicks and Mjolnir gestures share the async queue, but retain their own dispatch paths.
 * Retain its actual constructor/modes but suspend event dispatch until WebGPU picking completes.
 * Mjolnir's public emit() creates a fresh propagation scope: native cancelPan still runs before
 * lower-priority map controllers, rather than attempting to stop an already-dispatched event.
 */
export function makeResidentEditorLayer(nativeLayer: Layer): Layer {
  if (adaptedConstructors.has(nativeLayer.constructor))
    return nativeLayer.clone({extensions: makeResidentEditorExtensions()});
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
        const deckClickHandler = (event: MjolnirEvent) => {
          if (!readReplay(event)) {
            const click = {...event, [DECK_CLICK]: true};
            this._forwardEventToCurrentLayer(click);
          }
        };
        this.setState({
          residentEditorQueue: {
            events: [],
            draining: false,
            disposed: false,
            invalidDrag: false,
            deckClickHandler
          }
        });
        // v10 edits from DOM clicks, but Kepler selects features from Deck's recognized click.
        // Suspend that separate callback too, so it can use the editor-radius multi-pick.
        getEventManager(this.context.deck).on('click', deckClickHandler, {priority: 100});
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
          queue.press = undefined;
          getEventManager(this.context.deck).off('click', queue.deckClickHandler);
        }
        super.finalizeState();
      }

      override _forwardEventToCurrentLayer(event: EditorEvent): void {
        const current = this.getCurrentLayer() as ResidentEditorLayer | null;
        const queue = current?.state.residentEditorQueue;
        if (!current || !queue || queue.disposed) return;
        const replay = readReplay(event);
        if (replay) {
          if (replay.queue === queue && Reflect.get(event, DECK_CLICK) !== true)
            super._forwardEventToCurrentLayer(event);
          return;
        }
        // v10 registers DOM click/dblclick listeners instead of Mjolnir click listeners.
        // Keep DOM events intact: their coordinates, type and button are prototype getters.
        // Pointer-down and map-interaction guards need no pick and must run synchronously
        // before the map handles the press (including secondary buttons and multi-touch).
        if (!('srcEvent' in event) && event.type !== 'click' && event.type !== 'dblclick') {
          super._forwardEventToCurrentLayer(event);
          if (event.type === 'pointerdown' && Reflect.get(event, 'isPrimary') !== false) {
            const position = getEventScreenCoords(current, event);
            if (position) {
              queue.press = createEditorPress(current, event, position);
              Object.assign(current.state._editableLayerState, queue.press.nativeState);
            }
          }
          return;
        }
        // The native double-click guard must stop map zoom during DOM dispatch, before
        // the asynchronous final polygon pick. Calling it after readback is too late.
        if (!('srcEvent' in event) && event.type === 'dblclick')
          current._onNativeMapInteraction(event);
        if (event.type === 'panstart') {
          if (!queue.press || queue.press.ended) {
            // Recognizer-only input (without a DOM pointerdown) retains the native fallback.
            const position = getEventScreenCoords(current, event);
            if (position) {
              queue.press = createEditorPress(current, event, [
                position[0] - (Reflect.get(event, 'deltaX') || 0),
                position[1] - (Reflect.get(event, 'deltaY') || 0)
              ]);
            }
          }
          if (queue.press) queue.press.didDrag = true;
        }
        if ((event.type === 'panend' || event.type === 'pancancel') && queue.press) {
          queue.press.ended = true;
        }
        const input: CapturedInput = {
          press: queue.press,
          didDrag: queue.press?.didDrag ?? Boolean(current.state._editableLayerState.didDrag),
          nativeState: readNativePressState(current)
        };
        // Recognition, not GPU completion, decides whether this press can produce a click.
        if (
          !('srcEvent' in event) &&
          (event.type === 'click' || event.type === 'dblclick') &&
          input.didDrag
        )
          return;
        // Only gestures are replayed through Mjolnir with fresh propagation functions.
        if ('srcEvent' in event) event.stopImmediatePropagation();
        const captured =
          'srcEvent' in event
            ? {
                ...event,
                ...('offsetCenter' in event ? {offsetCenter: {...event.offsetCenter}} : {})
              }
            : event;
        capturedInputs.set(captured, input);
        const previous = queue.events.at(-1);
        // Moves sample the newest pointer location while a read is pending; start/end/click order
        // is retained. This bounds pointer backlog without replaying obsolete hover positions.
        if (
          (event.type === 'pointermove' || event.type === 'panmove') &&
          previous?.type === event.type &&
          capturedInputs.get(previous)?.press === input.press
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

      override _ownsPointerDown(_event: MouseEvent): boolean {
        // Native mousedown ownership in v10 calls toBasePointerEvent synchronously. This
        // Deck-controlled host reserves the native map press while editing instead. The
        // queued native mode's panstart still decides cancelPan from the exact async press
        // picks before Deck's controller runs; unowned presses therefore still pan the map.
        return true;
      }

      override toBasePointerEvent(event: MjolnirGestureEvent | MouseEvent): PointerEvent {
        const replay = readReplay(event);
        if (!replay) throw new Error('Native editor pointer events require asynchronous dispatch');
        // A queued replay belongs to the layer lifecycle that prepared it. A replaced or
        // finalized queue cannot apply edits to the new mode/data; native handlers treat
        // a null pointer as cancellation (panend also clears pointer-down state).
        if (replay.queue !== this.state.residentEditorQueue || replay.queue.disposed) return null;
        return replay.pointer;
      }

      override getPicks(): ReturnType<EditableGeoJsonLayer['getPicks']> {
        // v10 panstart asks again for the press picks after toBasePointerEvent. They were
        // resolved before dispatch, so this must never call Deck's WebGL-only sync picker.
        return this.residentPressPointer?.picks ?? [];
      }

      override getMapCoords(): number[] | null {
        return this.residentPressPointer?.mapCoords ?? null;
      }

      private residentPressPointer?: PointerEvent;

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
              const input = capturedInputs.get(event);
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
              const pressCoords = input?.press?.screenCoords;
              const pressPointer =
                event.type === 'panstart' &&
                pointer &&
                pressCoords &&
                !pressCoords.every(
                  (coordinate, index) => coordinate === pointer.screenCoords[index]
                )
                  ? await preparePointerEvent(current, event, pressCoords)
                  : pointer;
              const latest = this.getCurrentLayer() as ResidentEditorLayer | null;
              if (queue.disposed || !latest || latest.state.residentEditorQueue !== queue) continue;
              if (
                latest.props.data !== source ||
                latest.props.mode !== mode ||
                !sameModeConfig(latest.props.modeConfig, modeConfig) ||
                !getEventViewport(latest, event).equals(viewport) ||
                (event.type === 'panstart' &&
                  input?.press &&
                  !getEventViewport(latest, event, input.press.screenCoords).equals(
                    input.press.viewport
                  ))
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
              latest.residentPressPointer = pressPointer;
              try {
                replayEditorEvent(latest, event, queue, pointer);
              } finally {
                latest.residentPressPointer = undefined;
              }
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
        extensions: makeResidentEditorExtensions()
      });
    factories.set(NativeEditor, factory);
  }
  return factory(nativeLayer);
}

function replayEditorEvent(
  layer: EditableGeoJsonLayer,
  event: EditorEvent,
  queue: EditorQueue,
  pointer: PointerEvent
): void {
  const input = capturedInputs.get(event);
  if (!input) throw new Error('Native editor replay requires captured press state');
  const previous = readNativePressState(layer);
  const previousDidDrag = layer.state._editableLayerState.didDrag;
  // Native pointerdown must run synchronously. Temporarily lend the published handlers the
  // queued gesture's own handles, not the newer press that may have arrived during readback.
  Object.assign(layer.state._editableLayerState, input.press?.nativeState ?? input.nativeState, {
    didDrag: input.didDrag
  });
  try {
    dispatchEditorEvent(layer, event, queue, pointer);
  } finally {
    const current = layer.getCurrentLayer() as EditableGeoJsonLayer | null;
    if (current && Reflect.get(current.state, 'residentEditorQueue') === queue && !queue.disposed) {
      if (input.press) input.press.nativeState = readNativePressState(current);
      // Native handlers replace _editableLayerState via setState, so restore on its current
      // object. An older panend must not clear a newer press or change its click eligibility.
      Object.assign(current.state._editableLayerState, queue.press?.nativeState ?? previous, {
        didDrag: queue.press?.didDrag ?? previousDidDrag
      });
    }
  }
}

function dispatchEditorEvent(
  layer: EditableGeoJsonLayer,
  event: EditorEvent,
  queue: EditorQueue,
  pointer: PointerEvent
): void {
  if (!('srcEvent' in event)) {
    // Mjolnir no longer has a click subscription for the v10 editor. Forward the original
    // DOM event through its native handler, preserving button/detail and sourceEvent.
    nativeReplays.set(event, {queue, pointer});
    try {
      layer._forwardEventToCurrentLayer(event);
    } finally {
      nativeReplays.delete(event);
    }
    return;
  }
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

function readReplay(event: EditorEvent): Replay | undefined {
  return 'srcEvent' in event
    ? (Reflect.get(event, REPLAY) as Replay | undefined)
    : nativeReplays.get(event);
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
  // Deck 10 exposes no public point multi-pick async API. Its pinned _pickAsync forwards depth
  // to the async DeckPicker; public pickObjectAsync discards all but the first returned hit.
  const picker: unknown = Reflect.get(deck, '_pickAsync');
  if (typeof picker !== 'function')
    throw new Error('Pinned Deck asynchronous multi-picking seam is unavailable');
  return picker as AsyncPicker;
}

async function preparePointerEvent(
  layer: EditableGeoJsonLayer,
  event: EditorEvent,
  position?: number[]
): Promise<PointerEvent> {
  const coordinates = position ?? getEventScreenCoords(layer, event);
  if (!coordinates) return null;
  const screenCoords: [number, number] = [coordinates[0], coordinates[1]];
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
    const pointerViewport = getEventViewport(layer, event, screenCoords);
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
    sourceEvent: 'srcEvent' in event ? event.srcEvent : event
  };
}

function getEventScreenCoords(layer: EditableGeoJsonLayer, event: EditorEvent): number[] | null {
  if ('offsetCenter' in event) return [event.offsetCenter.x, event.offsetCenter.y];
  return 'clientX' in event ? layer.getScreenCoords(event) : null;
}

/**
 * The view under the pointer. A layer's `context.viewport` is whichever view drew or picked last,
 * which differs between split views and must not decide whether an event is still current.
 */
function getEventViewport(
  layer: EditableGeoJsonLayer,
  event: EditorEvent,
  coordinates?: number[]
): Viewport {
  const deck = layer.context.deck;
  const position = coordinates ?? getEventScreenCoords(layer, event);
  if (deck && position) {
    const [x, y] = position;
    const containing = deck.getViewports().find(candidate => candidate.containsPixel({x, y}));
    if (containing) return containing;
  }
  return layer.context.viewport;
}

function readNativePressState(layer: EditableGeoJsonLayer): NativePressState {
  const state = layer.state._editableLayerState;
  return {
    pressScreenCoords: state.pressScreenCoords ?? null,
    pointerDownScreenCoords: state.pointerDownScreenCoords ?? null,
    pointerDownMapCoords: state.pointerDownMapCoords ?? null,
    pointerDownPicks: state.pointerDownPicks ?? null
  };
}

function createEditorPress(
  layer: EditableGeoJsonLayer,
  event: EditorEvent,
  screenCoords: number[]
): EditorPress {
  return {
    screenCoords,
    viewport: getEventViewport(layer, event, screenCoords),
    didDrag: false,
    ended: false,
    nativeState: {
      pressScreenCoords: screenCoords,
      pointerDownScreenCoords: null,
      pointerDownMapCoords: null,
      pointerDownPicks: null
    }
  };
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

function makeResidentEditorExtensions() {
  return [...makeEditorDashExtensions(), new ResidentEditorPickingExtension()];
}
