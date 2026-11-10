// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneInstance, GpuSceneParameter} from './types';

type Position = [number, number];

/** Cursor over a picture with its own gesture (not on cards, whose surface is a link). */
const GESTURE_CURSORS: Record<string, string> = {
  place: 'crosshair',
  probe: 'crosshair',
  level: 'crosshair',
  reach: 'crosshair',
  aim: 'crosshair',
  window: 'move',
  drag: 'grab',
  swipe: 'ew-resize',
  time: 'ew-resize',
  pad: 'crosshair'
};

/** The initiating move can precede both React mounting the canvas and GPU scene compilation. */
const hoverPositions = new WeakMap<HTMLElement, Position>();

export function rememberCardHover(owner: HTMLElement, position: Position | null) {
  if (position) hoverPositions.set(owner, position);
  else hoverPositions.delete(owner);
}

/** Routes events through a card's stretched link to its live canvas, including events during startup. */
export function bindCanvasPointer(
  container: HTMLElement,
  isCard: boolean,
  onChange: () => void,
  onReadout: (parameter: GpuSceneParameter, value: number) => void,
  /** Gesture state for the chip: the scene's hint, whether the reader has moved, and pinning. */
  onGesture: (state: {hint?: string; hasMoved?: boolean; isPinned?: boolean}) => void = () => {}
) {
  const surface = (isCard && container.closest<HTMLElement>('.card')) || container;
  let instance: GpuSceneInstance | null = null;
  let scrub: GpuSceneParameter | undefined;
  let position: Position | null = null;
  let previousValue: number | undefined;
  let frame = 0;
  let hasMoved = false;
  // True while `position` is the placeholder a keyboard focus sent before the scene existed.
  let isPositionFromFocus = false;
  // On a page with controls a click pins the gesture, so the reader can go to the sliders.
  let isPinned = false;

  function apply() {
    frame = 0;
    if (!instance) return;
    if (instance.setPointer) {
      const ratio = window.devicePixelRatio || 1;
      instance.setPointer(
        position && [
          position[0] * container.clientWidth * ratio,
          position[1] * container.clientHeight * ratio
        ]
      );
    } else if (scrub && position && instance.setParameters) {
      const steps = Math.round((position[0] * (scrub.max - scrub.min)) / scrub.step);
      const value = Math.min(scrub.max, Number((scrub.min + steps * scrub.step).toPrecision(12)));
      if (value === previousValue) return;
      previousValue = value;
      instance.setParameters({[scrub.id]: value});
      onReadout(scrub, value);
    } else {
      return;
    }
    onChange();
  }

  function move(next: Position | null) {
    position = next;
    isPositionFromFocus = false;
    if (!frame) frame = requestAnimationFrame(apply);
  }

  function handlePointer(event: PointerEvent) {
    if (isCard && event.pointerType === 'touch') return;
    if (isPinned) return;
    if (instance && !hasMoved && event.type === 'pointermove') {
      hasMoved = true;
      onGesture({hasMoved});
    }
    moveClient(event.clientX, event.clientY);
  }

  function handleClick() {
    if (isCard || !instance?.gesture) return;
    isPinned = !isPinned;
    onGesture({isPinned});
  }

  function moveClient(clientX: number, clientY: number) {
    const bounds = container.getBoundingClientRect();
    const x = (clientX - bounds.left) / bounds.width;
    const y = (clientY - bounds.top) / bounds.height;
    move(x >= 0 && x <= 1 && y >= 0 && y <= 1 ? [x, y] : null);
  }

  function leave() {
    if (isPinned) return;
    move(null);
  }

  function focus() {
    if (!surface.querySelector(':focus-visible')) return;
    // A scene with its own gesture stays at its rest position (the poster) until an arrow key.
    if (instance?.gesture) return;
    move([0.5, 0.5]);
    isPositionFromFocus = true;
  }

  function handleKey(event: KeyboardEvent) {
    if (
      !surface.querySelector(':focus-visible') ||
      event.altKey ||
      event.metaKey ||
      event.ctrlKey ||
      event.shiftKey
    )
      return;
    const directions: Record<string, Position> = {
      ArrowLeft: [-1, 0],
      ArrowRight: [1, 0],
      ArrowUp: [0, -1],
      ArrowDown: [0, 1]
    };
    if (event.key === 'Home' && instance?.resetPointer) {
      // Back to the gesture's rest state: the picture of the poster.
      event.preventDefault();
      position = null;
      instance.resetPointer();
      onChange();
      return;
    }
    const delta = directions[event.key];
    if (!delta || (!instance?.setPointer && (!scrub || delta[1]))) return;
    event.preventDefault();
    if (!hasMoved) {
      hasMoved = true;
      onGesture({hasMoved});
    }
    const [x, y] = position ?? instance?.gesture?.rest ?? [0.5, 0.5];
    const step = scrub && !instance?.setPointer ? scrub.step / (scrub.max - scrub.min) : 0.05;
    move([
      Math.min(1, Math.max(0, x + delta[0] * step)),
      Math.min(1, Math.max(0, y + delta[1] * step))
    ]);
  }

  surface.addEventListener('pointermove', handlePointer);
  surface.addEventListener('pointerdown', handlePointer);
  surface.addEventListener('pointerleave', leave);
  surface.addEventListener('click', handleClick);
  if (isCard) {
    surface.addEventListener('focusin', focus);
    surface.addEventListener('focusout', leave);
    surface.addEventListener('keydown', handleKey);
    for (let owner: HTMLElement | null = surface; owner; owner = owner.parentElement) {
      const initial = hoverPositions.get(owner);
      if (initial) {
        moveClient(...initial);
        break;
      }
    }
    focus();
  }

  return {
    attach(created: GpuSceneInstance, parameters: readonly GpuSceneParameter[]) {
      instance = created;
      if (created.gesture) {
        onGesture({hint: created.gesture.hint, hasMoved, isPinned});
        if (!isCard) container.style.cursor = GESTURE_CURSORS[created.gesture.kind] ?? '';
        // Focus arrived before the scene: its centre placeholder must not move the gesture.
        if (isPositionFromFocus) position = null;
      }
      // A scene's own pointer interaction takes priority. Other cards scrub a real operation input.
      if (isCard && !created.setPointer && created.setParameters) {
        scrub =
          created.scrubParameter ??
          parameters.find(parameter => !parameter.options) ??
          parameters[0];
        if (scrub) onReadout(scrub, scrub.default);
      }
      if (position) move(position);
    },
    destroy() {
      cancelAnimationFrame(frame);
      instance = null;
      surface.removeEventListener('pointermove', handlePointer);
      surface.removeEventListener('pointerdown', handlePointer);
      surface.removeEventListener('pointerleave', leave);
      surface.removeEventListener('click', handleClick);
      if (!isCard) container.style.cursor = '';
      surface.removeEventListener('focusin', focus);
      surface.removeEventListener('focusout', leave);
      surface.removeEventListener('keydown', handleKey);
    }
  };
}
