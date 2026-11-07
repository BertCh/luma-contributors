// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {clamp, safeMod, toDegrees} from '@math.gl/core';

/** A pointer position in client (CSS pixel) coordinates. */
export type GesturePoint = {x: number; y: number};

/** Pure, DOM-free gesture math shared by the swipe divider and covered by node unit tests. */

/** Clamps a value into [minimum, maximum]. */
export function clampValue(value: number, minimum: number, maximum: number): number {
  return clamp(value, minimum, maximum);
}

/** Swipe divider position in percent for a pointer x, clamped to the view bounds [left, right]. */
export function getSwipePercentage(clientX: number, left: number, right: number): number {
  const width = right - left;
  if (!(width > 0)) return 50;
  return clampValue(((clientX - left) / width) * 100, 0, 100);
}

/** Midpoint between two touch points. */
export function getGestureCenter(first: GesturePoint, second: GesturePoint): GesturePoint {
  return {x: (first.x + second.x) / 2, y: (first.y + second.y) / 2};
}

/** Distance between two touch points. */
export function getGestureDistance(first: GesturePoint, second: GesturePoint): number {
  return Math.hypot(second.x - first.x, second.y - first.y);
}

/** Pinch scale between two frames of a two-finger gesture; 1 when the start distance is zero. */
export function getPinchScale(
  previous: readonly [GesturePoint, GesturePoint],
  next: readonly [GesturePoint, GesturePoint]
): number {
  const previousDistance = getGestureDistance(previous[0], previous[1]);
  if (previousDistance === 0) return 1;
  return getGestureDistance(next[0], next[1]) / previousDistance;
}

/** Zoom delta (log2 of the pinch scale), matching map zoom levels. */
export function getPinchZoomDelta(
  previous: readonly [GesturePoint, GesturePoint],
  next: readonly [GesturePoint, GesturePoint]
): number {
  return Math.log2(getPinchScale(previous, next));
}

/** Wraps an angle in degrees into (-180, 180]. */
export function normalizeAngleDegrees(angle: number): number {
  let wrapped = angle % 360;
  if (wrapped > 180) wrapped -= 360;
  if (wrapped <= -180) wrapped += 360;
  return wrapped;
}

/** Signed two-finger rotation in degrees (clockwise positive in screen space), shortest way round. */
export function getRotationDeltaDegrees(
  previous: readonly [GesturePoint, GesturePoint],
  next: readonly [GesturePoint, GesturePoint]
): number {
  const previousAngle = Math.atan2(previous[1].y - previous[0].y, previous[1].x - previous[0].x);
  const nextAngle = Math.atan2(next[1].y - next[0].y, next[1].x - next[0].x);
  return normalizeAngleDegrees(toDegrees(nextAngle - previousAngle));
}

/** Map bearing after a two-finger rotation; a clockwise gesture turns the bearing counter-clockwise. */
export function getBearingAfterRotation(bearing: number, rotationDeltaDegrees: number): number {
  return safeMod(bearing - rotationDeltaDegrees, 360);
}

/** What a pointer event means for the divider drag. */
export type DragEventResult = 'start' | 'move' | 'end' | 'ignore';

/**
 * Tracks the single pointer that owns a drag, so extra fingers, other devices and stray events
 * from a finished gesture cannot move the divider. Mouse, touch and pen are treated alike; mouse
 * drags only start from the primary button.
 */
export class PointerDragTracker {
  private activePointerId: number | null = null;

  get isDragging(): boolean {
    return this.activePointerId !== null;
  }

  get pointerId(): number | null {
    return this.activePointerId;
  }

  /** pointerdown: claims the drag unless another pointer already owns it. */
  handlePointerDown(pointerId: number, button = 0): DragEventResult {
    if (this.activePointerId !== null || button !== 0) return 'ignore';
    this.activePointerId = pointerId;
    return 'start';
  }

  /** pointermove: only the owning pointer moves the divider. */
  handlePointerMove(pointerId: number): DragEventResult {
    return pointerId === this.activePointerId ? 'move' : 'ignore';
  }

  /** pointerup and pointercancel: release the drag; a cancel commits the last move, never a jump. */
  handlePointerEnd(pointerId: number): DragEventResult {
    if (pointerId !== this.activePointerId) return 'ignore';
    this.activePointerId = null;
    return 'end';
  }

  /** lostpointercapture without pointerup (for example the element unmounted). */
  reset(): void {
    this.activePointerId = null;
  }
}
