// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Small factories for preview layers, layer items and slider parameters, shared by the
// function-specific builders in entries/*.ts. Options are passed through as given, so a builder
// states exactly the animation, speed, trail, pulse and blend it wants.

import type {GpuSceneParameter} from '../types.ts';
import type {
  FillLayer,
  PathItem,
  Point,
  PolygonItem,
  Role,
  SpriteItem,
  SpriteLayer,
  StrokeLayer
} from './types.ts';

export type StrokeLayerOptions = Omit<StrokeLayer, 'type' | 'items'>;
export type FillLayerOptions = Omit<FillLayer, 'type' | 'items'>;
export type SpriteLayerOptions = Omit<SpriteLayer, 'type' | 'items'>;

export function createStrokeLayer(
  items: PathItem[],
  options: StrokeLayerOptions = {}
): StrokeLayer {
  return {type: 'strokes', items, ...options};
}

export function createFillLayer(items: PolygonItem[], options: FillLayerOptions = {}): FillLayer {
  return {type: 'fills', items, ...options};
}

export function createSpriteLayer(
  items: SpriteItem[],
  options: SpriteLayerOptions = {}
): SpriteLayer {
  return {type: 'sprites', items, ...options};
}

/** A glowing dot at a scene point. */
export function createDot(
  point: Point,
  role: Role,
  radius: number,
  alpha = 0.9,
  phase = 0
): SpriteItem {
  return {x: point[0], y: point[1], radius, role, alpha, phase};
}

/** A stroked path. */
export function createPath(
  points: Point[],
  role: Role,
  width: number,
  alpha = 0.9,
  phase = 0
): PathItem {
  return {points, role, width, alpha, phase};
}

/** A numeric slider. */
export function createSlider(
  id: string,
  label: string,
  min: number,
  max: number,
  step: number,
  defaultValue: number,
  unit?: string
): GpuSceneParameter {
  return unit === undefined
    ? {id, label, min, max, step, default: defaultValue}
    : {id, label, min, max, step, default: defaultValue, unit};
}

/** A segmented choice; the parameter value is the index of the chosen option. */
export function createChoice(
  id: string,
  label: string,
  options: readonly string[],
  defaultIndex = 0
): GpuSceneParameter {
  return {id, label, min: 0, max: options.length - 1, step: 1, default: defaultIndex, options};
}
