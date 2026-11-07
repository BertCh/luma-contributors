// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Renderer-neutral description of a preview. The live luma.gl scene (runtime.ts) and the static SVG
// fallback (fallback.tsx) both draw exactly this, so they always match.

/** Scene space is a 160 by 100 box; canvases cover it. */
export const SCENE_WIDTH = 160;
export const SCENE_HEIGHT = 100;

export type Point = [number, number];

export type Role = 'primary' | 'warm' | 'hot' | 'cool' | 'violet' | 'ink' | 'dim';

export type RampName = 'heat' | 'ice' | 'terrain' | 'spectral';

export type SpriteItem = {
  x: number;
  y: number;
  /** Depth, only used by point clouds. */
  z?: number;
  /** Radius in scene units. */
  radius: number;
  role: Role;
  alpha?: number;
  /** 0..1, offsets the pulse. */
  phase?: number;
};

export type SpriteLayer = {
  type: 'sprites';
  items: SpriteItem[];
  /** Pulse amplitude, 0 for none. */
  pulse?: number;
  /** Turntable rotation with perspective, for point clouds. */
  turntable?: {angle: number; speed: number; perspective: number};
};

export type PathItem = {
  points: Point[];
  role: Role;
  /** Width in scene units. */
  width: number;
  alpha?: number;
  phase?: number;
};

export type StrokeLayer = {
  type: 'strokes';
  items: PathItem[];
  animation?: 'static' | 'flow' | 'draw' | 'pulse';
  /** Cycles per second. */
  speed?: number;
  /** Flow animation only: length of the bright trail, 1 is the default. */
  trail?: number;
  blend?: 'add' | 'alpha';
};

export type PolygonItem = {ring: Point[]; role: Role; alpha?: number; phase?: number};

export type FillLayer = {
  type: 'fills';
  items: PolygonItem[];
  animation?: 'static' | 'pulse';
  blend?: 'add' | 'alpha';
};

export type FieldStyle = 'ramp' | 'contours' | 'hillshade' | 'expand';

export type FieldLayer = {
  type: 'field';
  columns: number;
  rows: number;
  /** Row-major values in 0..1, row 0 at the top. */
  values: Float32Array;
  ramp: RampName;
  style: FieldStyle;
  /** Contour levels, or isochrone bands. */
  levels?: number;
  /** Nearest filtering for raster cells. */
  filter?: 'linear' | 'nearest';
  /** Breathing amplitude (ramp) or hillshade relief strength. */
  strength?: number;
  speed?: number;
  /** Hillshade only: fixed light direction in radians (image space, y down). Sweeps when unset. */
  light?: number;
  /** Isochrone expansion only: the value the expansion stops at, in 0..1. */
  cutoff?: number;
};

export type PreviewLayer = SpriteLayer | StrokeLayer | FillLayer | FieldLayer;

export type PreviewData = {
  layers: PreviewLayer[];
};
