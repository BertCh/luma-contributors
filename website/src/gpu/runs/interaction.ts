// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Pointer interaction of a run: what the pointer means for this operation, the glyphs drawn at the
// cursor and the text readout. A run declares `interaction` in its defineOperationRun spec; the
// framework calls `resolve` at most once per frame, applies the returned parameter values through
// the same path as the sliders, draws the glyphs and reports the readout. See ./README.md.
//
// This file has no luma.gl imports: glyphs are tessellated on the CPU into segments and dots that
// define-run.ts draws with its ordinary layers.

import type {GpuSceneParameter} from '../types';

/** What the pointer is for this operation. See dev-docs/roadmap.md section 7.3. */
export type InteractionKind =
  | 'place'
  | 'aim'
  | 'reach'
  | 'level'
  | 'window'
  | 'swipe'
  | 'probe'
  | 'drag'
  | 'time'
  | 'pad';

export type Point = readonly [number, number];

/** The pointer in the picture. */
export type RunPointer = {
  /** Drawing-world position: x right, y up. Degrees when the run sets `geographicWindow`. */
  x: number;
  y: number;
  /** Position across the canvas, 0 to 1, x right and y down. */
  unitX: number;
  unitY: number;
};

/** A mark drawn over the picture at or around the cursor. Positions and radii are in world units. */
export type RunGlyph =
  | {shape: 'crosshair'; at: Point /** Arm length in canvas pixels. */; size?: number}
  | {shape: 'dot'; at: Point}
  | {shape: 'ring'; at: Point; radius: number; dashed?: boolean}
  | {shape: 'box'; minimum: Point; maximum: Point; dashed?: boolean}
  | {shape: 'line'; from: Point; to: Point; dashed?: boolean}
  | {shape: 'arrow'; from: Point; to: Point}
  /** A full-height vertical line at world x: a swipe curtain or a chart rule. */
  | {shape: 'curtain'; x: number}
  /** Consecutive (x, y) pairs. */
  | {shape: 'polyline'; points: ArrayLike<number>; closed?: boolean; dashed?: boolean};

export type ParameterPatch = Record<string, number>;

/** What `resolve` returns for one pointer position. Every field is optional. */
export type RunInteractionResult = {
  /**
   * Real props or inputs of the operation, by parameter id. Applied like a slider change: `update`
   * writes them and the operation re-runs. Never return a compile-time (`rebuildOn`) id here.
   */
  values?: ParameterPatch;
  /**
   * Compile-time parameter values. Applied only after the pointer has rested, so the graph never
   * recompiles under a moving pointer. Snap them to a few steps.
   */
  settle?: ParameterPatch;
  /** Marks drawn over the picture. */
  glyphs?: readonly RunGlyph[];
  /** One short line: real prop names, values and units. */
  readout?: string;
  /** World x of a swipe curtain; use it in `draw` with a layer's `clip`. */
  curtain?: number;
  /** Hovered element id (or ids) for `draw`; the run decides what it indexes. */
  focus?: number | readonly number[];
  /** True when `resolve` rewrote a buffer the operation reads: re-run even without `values`. */
  rerun?: boolean;
  /**
   * The bright circle of a probe: everything outside it is dimmed while the reader points. Left
   * out, a `probe` gets one around its glyphs; null turns it off; other gestures have none unless
   * they set it.
   */
  spotlight?: Spotlight | null;
};

/** A circle of full brightness; the rest of the picture is dimmed. World units. */
export type Spotlight = {at: Point; radius: number};

/** A CPU copy of a GPU buffer, refreshed after each run. */
export type RunPeek = (buffer: {
  readonly buffer: unknown;
  readonly format: string;
}) => Float32Array | Uint32Array | Int32Array | null;

export type RunInteractionState<Inputs, Outputs> = {
  inputs: Inputs;
  /** The build's buffers. `resolve` may `write` render-only buffers here (masks, link segments). */
  outputs: Outputs;
  /** Current parameter values, including earlier pointer writes. */
  values: Readonly<Record<string, number>>;
  world: {width: number; height: number};
  /** World units per canvas pixel, for glyph sizes and pick tolerances. */
  pixelSize: number;
  /** World position to a position across the canvas (0 to 1, y down), for `rest`. */
  toUnit(point: Point): Point;
  /**
   * The latest CPU copy of a buffer the operation wrote, or null until the first readback arrives.
   * Asking for a buffer subscribes it: it is read back after every run and `resolve` is called
   * again when the copy changes.
   */
  peek: RunPeek;
  /**
   * Counts the run's builds. After a rebuild the buffers in `outputs` are new and empty, so a
   * `resolve` that skips rewriting a render-only buffer when its focus is unchanged must also
   * compare this number.
   */
  build: number;
};

export type RunInteraction<Inputs = unknown, Outputs = unknown> = {
  kind: InteractionKind;
  /** Two or three words for the chip and the accessible description: 'move the observer'. */
  hint: string;
  /**
   * Pointer position (0 to 1 across the canvas, y down) that reproduces the default parameters.
   * The poster is captured there. Default: the canvas center.
   */
  rest?: Point | ((state: RunInteractionState<Inputs, Outputs>) => Point);
  /** Pure and cheap: called at most once per frame. */
  resolve(pointer: RunPointer, state: RunInteractionState<Inputs, Outputs>): RunInteractionResult;
};

// ---------------------------------------------------------------------------------------------
// Parameter mapping helpers

export function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

/** Rounds `value` to the parameter's step and clamps it to its range. */
export function snapToParameter(parameter: GpuSceneParameter, value: number): number {
  const steps = Math.round((value - parameter.min) / parameter.step);
  const snapped = Number((parameter.min + steps * parameter.step).toPrecision(12));
  return clamp(snapped, parameter.min, parameter.max);
}

export function findParameter(
  parameters: readonly GpuSceneParameter[],
  id: string
): GpuSceneParameter {
  const parameter = parameters.find(candidate => candidate.id === id);
  // Interaction bindings name parameters by id; a typo must fail at load, not silently do nothing.
  if (!parameter) throw new Error(`Unknown interaction parameter "${id}"`);
  return parameter;
}

export type AxisScale = 'linear' | 'logarithmic';

/**
 * Maps a fraction (0 to 1) to a parameter value. The inner 90% spans the whole range, so both ends
 * are reachable without pixel hunting at the picture's edge.
 */
export function getParameterAt(
  parameter: GpuSceneParameter,
  fraction: number,
  scale: AxisScale = 'linear'
): number {
  const amount = clamp((fraction - 0.05) / 0.9, 0, 1);
  if (scale === 'logarithmic' && parameter.min > 0) {
    return snapToParameter(
      parameter,
      parameter.min * Math.pow(parameter.max / parameter.min, amount)
    );
  }
  return snapToParameter(parameter, parameter.min + (parameter.max - parameter.min) * amount);
}

/** Inverse of getParameterAt: the fraction at which `value` sits. */
export function getFractionOf(
  parameter: GpuSceneParameter,
  value: number,
  scale: AxisScale = 'linear'
): number {
  const amount =
    scale === 'logarithmic' && parameter.min > 0
      ? Math.log(value / parameter.min) / Math.log(parameter.max / parameter.min)
      : (value - parameter.min) / (parameter.max - parameter.min || 1);
  return 0.05 + clamp(amount, 0, 1) * 0.9;
}

export function formatNumber(value: number, digits = 2): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(digits);
}

/** `label value unit` for a parameter, using its declared step for the number of decimals. */
export function formatParameter(parameter: GpuSceneParameter, value: number): string {
  if (parameter.options) return `${parameter.label} ${parameter.options[Math.round(value)]}`;
  const decimals = (String(parameter.step).split('.')[1] ?? '').length;
  const unit = parameter.unit ?? '';
  const label = parameter.label.replace(/^Data: /, '');
  return `${label} ${value.toFixed(decimals)}${/^[A-Za-z]/.test(unit) ? ' ' : ''}${unit}`;
}

// ---------------------------------------------------------------------------------------------
// Shorthands for common bindings. Each returns a complete RunInteraction.

type AxisBinding =
  | string
  | {parameter: string; scale?: AxisScale /** Flip the axis. */; flip?: boolean};

function readAxis(binding: AxisBinding): {parameter: string; scale: AxisScale; flip: boolean} {
  return typeof binding === 'string'
    ? {parameter: binding, scale: 'linear', flip: false}
    : {parameter: binding.parameter, scale: binding.scale ?? 'linear', flip: Boolean(binding.flip)};
}

/**
 * Binds the pointer's position across the canvas to one or two parameters: x left to right, y
 * bottom to top (so "up" increases the value). Use for `pad`, `time` and for `place` when the run
 * already exposes a position as two percentage parameters.
 */
export function bindAxes<Inputs = unknown, Outputs = unknown>(
  parameters: readonly GpuSceneParameter[],
  options: {
    kind: InteractionKind;
    hint: string;
    x?: AxisBinding;
    y?: AxisBinding;
    /** Extra glyphs and readout for the resolved values. */
    decorate?: (
      values: Record<string, number>,
      pointer: RunPointer,
      state: RunInteractionState<Inputs, Outputs>
    ) => Pick<RunInteractionResult, 'glyphs' | 'readout' | 'focus' | 'curtain'>;
  }
): RunInteraction<Inputs, Outputs> {
  const xAxis = options.x ? readAxis(options.x) : null;
  const yAxis = options.y ? readAxis(options.y) : null;
  const xParameter = xAxis ? findParameter(parameters, xAxis.parameter) : null;
  const yParameter = yAxis ? findParameter(parameters, yAxis.parameter) : null;
  return {
    kind: options.kind,
    hint: options.hint,
    rest: () => [
      xAxis && xParameter
        ? flipIf(getFractionOf(xParameter, xParameter.default, xAxis.scale), xAxis.flip)
        : 0.5,
      yAxis && yParameter
        ? 1 - flipIf(getFractionOf(yParameter, yParameter.default, yAxis.scale), yAxis.flip)
        : 0.5
    ],
    resolve(pointer, state) {
      const values: Record<string, number> = {};
      if (xAxis && xParameter) {
        values[xParameter.id] = getParameterAt(
          xParameter,
          flipIf(pointer.unitX, xAxis.flip),
          xAxis.scale
        );
      }
      if (yAxis && yParameter) {
        values[yParameter.id] = getParameterAt(
          yParameter,
          flipIf(1 - pointer.unitY, yAxis.flip),
          yAxis.scale
        );
      }
      const decorated = options.decorate?.({...state.values, ...values}, pointer, state);
      const readout =
        decorated?.readout ??
        [xParameter, yParameter]
          .filter((parameter): parameter is GpuSceneParameter => Boolean(parameter))
          .map(parameter => formatParameter(parameter, values[parameter.id]))
          .join(' · ');
      return {...decorated, values, readout};
    }
  };
}

function flipIf(fraction: number, flip: boolean): number {
  return flip ? 1 - fraction : fraction;
}

/**
 * Aim: the pointer's compass angle around an anchor sets one parameter (degrees, clockwise from
 * north by default) and its distance from the anchor optionally sets another.
 */
export function aimAround<Inputs = unknown, Outputs = unknown>(
  parameters: readonly GpuSceneParameter[],
  options: {
    hint: string;
    /** World position of the anchor. Default: the picture center. */
    anchor?: (state: RunInteractionState<Inputs, Outputs>) => Point;
    /** Parameter that takes the angle in degrees. */
    angle: string;
    /** 'compass': 0 is up (north), clockwise. 'math': 0 is right (east), counterclockwise. */
    convention?: 'compass' | 'math';
    /** Parameter that takes the distance, mapped from 0 at the anchor to `reach` world units. */
    distance?: {
      parameter: string /** Far is the low end of the range. */;
      invert?: boolean;
      reach?: number;
    };
    readout?: (values: Record<string, number>) => string;
  }
): RunInteraction<Inputs, Outputs> {
  const angleParameter = findParameter(parameters, options.angle);
  const distanceParameter = options.distance
    ? findParameter(parameters, options.distance.parameter)
    : null;
  const convention = options.convention ?? 'compass';
  const getAnchor = (state: RunInteractionState<Inputs, Outputs>): Point =>
    options.anchor?.(state) ?? [state.world.width / 2, state.world.height / 2];
  const getReach = (state: RunInteractionState<Inputs, Outputs>) =>
    options.distance?.reach ?? state.world.height * 0.45;
  const toRadians = (degrees: number) =>
    convention === 'compass' ? ((90 - degrees) * Math.PI) / 180 : (degrees * Math.PI) / 180;
  return {
    kind: 'aim',
    hint: options.hint,
    resolve(pointer, state) {
      const anchor = getAnchor(state);
      const deltaX = pointer.x - anchor[0];
      const deltaY = pointer.y - anchor[1];
      const length = Math.hypot(deltaX, deltaY);
      const values: Record<string, number> = {};
      if (length > state.pixelSize * 4) {
        const mathDegrees = (Math.atan2(deltaY, deltaX) * 180) / Math.PI;
        let degrees = convention === 'compass' ? 90 - mathDegrees : mathDegrees;
        const span = angleParameter.max - angleParameter.min;
        // Wrap into the parameter's range when it covers a full turn, otherwise clamp.
        if (span >= 359) {
          degrees = angleParameter.min + ((((degrees - angleParameter.min) % 360) + 360) % 360);
        } else {
          while (degrees > angleParameter.max && degrees - 360 >= angleParameter.min)
            degrees -= 360;
          while (degrees < angleParameter.min && degrees + 360 <= angleParameter.max)
            degrees += 360;
        }
        values[angleParameter.id] = snapToParameter(angleParameter, degrees);
      }
      if (distanceParameter && options.distance) {
        const amount = clamp(length / getReach(state), 0, 1);
        const fraction = options.distance.invert ? 1 - amount : amount;
        values[distanceParameter.id] = snapToParameter(
          distanceParameter,
          distanceParameter.min + (distanceParameter.max - distanceParameter.min) * fraction
        );
      }
      const merged = {...state.values, ...values};
      return {
        values,
        glyphs: [
          {shape: 'arrow', from: anchor, to: [pointer.x, pointer.y]},
          {shape: 'dot', at: anchor}
        ],
        readout:
          options.readout?.(merged) ??
          [angleParameter, distanceParameter]
            .filter((parameter): parameter is GpuSceneParameter => Boolean(parameter))
            .map(parameter => formatParameter(parameter, merged[parameter.id]))
            .join(' · ')
      };
    },
    rest(state) {
      const anchor = getAnchor(state);
      const radians = toRadians(angleParameter.default);
      let length = getReach(state) * 0.6;
      if (distanceParameter && options.distance) {
        const amount =
          (distanceParameter.default - distanceParameter.min) /
          (distanceParameter.max - distanceParameter.min || 1);
        length = getReach(state) * (options.distance.invert ? 1 - amount : amount);
      }
      const x = anchor[0] + Math.cos(radians) * length;
      const y = anchor[1] + Math.sin(radians) * length;
      const [unitX, unitY] = state.toUnit([x, y]);
      return [clamp(unitX, 0, 1), clamp(unitY, 0, 1)];
    }
  };
}

// ---------------------------------------------------------------------------------------------
// Glyph tessellation

export type GlyphGeometry = {
  /** (x0, y0, x1, y1) per segment, in world units. */
  segments: Float32Array;
  segmentCount: number;
  /** (x, y) per dot. */
  dots: Float32Array;
  dotCount: number;
};

export const GLYPH_SEGMENT_CAPACITY = 1024;
export const GLYPH_DOT_CAPACITY = 32;

export function createGlyphGeometry(): GlyphGeometry {
  return {
    segments: new Float32Array(GLYPH_SEGMENT_CAPACITY * 4),
    segmentCount: 0,
    dots: new Float32Array(GLYPH_DOT_CAPACITY * 2),
    dotCount: 0
  };
}

/**
 * Writes `glyphs` into `geometry` as line segments and dots. `pixelSize` is world units per canvas
 * pixel; `bounds` is the visible world rectangle (for curtains).
 */
export function tessellateGlyphs(
  glyphs: readonly RunGlyph[],
  geometry: GlyphGeometry,
  pixelSize: number,
  bounds: {minimumX: number; minimumY: number; maximumX: number; maximumY: number},
  /** Horizontal stretch of world x relative to y (geographic runs); rings stay round on screen. */
  aspectX = 1
): void {
  geometry.segmentCount = 0;
  geometry.dotCount = 0;
  const dash = pixelSize * 7;
  const addSegment = (x0: number, y0: number, x1: number, y1: number) => {
    if (geometry.segmentCount >= GLYPH_SEGMENT_CAPACITY) return;
    geometry.segments.set([x0, y0, x1, y1], geometry.segmentCount * 4);
    geometry.segmentCount++;
  };
  const addLine = (x0: number, y0: number, x1: number, y1: number, dashed?: boolean) => {
    if (!dashed) return addSegment(x0, y0, x1, y1);
    const length = Math.hypot((x1 - x0) / aspectX, y1 - y0);
    const pieces = Math.max(1, Math.min(64, Math.round(length / dash)));
    for (let piece = 0; piece < pieces; piece += 2) {
      const start = piece / pieces;
      const end = Math.min(1, (piece + 1) / pieces);
      addSegment(
        x0 + (x1 - x0) * start,
        y0 + (y1 - y0) * start,
        x0 + (x1 - x0) * end,
        y0 + (y1 - y0) * end
      );
    }
    return undefined;
  };
  const addDot = (x: number, y: number) => {
    if (geometry.dotCount >= GLYPH_DOT_CAPACITY) return;
    geometry.dots.set([x, y], geometry.dotCount * 2);
    geometry.dotCount++;
  };
  for (const glyph of glyphs) {
    switch (glyph.shape) {
      case 'crosshair': {
        const arm = (glyph.size ?? 9) * pixelSize;
        const gap = 3 * pixelSize;
        const [x, y] = glyph.at;
        addSegment(x - arm * aspectX, y, x - gap * aspectX, y);
        addSegment(x + gap * aspectX, y, x + arm * aspectX, y);
        addSegment(x, y - arm, x, y - gap);
        addSegment(x, y + gap, x, y + arm);
        break;
      }
      case 'dot':
        addDot(glyph.at[0], glyph.at[1]);
        break;
      case 'ring': {
        const steps = 64;
        for (let step = 0; step < steps; step++) {
          if (glyph.dashed && step % 2 === 1) continue;
          const start = (step / steps) * Math.PI * 2;
          const end = ((step + 1) / steps) * Math.PI * 2;
          addSegment(
            glyph.at[0] + Math.cos(start) * glyph.radius * aspectX,
            glyph.at[1] + Math.sin(start) * glyph.radius,
            glyph.at[0] + Math.cos(end) * glyph.radius * aspectX,
            glyph.at[1] + Math.sin(end) * glyph.radius
          );
        }
        break;
      }
      case 'box': {
        const [x0, y0] = glyph.minimum;
        const [x1, y1] = glyph.maximum;
        addLine(x0, y0, x1, y0, glyph.dashed);
        addLine(x1, y0, x1, y1, glyph.dashed);
        addLine(x1, y1, x0, y1, glyph.dashed);
        addLine(x0, y1, x0, y0, glyph.dashed);
        break;
      }
      case 'line':
        addLine(glyph.from[0], glyph.from[1], glyph.to[0], glyph.to[1], glyph.dashed);
        break;
      case 'arrow': {
        const [x0, y0] = glyph.from;
        const [x1, y1] = glyph.to;
        addSegment(x0, y0, x1, y1);
        const deltaX = (x1 - x0) / aspectX;
        const deltaY = y1 - y0;
        const length = Math.hypot(deltaX, deltaY);
        if (length > pixelSize * 6) {
          const head = 9 * pixelSize;
          const unitX = deltaX / length;
          const unitY = deltaY / length;
          for (const side of [-1, 1]) {
            const wingX = -unitX * 0.9 + side * -unitY * 0.45;
            const wingY = -unitY * 0.9 + side * unitX * 0.45;
            addSegment(x1, y1, x1 + wingX * head * aspectX, y1 + wingY * head);
          }
        }
        break;
      }
      case 'curtain':
        addSegment(glyph.x, bounds.minimumY, glyph.x, bounds.maximumY);
        break;
      case 'polyline': {
        const count = Math.floor(glyph.points.length / 2);
        const last = glyph.closed ? count : count - 1;
        for (let index = 0; index < last; index++) {
          const next = (index + 1) % count;
          addLine(
            glyph.points[index * 2],
            glyph.points[index * 2 + 1],
            glyph.points[next * 2],
            glyph.points[next * 2 + 1],
            glyph.dashed
          );
        }
        break;
      }
    }
  }
}
