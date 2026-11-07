// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {METERS_PER_DEGREE_LATITUDE, type FlowFieldColumnMode} from './flow-field-grid';

import {clamp, toRadians} from '@math.gl/core';

/** Subset of the pinned Kepler flow-field `visConfig` that the GPU particle renderer consumes. */
export type FlowFieldVisConfig = {
  opacity?: number;
  strokeWidth?: number;
  colorBySpeed?: boolean;
  trailLength?: number;
  cycleSeconds?: number;
  lineLifetime?: number;
  seamlessLoop?: boolean;
  linesPerScreen?: number;
  zoomResponse?: number;
  smoothing?: number;
  gridResolution?: number;
  elevationMultiplier?: number;
  debugGrid?: boolean;
  /** Native layer color range, used for color by speed. */
  colorRange?: {colors?: readonly (string | readonly number[])[]};
};

/**
 * `visual`: field speeds are rescaled so the fastest cell travels a fixed share of the field per
 * lifetime (native streamlines are cosmetic and always visibly flow). `physical`: m/s are used
 * verbatim in real seconds; real-world fields then move very slowly on screen.
 */
export type FlowFieldUnitPolicy = 'visual' | 'physical';

export const MINIMUM_FLOW_FIELD_PARTICLES = 1024;
export const MAXIMUM_FLOW_FIELD_PARTICLES = 262144;
/** Particles per native "line per screen" setting. */
export const FLOW_FIELD_PARTICLES_PER_LINE = 24;
/** Share of the field's smaller extent that the fastest particle crosses during one lifetime. */
export const FLOW_FIELD_VISUAL_TRAVEL_SHARE = 0.25;

export type FlowFieldPlan = {
  particleCount: number;
  lifetimeSeconds: number;
  trailSeconds: number;
  widthPixels: number;
  opacity: number;
  /** Multiplies every velocity texel before upload (1 while the GPU resolves the scale itself). */
  velocityScale: number;
  /**
   * GPU-resident grids: `velocityScale` is `velocityScaleNumerator / maximumSpeed`, evaluated on the
   * GPU from the reduced maximum speed so no scalar is read back. Zero means unit scale (physical).
   */
  velocityScaleNumerator: number;
  unitPolicy: FlowFieldUnitPolicy;
  /** Native options this renderer cannot honor, as specific reasons. */
  limits: string[];
};

function clampNumber(value: unknown, fallback: number, minimum: number, maximum: number): number {
  const number = typeof value === 'number' && Number.isFinite(value) ? value : fallback;
  return clamp(number, minimum, maximum);
}

/** Stable key: the field is rebuilt (and particles reset) only when this changes. */
export function getFlowFieldGridKey(
  mode: FlowFieldColumnMode,
  columns: readonly (string | undefined)[],
  visConfig: FlowFieldVisConfig
): string {
  return [mode, ...columns, visConfig.gridResolution ?? 64, visConfig.smoothing ?? 4].join('|');
}

/** Visual-policy numerator (`share * smaller extent / lifetime`, m/s); zero for the physical policy. */
export function getFlowFieldVelocityNumerator(
  bounds: readonly [number, number, number, number],
  lifetimeSeconds: number,
  unitPolicy: FlowFieldUnitPolicy
): number {
  const [west, south, east, north] = bounds;
  const middleLatitude = (south + north) / 2;
  const extentMeters = Math.min(
    (east - west) * METERS_PER_DEGREE_LATITUDE * Math.max(0.2, Math.cos(toRadians(middleLatitude))),
    (north - south) * METERS_PER_DEGREE_LATITUDE
  );
  return unitPolicy === 'physical'
    ? 0
    : (FLOW_FIELD_VISUAL_TRAVEL_SHARE * extentMeters) / lifetimeSeconds;
}

export const MAXIMUM_FLOW_FIELD_PALETTE_COLORS = 8;

function parseHexColor(hex: string): [number, number, number] {
  const digits = hex.replace('#', '');
  if (digits.length < 6) return [186, 233, 223];
  return [
    parseInt(digits.slice(0, 2), 16),
    parseInt(digits.slice(2, 4), 16),
    parseInt(digits.slice(4, 6), 16)
  ];
}

/**
 * Native color-by-speed palette as linear 0-1 RGB, at most eight equal-width speed bands (like
 * native, the first color covers the slowest band and the last the fastest). Longer palettes are
 * resampled evenly to eight bands. Empty when the layer has no color range.
 */
export function getFlowFieldPalette(
  colorRange: FlowFieldVisConfig['colorRange']
): [number, number, number][] {
  const colors = (colorRange?.colors ?? []).map(color =>
    typeof color === 'string'
      ? parseHexColor(color)
      : ([color[0] ?? 0, color[1] ?? 0, color[2] ?? 0] as [number, number, number])
  );
  const picked =
    colors.length > MAXIMUM_FLOW_FIELD_PALETTE_COLORS
      ? Array.from(
          {length: MAXIMUM_FLOW_FIELD_PALETTE_COLORS},
          (_, index) =>
            colors[Math.floor(((index + 0.5) * colors.length) / MAXIMUM_FLOW_FIELD_PALETTE_COLORS)]
        )
      : colors;
  return picked.map(([red, green, blue]) => [red / 255, green / 255, blue / 255]);
}

/**
 * Spawn rectangle for a viewport, normalized to the field bounds ([west, south, east, north] in
 * 0..1), padded by a tenth of the viewport. Undefined when the viewport covers the field, misses
 * it, or has no usable bounds: particles then spread over the whole field.
 */
export function getFlowFieldSpawnRegion(
  viewportBounds: readonly [number, number, number, number] | undefined,
  fieldBounds: readonly [number, number, number, number]
): [number, number, number, number] | undefined {
  if (!viewportBounds || !viewportBounds.every(Number.isFinite)) return undefined;
  let [west, south, east, north] = viewportBounds;
  if (east - west >= 360) return undefined;
  const padLongitude = (east - west) * 0.1;
  const padLatitude = (north - south) * 0.1;
  west -= padLongitude;
  east += padLongitude;
  south -= padLatitude;
  north += padLatitude;
  const [fieldWest, fieldSouth, fieldEast, fieldNorth] = fieldBounds;
  for (const shift of [0, -360, 360]) {
    const shiftedWest = west + shift;
    const shiftedEast = east + shift;
    if (shiftedEast < fieldWest || shiftedWest > fieldEast) continue;
    const region: [number, number, number, number] = [
      Math.max(0, (shiftedWest - fieldWest) / (fieldEast - fieldWest)),
      Math.max(0, (south - fieldSouth) / (fieldNorth - fieldSouth)),
      Math.min(1, (shiftedEast - fieldWest) / (fieldEast - fieldWest)),
      Math.min(1, (north - fieldSouth) / (fieldNorth - fieldSouth))
    ];
    if (!(region[2] > region[0] && region[3] > region[1])) return undefined;
    return region[0] <= 0 && region[1] <= 0 && region[2] >= 1 && region[3] >= 1
      ? undefined
      : region;
  }
  return undefined;
}

export function planFlowField(
  visConfig: FlowFieldVisConfig,
  field: {
    bounds: readonly [number, number, number, number];
    /** Omitted when the grid lives on the GPU and the maximum speed is resolved there. */
    maximumSpeed?: number;
  },
  unitPolicy: FlowFieldUnitPolicy = 'visual'
): FlowFieldPlan {
  const cycleSeconds = clampNumber(visConfig.cycleSeconds, 20, 2, 180);
  const lifetimeShare = clampNumber(visConfig.lineLifetime, 0.55, 0.05, 1);
  const lifetimeSeconds = cycleSeconds * lifetimeShare;
  const trailShare = clampNumber(visConfig.trailLength, 22, 1, 100) / 100;
  const linesPerScreen = clampNumber(visConfig.linesPerScreen, 120, 10, 100000);
  const particleCount = Math.round(
    clampNumber(
      linesPerScreen * FLOW_FIELD_PARTICLES_PER_LINE,
      MINIMUM_FLOW_FIELD_PARTICLES,
      MINIMUM_FLOW_FIELD_PARTICLES,
      MAXIMUM_FLOW_FIELD_PARTICLES
    )
  );
  const velocityScaleNumerator = getFlowFieldVelocityNumerator(
    field.bounds,
    lifetimeSeconds,
    unitPolicy
  );
  const velocityScale =
    field.maximumSpeed === undefined || !(field.maximumSpeed > 0)
      ? 1
      : velocityScaleNumerator
        ? velocityScaleNumerator / field.maximumSpeed
        : 1;
  const limits: string[] = [];
  return {
    particleCount,
    lifetimeSeconds,
    trailSeconds: Math.max(0.05, trailShare * lifetimeSeconds),
    widthPixels: clampNumber(visConfig.strokeWidth, 1.5, 0.2, 50),
    opacity: clampNumber(visConfig.opacity, 0.85, 0, 1),
    velocityScale,
    velocityScaleNumerator,
    unitPolicy,
    limits
  };
}

/** Native layer color is 0-255 RGB; the particle layer takes linear 0-1 RGBA. */
export function normalizeFlowFieldColor(
  color: readonly number[] | undefined,
  opacity: number
): [number, number, number, number] {
  const [red = 186, green = 233, blue = 223] = color ?? [];
  return [red / 255, green / 255, blue / 255, opacity];
}

/** Longitude/latitude rectangle intersection, allowing the viewport to be shifted by world copies. */
export function isFlowFieldVisible(
  viewportBounds: readonly [number, number, number, number],
  fieldBounds: readonly [number, number, number, number]
): boolean {
  const [west, south, east, north] = viewportBounds;
  if (north < fieldBounds[1] || south > fieldBounds[3]) return false;
  if (east - west >= 360) return true;
  for (const shift of [-360, 0, 360])
    if (east + shift >= fieldBounds[0] && west + shift <= fieldBounds[2]) return true;
  return false;
}
