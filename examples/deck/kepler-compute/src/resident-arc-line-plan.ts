// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Native arrowhead shapes of the pinned Kepler 3.3 `target-arrow.js` (target end only). */
export const NATIVE_ARROW_SHAPES = ['chevron', 'triangle', 'stealth'] as const;
export type ConnectionArrowShape = (typeof NATIVE_ARROW_SHAPES)[number];

/** Altitude columns folded into the prepared float32x3 connection endpoints. */
export type ConnectionPreparedAltitude = {alt0?: string | null; alt1?: string | null};

export type ConnectionConfigView = {
  columnMode?: string;
  columns: {[name: string]: {value?: string | null} | undefined};
  visConfig: {
    targetArrowEnabled?: boolean;
    targetArrowShape?: string;
    targetArrowSize?: number;
    elevationScale?: number;
    sourceArrowEnabled?: boolean;
  };
};

/**
 * Specific reason a native Arc or Line configuration has no resident rendering, or null.
 * Native Arc has no altitude column (its height is the constant arc profile). Native Line reads
 * optional `alt0`/`alt1` columns; they render through float32x3 endpoints when the prepared
 * connection positions carry the same altitude columns (`preparedAltitude`).
 */
export function getConnectionLimitReason(
  layerType: 'line' | 'arc',
  config: ConnectionConfigView,
  preparedAltitude?: ConnectionPreparedAltitude
): string | null {
  const mode = config.columnMode ?? 'points';
  if (mode === 'neighbors')
    return 'Point-and-neighbors connections need a resident neighbor-list expansion; use source and target columns.';
  if (mode === 'geoarrow')
    return 'GeoArrow connection points have no resident binding; use source and target coordinate columns.';
  if (mode !== 'points') return 'This connection column mode has no resident binding.';
  const columns = config.columns;
  const altitudeRoles = layerType === 'line' ? ['alt0', 'alt1'] : ['alt0', 'alt1', 'alt'];
  if (!altitudeRoles.some(role => columns[role]?.value)) return null;
  if (layerType === 'arc')
    return 'Native Arc has no altitude columns; its height is the fixed arc profile.';
  const matches =
    (columns['alt0']?.value ?? null) === (preparedAltitude?.alt0 ?? null) &&
    (columns['alt1']?.value ?? null) === (preparedAltitude?.alt1 ?? null);
  return matches
    ? null
    : 'Line altitude columns (alt0/alt1) differ from the prepared connection positions; prepare the dataset with these columns.';
}

/** Arrowhead capabilities beyond the native target arrow; the native layers expose only the target end. */
export function getConnectionArrowLimits(config: ConnectionConfigView): string[] {
  const limits: string[] = [];
  if (config.visConfig.sourceArrowEnabled)
    limits.push('Source-end arrowheads are not a native Kepler option and are not rendered.');
  const shape = config.visConfig.targetArrowShape;
  if (shape && !(NATIVE_ARROW_SHAPES as readonly string[]).includes(shape))
    limits.push(`Arrow shape "${shape}" is unknown; a chevron is drawn.`);
  return limits;
}

/** Pixel arrow size, falling back to the native default of 16. */
export function getConnectionArrowSize(size: unknown): number {
  return typeof size === 'number' && Number.isFinite(size) && size > 0 ? size : 16;
}

/** Native lines scale altitude by `elevationScale` on maps; globe lines ignore it. */
export function getConnectionElevationScale(elevationScale: unknown, globe: boolean): number {
  return !globe && typeof elevationScale === 'number' && Number.isFinite(elevationScale)
    ? elevationScale
    : 1;
}
