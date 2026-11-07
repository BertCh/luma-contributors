// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {clamp} from '@math.gl/core';

/**
 * Pure planning and reference semantics for native Kepler Grid/Hexagon color and elevation
 * channels. Everything here is GPU free so node specs can compare the planner and the reference
 * implementations against the native d3/deck.gl behavior that the GPU stage reproduces.
 */

/** Aggregation names exposed by the native Kepler UI (AGGREGATION_TYPES). */
export type NativeAggregationName =
  | 'count'
  | 'average'
  | 'sum'
  | 'minimum'
  | 'maximum'
  | 'median'
  | 'stdev'
  | 'variance'
  | 'mode'
  | 'countUnique';

export const NATIVE_AGGREGATION_NAMES: readonly NativeAggregationName[] = [
  'count',
  'average',
  'sum',
  'minimum',
  'maximum',
  'median',
  'stdev',
  'variance',
  'mode',
  'countUnique'
];

/** Statistic names understood by GPUGroupStatistics. */
export type GroupStatisticName =
  | 'mean'
  | 'sum'
  | 'minimum'
  | 'maximum'
  | 'median'
  | 'standardDeviation'
  | 'variance'
  | 'mode'
  | 'uniqueCount';

/** Maps a native aggregation to the GPUGroupStatistics statistic; count needs no weight column. */
export const GROUP_STATISTIC_BY_AGGREGATION: Readonly<
  Record<Exclude<NativeAggregationName, 'count'>, GroupStatisticName>
> = {
  average: 'mean',
  sum: 'sum',
  minimum: 'minimum',
  maximum: 'maximum',
  median: 'median',
  stdev: 'standardDeviation',
  variance: 'variance',
  mode: 'mode',
  countUnique: 'uniqueCount'
};

export type AggregationChannelScale =
  | 'quantize'
  | 'linear'
  | 'quantile'
  | 'custom'
  | 'jenks'
  | 'ordinal';

export type AggregationChannelPlan = {
  channel: 'color' | 'elevation';
  aggregation: NativeAggregationName;
  /** Present for every aggregation except count. */
  weightColumn?: string;
  scale: AggregationChannelScale;
  lowerPercentile: number;
  upperPercentile: number;
  /** Sorted finite thresholds of the native custom color map. */
  customThresholds: readonly number[];
};

export type AggregationStylePlan = {
  color: AggregationChannelPlan;
  /** Present when the elevation channel can change what is drawn (extrusion or percentile). */
  elevation?: AggregationChannelPlan;
  extruded: boolean;
  /** Native visConfig.sizeRange, in meters before elevationScale. */
  elevationRange: readonly [number, number];
  /** Color interpolation: linear scales blend palette texels, the others pick the nearest. */
  interpolation: 'linear' | 'nearest';
};

/** Layer subset consumed by the planner. */
export type AggregationPlanLayer = {
  type: 'grid' | 'hexagon';
  config: {
    columnMode?: string;
    colorField?: {name: string; type?: string} | null;
    colorScale?: PropertyKey;
    sizeField?: {name: string; type?: string} | null;
    sizeScale?: PropertyKey;
    visConfig: {
      enable3d?: boolean;
      colorAggregation?: string;
      sizeAggregation?: string;
      percentile?: readonly number[];
      elevationPercentile?: readonly number[];
      sizeRange?: readonly number[];
      colorRange?: {colors: readonly string[]; colorMap?: unknown};
    };
  };
};

function normalizeAggregation(value: string | undefined, fallback: string): NativeAggregationName {
  const name = (value ?? fallback).toLowerCase();
  const aliases: Record<string, NativeAggregationName> = {
    mean: 'average',
    average: 'average',
    count: 'count',
    sum: 'sum',
    minimum: 'minimum',
    min: 'minimum',
    maximum: 'maximum',
    max: 'maximum',
    median: 'median',
    stdev: 'stdev',
    variance: 'variance',
    mode: 'mode',
    countunique: 'countUnique'
  };
  const normalized = aliases[name];
  if (!normalized)
    throw new Error(
      `Resident aggregation does not support ${value}. Choose count, average, sum, minimum, maximum, median, stdev, variance, mode or countUnique.`
    );
  return normalized;
}

function getPercentiles(value: readonly number[] | undefined, label: string): [number, number] {
  const lower = value?.[0] ?? 0;
  const upper = value?.[1] ?? 100;
  if (!Number.isFinite(lower) || !Number.isFinite(upper))
    throw new Error(`${label} percentile bounds must be finite numbers.`);
  return [lower, upper];
}

/** Sorted finite thresholds of a native custom colorMap: entries `[threshold, color]`. */
export function getCustomColorThresholds(colorMap: unknown): number[] {
  return (Array.isArray(colorMap) ? colorMap : [])
    .flatMap(entry =>
      Array.isArray(entry) && typeof entry[0] === 'number' && Number.isFinite(entry[0])
        ? [entry[0]]
        : []
    )
    .sort((first, second) => first - second);
}

/** Returns true when a layer needs more than the count/sum/mean color-only fast path. */
export function isAdvancedAggregationLayer(layer: AggregationPlanLayer): boolean {
  const {config} = layer;
  const {visConfig} = config;
  const aggregation = config.colorField
    ? normalizeAggregation(visConfig.colorAggregation, 'average')
    : 'count';
  const [lower, upper] = getPercentiles(visConfig.percentile, 'Color');
  const [elevationLower, elevationUpper] = getPercentiles(
    visConfig.elevationPercentile,
    'Elevation'
  );
  return (
    !(aggregation === 'count' || aggregation === 'sum' || aggregation === 'average') ||
    lower !== 0 ||
    upper !== 100 ||
    elevationLower !== 0 ||
    elevationUpper !== 100 ||
    Boolean(visConfig.enable3d) ||
    ['custom', 'jenks', 'ordinal'].includes(String(config.colorScale ?? 'quantize'))
  );
}

/**
 * Interprets the native channel controls. Throws a specific, user-visible reason when a control
 * cannot be reproduced instead of silently changing its meaning.
 */
export function getAggregationStylePlan(layer: AggregationPlanLayer): AggregationStylePlan {
  const {config} = layer;
  const {visConfig} = config;
  const colorScale = String(config.colorScale ?? 'quantize');
  if (
    colorScale !== 'quantize' &&
    colorScale !== 'quantile' &&
    colorScale !== 'linear' &&
    colorScale !== 'custom' &&
    colorScale !== 'jenks' &&
    colorScale !== 'ordinal'
  )
    throw new Error(`Resident aggregation does not support the ${colorScale} color scale.`);
  const colorAggregation = config.colorField
    ? normalizeAggregation(visConfig.colorAggregation, 'average')
    : 'count';
  const [lower, upper] = getPercentiles(visConfig.percentile, 'Color');
  const customThresholds = getCustomColorThresholds(visConfig.colorRange?.colorMap);
  if (colorScale === 'custom') {
    if (!customThresholds.length && !visConfig.colorRange?.colorMap)
      throw new Error('The custom color scale needs a color map with breaks.');
    if (customThresholds.length > 63)
      throw new Error('Resident custom color maps support at most 63 breaks.');
    if (lower !== 0 || upper !== 100)
      throw new Error(
        'Resident aggregation cannot combine a custom color scale with a percentile filter.'
      );
  }
  // Jenks and ordinal classify by lookups derived from the aggregated values; their domains are
  // class indices, so a percentile cutoff over raw values has no native meaning here.
  if ((colorScale === 'jenks' || colorScale === 'ordinal') && (lower !== 0 || upper !== 100))
    throw new Error(
      `Resident aggregation cannot combine the ${colorScale} color scale with a percentile filter.`
    );
  const color: AggregationChannelPlan = {
    channel: 'color',
    aggregation: colorAggregation,
    weightColumn: colorAggregation === 'count' ? undefined : config.colorField?.name,
    scale: colorScale,
    lowerPercentile: lower,
    upperPercentile: upper,
    customThresholds
  };
  const [elevationLower, elevationUpper] = getPercentiles(
    visConfig.elevationPercentile,
    'Elevation'
  );
  const elevationScale = String(config.sizeScale ?? 'linear');
  if (elevationScale === 'ordinal')
    throw new Error('Resident aggregation does not support the ordinal elevation scale.');
  const elevationFiltered = elevationLower !== 0 || elevationUpper !== 100;
  const extruded = Boolean(visConfig.enable3d);
  let elevation: AggregationChannelPlan | undefined;
  if (extruded || elevationFiltered) {
    const aggregation = config.sizeField
      ? normalizeAggregation(visConfig.sizeAggregation, 'sum')
      : 'count';
    elevation = {
      channel: 'elevation',
      aggregation,
      weightColumn: aggregation === 'count' ? undefined : config.sizeField?.name,
      // Deck 9 only reads quantile specially; sqrt/log/linear/quantize all stay raw.
      scale: elevationScale === 'quantile' ? 'quantile' : 'linear',
      lowerPercentile: elevationLower,
      upperPercentile: elevationUpper,
      customThresholds: []
    };
  }
  const range = visConfig.sizeRange ?? [0, 500];
  if (range.length < 2 || !Number.isFinite(range[0]) || !Number.isFinite(range[1]))
    throw new Error('The elevation range must be two finite numbers.');
  return {
    color,
    elevation,
    extruded,
    elevationRange: [range[0], range[1]],
    interpolation: colorScale === 'linear' ? 'linear' : 'nearest'
  };
}

// ---------------------------------------------------------------------------------------------
// Reference implementations (d3 / deck.gl 9 semantics) used by specs and by the GPU qualifier.
// ---------------------------------------------------------------------------------------------

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

/** d3.quantile on sorted values (linear / R-7). */
export function getReferenceQuantile(sorted: readonly number[], fraction: number): number {
  const length = sorted.length;
  if (!length) return NaN;
  if (fraction <= 0 || length < 2) return sorted[0];
  if (fraction >= 1) return sorted[length - 1];
  const position = (length - 1) * fraction;
  const index = Math.floor(position);
  return sorted[index] + (sorted[index + 1] - sorted[index]) * (position - index);
}

/**
 * Native `aggregate(points.map(accessor), technique)` for one cell. Missing values (null, NaN,
 * infinite) are skipped as d3 does; count is the row count including nulls. Undefined results
 * are NaN. Mode ties resolve to the smallest value (the GPU statistic's documented rule).
 */
export function getReferenceStatistic(
  values: readonly (number | null | undefined)[],
  aggregation: NativeAggregationName
): number {
  if (aggregation === 'count') return values.length;
  const finite = values.filter(isFiniteNumber);
  if (aggregation === 'countUnique') return values.length ? new Set(finite).size : NaN;
  if (!finite.length) return aggregation === 'sum' && values.length ? 0 : NaN;
  const sum = finite.reduce((total, value) => total + value, 0);
  const mean = sum / finite.length;
  switch (aggregation) {
    case 'sum':
      return sum;
    case 'average':
      return mean;
    case 'minimum':
      return Math.min(...finite);
    case 'maximum':
      return Math.max(...finite);
    case 'median':
      return getReferenceQuantile(
        [...finite].sort((first, second) => first - second),
        0.5
      );
    case 'variance':
    case 'stdev': {
      if (finite.length < 2) return NaN;
      const variance =
        finite.reduce((total, value) => total + (value - mean) ** 2, 0) / (finite.length - 1);
      return aggregation === 'variance' ? variance : Math.sqrt(variance);
    }
    case 'mode': {
      const frequency = new Map<number, number>();
      for (const value of finite) frequency.set(value, (frequency.get(value) ?? 0) + 1);
      let best = NaN;
      let bestCount = 0;
      for (const [value, count] of frequency)
        if (count > bestCount || (count === bestCount && value < best)) {
          best = value;
          bestCount = count;
        }
      return best;
    }
    default:
      return NaN;
  }
}

/** Reference cell assignment: groups source rows by cell and reduces each non-empty cell. */
export function getReferenceCellStatistics(
  cellIds: ArrayLike<number>,
  values: ArrayLike<number | null | undefined>,
  aggregation: NativeAggregationName,
  cellCount: number
): Float64Array {
  const groups = new Map<number, (number | null | undefined)[]>();
  for (let row = 0; row < cellIds.length; row++) {
    const cell = cellIds[row];
    if (!(cell >= 0 && cell < cellCount)) continue;
    const group = groups.get(cell) ?? [];
    group.push(values[row]);
    groups.set(cell, group);
  }
  const result = new Float64Array(cellCount).fill(NaN);
  for (const [cell, group] of groups) result[cell] = getReferenceStatistic(group, aggregation);
  return result;
}

/** deck.gl applyScaleQuantile: 99 thresholds over the finite sorted values. */
export function getReferencePercentileThresholds(values: ArrayLike<number>): number[] {
  const sorted = Array.from(values)
    .filter(isFiniteNumber)
    .sort((first, second) => first - second);
  return Array.from({length: 99}, (_, index) => getReferenceQuantile(sorted, (index + 1) / 100));
}

/** Count of thresholds less than or equal to the value (d3 bisectRight). */
export function getReferenceBisectRight(thresholds: readonly number[], value: number): number {
  let low = 0;
  let high = thresholds.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (thresholds[middle] > value) high = middle;
    else low = middle + 1;
  }
  return low;
}

/** Native classifyBinsByCustomBreaks index of a value. */
export function getReferenceCustomIndex(
  value: number,
  thresholds: readonly number[],
  colorCount: number
): number {
  if (!Number.isFinite(value)) return NaN;
  for (let index = 0; index < thresholds.length; index++)
    if (value < thresholds[index]) return index;
  return colorCount - 1;
}

export type ReferenceChannelScale = {
  /** Per-cell value after the scale transform; NaN means no value. */
  scaled: Float64Array;
  /** The raw domain `[min, max]` of the scaled values (or the fixed quantile/custom domain). */
  domain: [number, number];
  /** Lower and upper cutoff of scaled values; infinities when no filter applies. */
  cutoff: [number, number];
};

/** deck.gl AttributeWithScale.update + getCutoff for one channel. */
export function getReferenceChannelScale(props: {
  values: ArrayLike<number>;
  scale: AggregationChannelScale;
  lowerPercentile: number;
  upperPercentile: number;
  customThresholds?: readonly number[];
  colorCount?: number;
}): ReferenceChannelScale {
  const {values, scale, lowerPercentile, upperPercentile} = props;
  let scaled = Float64Array.from(values);
  let domain: [number, number];
  let cutoff: [number, number] = [-Infinity, Infinity];
  if (scale === 'quantile') {
    const thresholds = getReferencePercentileThresholds(values);
    scaled = scaled.map(value =>
      Number.isFinite(value) ? getReferenceBisectRight(thresholds, value) : NaN
    );
    domain = [0, 99];
    cutoff = [lowerPercentile, upperPercentile - 1];
  } else if (scale === 'custom') {
    const colorCount = props.colorCount ?? (props.customThresholds?.length ?? 0) + 1;
    scaled = scaled.map(value =>
      getReferenceCustomIndex(value, props.customThresholds ?? [], colorCount)
    );
    domain = [0, colorCount - 1];
  } else {
    const finite = Array.from(scaled).filter(Number.isFinite);
    domain = finite.length ? [Math.min(...finite), Math.max(...finite)] : [Infinity, -Infinity];
    if (lowerPercentile > 0 || upperPercentile < 100) {
      const thresholds = getReferencePercentileThresholds(values);
      cutoff = [
        thresholds[Math.floor(lowerPercentile) - 1] ?? -Infinity,
        thresholds[Math.floor(upperPercentile) - 1] ?? Infinity
      ];
    }
  }
  return {scaled, domain, cutoff};
}

export type ReferenceCellAppearance = {
  visible: boolean;
  /** Fraction along the clipped domain for color, or the elevation interpolation fraction. */
  fraction: number;
};

/** The grid-cell vertex shader's draw/hide test and domain fraction for one value. */
export function getReferenceCellAppearance(
  value: number,
  domain: readonly [number, number],
  cutoff: readonly [number, number]
): ReferenceCellAppearance {
  const start = Math.max(domain[0], cutoff[0]);
  const end = Math.min(domain[1], cutoff[1]);
  const hideBelow = Math.max(domain[0] - 1, cutoff[0]);
  const hideAbove = Math.min(domain[1] + 1, cutoff[1]);
  const visible = !(Number.isNaN(value) || value < hideBelow || value > hideAbove);
  const fraction = end === start ? 0 : clamp((value - start) / (end - start), 0, 1);
  return {visible, fraction};
}

/** Texture-nearest palette index for a domain fraction (clamp-to-edge). */
export function getReferencePaletteIndex(fraction: number, colorCount: number): number {
  return clamp(Math.floor(fraction * colorCount), 0, colorCount - 1);
}

/** Native elevation (before elevationScale) for a fraction of the clipped elevation domain. */
export function getReferenceElevation(fraction: number, range: readonly [number, number]): number {
  return range[0] + (range[1] - range[0]) * fraction;
}
