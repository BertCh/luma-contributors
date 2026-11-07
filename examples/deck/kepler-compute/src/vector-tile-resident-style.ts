// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * GPU filter and style contract for resident vector tiles.
 *
 * One compute kernel reads the per-feature Float32 property columns of a tile and writes both the
 * packed filter mask (selection) and the packed RGBA8 fill colors (style). The kernel parameters
 * are plain data, so changing a filter or a style never recompiles the kernel or rebuilds geometry.
 *
 * Styles are translated from a Kepler vectorTile layer by the host's `createNativeStylePlan`
 * (native-style-plan.ts), which owns the palette, scale and domain semantics; this file only maps
 * the resulting resident color plan to kernel parameters. Pure TypeScript, no device access, so the
 * packing and the CPU reference of the WGSL are covered by node specs.
 */

import {
  createNativeStylePlan,
  type NativeStyleLayer,
  type NativeStylePlan
} from './native-style-plan';
import type {DecodedTileFilter} from './vector-tile-decoded-batch';

import {clamp} from '@math.gl/core';

export const VECTOR_TILE_MAXIMUM_FILTERS = 4;
export const VECTOR_TILE_MAXIMUM_SELECT_VALUES = 7;
/** Continuous palettes stay small; the capacity is sized for ordinal palettes (one color per category code). */
export const VECTOR_TILE_MAXIMUM_PALETTE = 256;
export const VECTOR_TILE_MAXIMUM_DOMAIN = 257;

const FILTER_WORD_COUNT = 12;
const FILTER_OFFSET = 11;
const COLOR_HEADER_WORD_COUNT = 9;
const SCALAR_WORD_COUNT = 10;
const PALETTE_OFFSET = FILTER_OFFSET + VECTOR_TILE_MAXIMUM_FILTERS * FILTER_WORD_COUNT;
const DOMAIN_OFFSET = PALETTE_OFFSET + VECTOR_TILE_MAXIMUM_PALETTE;
// Stroke color block: same layout as the fill block (header at words 1..9) with its own palette/domain.
const STROKE_HEADER_OFFSET = DOMAIN_OFFSET + VECTOR_TILE_MAXIMUM_DOMAIN;
const STROKE_PALETTE_OFFSET = STROKE_HEADER_OFFSET + COLOR_HEADER_WORD_COUNT;
const STROKE_DOMAIN_OFFSET = STROKE_PALETTE_OFFSET + VECTOR_TILE_MAXIMUM_PALETTE;
const WIDTH_OFFSET = STROKE_DOMAIN_OFFSET + VECTOR_TILE_MAXIMUM_DOMAIN;
const RADIUS_OFFSET = WIDTH_OFFSET + SCALAR_WORD_COUNT;
export const VECTOR_TILE_PARAMETER_WORD_COUNT = RADIUS_OFFSET + SCALAR_WORD_COUNT;
export const VECTOR_TILE_DEFAULT_LINE_WIDTH = 1;
export const VECTOR_TILE_DEFAULT_RADIUS = 3;

export type VectorTileScaleKind =
  | 'linear'
  | 'sqrt'
  | 'log'
  | 'quantile'
  | 'threshold'
  | 'equal-interval'
  | 'ordinal';

/** Kernel-ready fill style. Color words are packed `r | g << 8 | b << 16 | a << 24`. */
export type VectorTileGpuStyle =
  | Readonly<{kind: 'constant'; color: number}>
  | Readonly<{
      kind: 'field';
      fieldName: string;
      scale: VectorTileScaleKind;
      domain: readonly number[];
      palette: readonly number[];
      interpolation: 'linear' | 'step';
      outsideDomain: 'extrapolate' | 'clamp' | 'reject';
      noDataColor: number;
    }>;

/** Kernel-ready scalar (line width or point radius) style: a constant or a clamped field mapping. */
export type VectorTileScalarStyle =
  | Readonly<{kind: 'constant'; value: number}>
  | Readonly<{
      kind: 'field';
      fieldName: string;
      scale: 'linear' | 'sqrt' | 'log';
      domain: readonly [number, number];
      range: readonly [number, number];
      nullValue: number;
    }>;

/** Data-driven channels beyond the fill color. Absent channels use the defaults. */
export type VectorTileExtraChannels = Readonly<{
  /** Absent or null: stroke color equals the fill color. */
  stroke?: VectorTileGpuStyle | null;
  width?: VectorTileScalarStyle;
  radius?: VectorTileScalarStyle;
}>;

export type VectorTileStyleRender = NativeStylePlan['render'];

export type VectorTileStylePlanResult =
  | {
      supported: true;
      style: VectorTileGpuStyle;
      channels: VectorTileExtraChannels;
      render: VectorTileStyleRender;
    }
  | {supported: false; reasons: readonly string[]};

export function packVectorTileColor(
  red: number,
  green: number,
  blue: number,
  alpha: number = 255
): number {
  return ((red & 255) | ((green & 255) << 8) | ((blue & 255) << 16) | ((alpha & 255) << 24)) >>> 0;
}

function sortFiniteValues(values: Iterable<number>): number[] {
  return [...values].filter(Number.isFinite).sort((left, right) => left - right);
}

function readQuantile(sorted: readonly number[], probability: number): number {
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  return sorted[lower] + (sorted[Math.ceil(position)] - sorted[lower]) * (position - lower);
}

export type PlanVectorTileStyleOptions = {
  /** Vector tile layer id; becomes the native style source id. */
  layerId: string;
  zoom: number;
  /** Bounded sample of the loaded Float32 values of a field (already resident on the CPU at decode). */
  getFieldValues: (fieldName: string) => Iterable<number>;
  /**
   * Append-only category dictionary (code = index) of a categorical field, shared by all tiles of the
   * layer. Without it ordinal color scales are reported as unsupported.
   */
  getFieldCategories?: (fieldName: string) => readonly (string | number)[] | null;
  /** Fallback line width and point radius when no field drives the channel. */
  constants?: {lineWidth?: number; radius?: number};
};

type NativeChannelPlan = NativeStylePlan['channels'][keyof NativeStylePlan['channels']];
type ColorChannelResult = {style: VectorTileGpuStyle} | {reason: string};
type ScalarChannelResult = {style: VectorTileScalarStyle} | {reason: string};

function isCategoricalScale(scale: unknown): boolean {
  return scale === 'ordinal' || scale === 'customOrdinal';
}

function resolveColorDomain(
  scale: string,
  sorted: readonly number[],
  classCount: number
): {kind: 'fixed'; values: number[]} | undefined {
  if (!sorted.length || scale === 'custom' || scale === 'threshold' || scale === 'jenks')
    return undefined;
  return {
    kind: 'fixed',
    values:
      scale === 'quantile'
        ? Array.from({length: classCount + 1}, (_, index) =>
            Math.fround(readQuantile(sorted, index / classCount))
          )
        : [Math.fround(sorted[0]), Math.fround(sorted[sorted.length - 1])]
  };
}

function convertColorChannel(channel: NativeChannelPlan): ColorChannelResult {
  if (channel?.kind === 'constant') {
    const value = Array.isArray(channel.value) ? channel.value : [0, 0, 0];
    return {
      style: {
        kind: 'constant',
        color: packVectorTileColor(value[0], value[1], value[2], value[3] ?? 255)
      }
    };
  }
  if (channel?.kind !== 'resident' || channel.plan.kind !== 'color')
    return {reason: 'Vector tile colors require a constant or color-scale plan.'};
  const plan = channel.plan;
  if (plan.scale === 'ordinal') {
    if (plan.palette.length > VECTOR_TILE_MAXIMUM_PALETTE)
      return {reason: 'The category palette exceeds the GPU kernel capacity.'};
    return {
      style: {
        kind: 'field',
        fieldName: plan.column,
        scale: 'ordinal',
        domain: [],
        palette: plan.palette,
        interpolation: 'step',
        outsideDomain: 'reject',
        noDataColor: plan.noDataColor ?? 0
      }
    };
  }
  if (plan.domain.kind !== 'fixed')
    return {reason: 'Vector tile styles require a fixed resolved domain.'};
  if (
    plan.palette.length > VECTOR_TILE_MAXIMUM_PALETTE ||
    plan.domain.values.length > VECTOR_TILE_MAXIMUM_DOMAIN
  )
    return {reason: 'The palette or domain exceeds the GPU kernel capacity.'};
  return {
    style: {
      kind: 'field',
      fieldName: plan.column,
      scale: plan.scale as VectorTileScaleKind,
      domain: plan.domain.values,
      palette: plan.palette,
      interpolation: plan.interpolation ?? 'linear',
      outsideDomain: plan.outsideDomain,
      noDataColor: plan.noDataColor ?? 0
    }
  };
}

function convertScalarChannel(channel: NativeChannelPlan, constant: number): ScalarChannelResult {
  if (!channel || channel.kind === 'constant') return {style: {kind: 'constant', value: constant}};
  if (channel.kind !== 'resident' || channel.plan.kind !== 'scalar')
    return {reason: 'Vector tile sizes require a field scale plan.'};
  const plan = channel.plan;
  if (
    (plan.scale !== 'linear' && plan.scale !== 'sqrt' && plan.scale !== 'log') ||
    plan.domain.kind !== 'fixed' ||
    plan.domain.values.length !== 2
  )
    return {reason: `Vector tile ${plan.scale} size scales have no GPU binding.`};
  return {
    style: {
      kind: 'field',
      fieldName: plan.column,
      scale: plan.scale,
      domain: [plan.domain.values[0], plan.domain.values[1]],
      range: [plan.range[0], plan.range[1]],
      nullValue: plan.nullValue ?? 0
    }
  };
}

/**
 * Runs the pinned native style translation for a vector tile layer's color, stroke color, line width
 * and point radius channels. The layer is translated as a GeoJSON-family layer (MVT features are
 * GeoJSON-shaped); source domains are resolved from the loaded tiles' values because tiles stream in
 * and have no complete-dataset extent. Categorical (ordinal) colors use the layer's category dictionary.
 */
export function planVectorTileStyle(
  layer: NativeStyleLayer,
  options: PlanVectorTileStyleOptions
): VectorTileStylePlanResult {
  const {config} = layer;
  const domains: Parameters<typeof createNativeStylePlan>[1]['domains'] = {};
  const columns: Record<string, {format: string; dictionary?: readonly (string | number)[]}> = {};
  const addColumn = (fieldName: string, categorical: boolean): void => {
    const dictionary = categorical ? options.getFieldCategories?.(fieldName) : null;
    columns[fieldName] = dictionary ? {format: 'uint32', dictionary} : {format: 'float32'};
  };
  for (const [channel, field, scaleKey, rangeColors] of [
    ['fillColor', config.colorField, config.colorScale, config.visConfig.colorRange],
    [
      'strokeColor',
      config.strokeColorField,
      config.strokeColorScale,
      config.visConfig.strokeColorRange
    ]
  ] as const) {
    if (!field) continue;
    const scale = String(scaleKey ?? 'quantile');
    addColumn(field.name, isCategoricalScale(scale));
    if (isCategoricalScale(scale)) continue;
    const sorted = sortFiniteValues(options.getFieldValues(field.name));
    const domain = resolveColorDomain(scale, sorted, rangeColors?.colors.length ?? 1);
    if (domain) domains[channel] = domain;
  }
  for (const [channel, field] of [
    ['size', config.sizeField],
    ['pointSize', config.radiusField]
  ] as const) {
    if (!field) continue;
    addColumn(field.name, false);
    const sorted = sortFiniteValues(options.getFieldValues(field.name));
    if (sorted.length)
      domains[channel] = {
        kind: 'fixed',
        values: [Math.fround(sorted[0]), Math.fround(sorted[sorted.length - 1])]
      };
  }
  const result = createNativeStylePlan(
    {...layer, type: 'geojson', config: {...layer.config, dataId: options.layerId}},
    {
      source: {datasetId: options.layerId, revision: 1},
      columns,
      domains,
      zoom: options.zoom,
      geojsonPointFeatures: true,
      geojsonProperties: {
        fillColor: 'absent',
        lineColor: 'absent',
        lineWidth: 'absent',
        radius: 'absent',
        elevation: 'absent',
        elevationOffset: 'absent'
      }
    }
  );
  if (!result.supported) return result;
  const {channels} = result.plan;
  const fill = convertColorChannel(channels.fillColor);
  if ('reason' in fill) return {supported: false, reasons: [fill.reason]};
  const stroke = channels.strokeColor ? convertColorChannel(channels.strokeColor) : null;
  const width = convertScalarChannel(
    channels.size,
    options.constants?.lineWidth ?? VECTOR_TILE_DEFAULT_LINE_WIDTH
  );
  const radius = convertScalarChannel(
    channels.pointSize,
    options.constants?.radius ?? VECTOR_TILE_DEFAULT_RADIUS
  );
  const reasons = [stroke, width, radius].flatMap(entry =>
    entry && 'reason' in entry ? [entry.reason] : []
  );
  if (reasons.length) return {supported: false, reasons};
  return {
    supported: true,
    render: result.plan.render,
    style: fill.style,
    channels: {
      stroke: stroke && 'style' in stroke ? stroke.style : null,
      width: (width as {style: VectorTileScalarStyle}).style,
      radius: (radius as {style: VectorTileScalarStyle}).style
    }
  };
}

const SCALE_CODES: Readonly<Record<VectorTileScaleKind, number>> = {
  linear: 0,
  sqrt: 1,
  log: 2,
  quantile: 3,
  threshold: 3,
  'equal-interval': 4,
  ordinal: 5
};
const SCALAR_SCALE_CODES = {linear: 0, sqrt: 1, log: 2} as const;
const OUTSIDE_CODES = {extrapolate: 0, clamp: 1, reject: 2} as const;
/** Stroke color header mode that copies the fill color. */
const COPY_FILL_MODE = 2;

export type PackVectorTileParametersOptions = {
  rowCount: number;
  /** Column slot of each Float32 property in the tile's column buffer. */
  fieldIndexByName: ReadonlyMap<string, number>;
  filters: readonly DecodedTileFilter[];
  style: VectorTileGpuStyle;
  channels?: VectorTileExtraChannels;
};

function writeColorBlock(
  words: Uint32Array,
  floats: Float32Array,
  offsets: {header: number; palette: number; domain: number},
  style: VectorTileGpuStyle,
  fieldIndexByName: ReadonlyMap<string, number>
): void {
  const {header} = offsets;
  if (style.kind === 'constant') {
    words[header] = 0;
    words[header + 7] = style.color;
    return;
  }
  words[header] = 1;
  words[header + 1] = fieldIndexByName.get(style.fieldName) ?? 0xffffffff;
  words[header + 2] = SCALE_CODES[style.scale];
  const palette = style.palette.slice(0, VECTOR_TILE_MAXIMUM_PALETTE);
  const domain = style.domain.slice(0, VECTOR_TILE_MAXIMUM_DOMAIN);
  words[header + 3] = palette.length;
  words[header + 4] = domain.length;
  words[header + 5] = style.interpolation === 'step' ? 1 : 0;
  words[header + 6] = OUTSIDE_CODES[style.outsideDomain];
  words[header + 8] = style.noDataColor;
  palette.forEach((color, index) => {
    words[offsets.palette + index] = color >>> 0;
  });
  domain.forEach((value, index) => {
    floats[offsets.domain + index] = value;
  });
}

function writeScalarBlock(
  words: Uint32Array,
  floats: Float32Array,
  offset: number,
  style: VectorTileScalarStyle,
  fieldIndexByName: ReadonlyMap<string, number>
): void {
  if (style.kind === 'constant') {
    words[offset] = 0;
    floats[offset + 3] = style.value;
    return;
  }
  words[offset] = 1;
  words[offset + 1] = fieldIndexByName.get(style.fieldName) ?? 0xffffffff;
  words[offset + 2] = SCALAR_SCALE_CODES[style.scale];
  floats[offset + 4] = style.domain[0];
  floats[offset + 5] = style.domain[1];
  floats[offset + 6] = style.range[0];
  floats[offset + 7] = style.range[1];
  floats[offset + 8] = style.nullValue;
}

/**
 * Packs filters and style into the kernel parameter words. Filters on a field missing from the
 * tile reference slot 0xffffffff, which the kernel treats as NaN (the row fails the filter),
 * matching `computeDecodedTileFilterMask`.
 */
export function packVectorTileParameters(options: PackVectorTileParametersOptions): Uint32Array {
  const words = new Uint32Array(VECTOR_TILE_PARAMETER_WORD_COUNT);
  const floats = new Float32Array(words.buffer);
  const filters = options.filters.slice(0, VECTOR_TILE_MAXIMUM_FILTERS);
  words[0] = filters.length;
  words[10] = options.rowCount;
  filters.forEach((filter, index) => {
    const base = FILTER_OFFSET + index * FILTER_WORD_COUNT;
    words[base] = filter.kind === 'range' ? 0 : 1;
    words[base + 1] = options.fieldIndexByName.get(filter.field) ?? 0xffffffff;
    if (filter.kind === 'range') {
      floats[base + 2] = filter.min;
      floats[base + 3] = filter.max;
    } else {
      const values = filter.values.slice(0, VECTOR_TILE_MAXIMUM_SELECT_VALUES);
      words[base + 4] = values.length;
      values.forEach((value, valueIndex) => {
        floats[base + 5 + valueIndex] = value;
      });
    }
  });
  writeColorBlock(
    words,
    floats,
    {header: 1, palette: PALETTE_OFFSET, domain: DOMAIN_OFFSET},
    options.style,
    options.fieldIndexByName
  );
  const {channels} = options;
  if (channels?.stroke)
    writeColorBlock(
      words,
      floats,
      {header: STROKE_HEADER_OFFSET, palette: STROKE_PALETTE_OFFSET, domain: STROKE_DOMAIN_OFFSET},
      channels.stroke,
      options.fieldIndexByName
    );
  else words[STROKE_HEADER_OFFSET] = COPY_FILL_MODE;
  writeScalarBlock(
    words,
    floats,
    WIDTH_OFFSET,
    channels?.width ?? {kind: 'constant', value: VECTOR_TILE_DEFAULT_LINE_WIDTH},
    options.fieldIndexByName
  );
  writeScalarBlock(
    words,
    floats,
    RADIUS_OFFSET,
    channels?.radius ?? {kind: 'constant', value: VECTOR_TILE_DEFAULT_RADIUS},
    options.fieldIndexByName
  );
  return words;
}

function transformValue(scale: number, value: number): number {
  return scale === 1 ? Math.sqrt(Math.max(value, 0)) : scale === 2 ? Math.log(value) : value;
}

function mixChannel(first: number, second: number, amount: number, shift: number): number {
  const left = (first >>> shift) & 255;
  const right = (second >>> shift) & 255;
  return Math.floor(left + (right - left) * amount + 0.5) & 255;
}

function mixColors(first: number, second: number, amount: number): number {
  return packVectorTileColor(
    mixChannel(first, second, amount, 0),
    mixChannel(first, second, amount, 8),
    mixChannel(first, second, amount, 16),
    mixChannel(first, second, amount, 24)
  );
}

/** CPU mirror of the kernel's color function, used by specs and for legend/readback comparison. */
export function evaluateVectorTileStyleReference(style: VectorTileGpuStyle, value: number): number {
  if (style.kind === 'constant') return style.color;
  const {palette, domain} = style;
  const scale = SCALE_CODES[style.scale];
  const paletteCount = palette.length;
  const domainCount = domain.length;
  if (!Number.isFinite(value) || paletteCount === 0) return style.noDataColor;
  if (scale === 5) {
    const code = Math.floor(value);
    return value < 0 || code >= paletteCount ? style.noDataColor : palette[code];
  }
  if (domainCount === 0) return style.noDataColor;
  if (scale === 2 && value <= 0) return style.noDataColor;
  const outside = OUTSIDE_CODES[style.outsideDomain];
  if (scale === 3) {
    let classIndex = 0;
    for (let index = 1; index < domainCount - 1; index++) if (domain[index] <= value) classIndex++;
    return palette[Math.min(classIndex, paletteCount - 1)];
  }
  const transformed = transformValue(scale, value);
  const first = transformValue(scale, domain[0]);
  const last = transformValue(scale, domain[domainCount - 1]);
  let position = last === first ? 0.5 : (transformed - first) / (last - first);
  if ((position < 0 || position > 1) && outside === 2) return style.noDataColor;
  position = clamp(position, 0, 1);
  if (scale === 4 || style.interpolation === 'step')
    return palette[Math.min(Math.floor(position * paletteCount), paletteCount - 1)];
  if (domainCount > 2 && paletteCount === domainCount) {
    let segment = 0;
    while (segment < domainCount - 2 && value >= domain[segment + 1]) segment++;
    const start = transformValue(scale, domain[segment]);
    const end = transformValue(scale, domain[segment + 1]);
    const amount = end === start ? 0 : clamp((transformed - start) / (end - start), 0, 1);
    return mixColors(palette[segment], palette[segment + 1], amount);
  }
  const scaled = position * (paletteCount - 1);
  const lower = Math.min(Math.floor(scaled), paletteCount - 1);
  const upper = Math.min(lower + 1, paletteCount - 1);
  return mixColors(palette[lower], palette[upper], scaled - lower);
}

/** CPU mirror of the kernel's scalar (line width / radius) function. */
export function evaluateVectorTileScalarReference(
  style: VectorTileScalarStyle,
  value: number
): number {
  if (style.kind === 'constant') return style.value;
  const scale = SCALAR_SCALE_CODES[style.scale];
  if (!Number.isFinite(value) || (scale === 2 && value <= 0)) return style.nullValue;
  const first = transformValue(scale, style.domain[0]);
  const last = transformValue(scale, style.domain[1]);
  const position = last === first ? 0.5 : (transformValue(scale, value) - first) / (last - first);
  const clamped = clamp(position, 0, 1);
  return style.range[0] + (style.range[1] - style.range[0]) * clamped;
}

/** CPU mirror of the kernel's filter mask for one row of column values. */
export function evaluateVectorTileFilterReference(
  filters: readonly DecodedTileFilter[],
  getValue: (fieldName: string) => number
): number {
  for (const filter of filters.slice(0, VECTOR_TILE_MAXIMUM_FILTERS)) {
    const value = getValue(filter.field);
    const passes =
      filter.kind === 'range'
        ? value >= filter.min && value <= filter.max
        : filter.values.slice(0, VECTOR_TILE_MAXIMUM_SELECT_VALUES).includes(value);
    if (!passes) return 0;
  }
  return 1;
}

/**
 * Filter + style kernel. Bindings: parameters, columns (fieldSlot * rowCount + row), mask, fill colors,
 * stroke colors, line widths and point radii.
 */
export const VECTOR_TILE_KERNEL_SOURCE = /* wgsl */ `
const FILTER_OFFSET: u32 = ${FILTER_OFFSET}u;
const FILTER_WORD_COUNT: u32 = ${FILTER_WORD_COUNT}u;
const MAXIMUM_FILTERS: u32 = ${VECTOR_TILE_MAXIMUM_FILTERS}u;
const PALETTE_OFFSET: u32 = ${PALETTE_OFFSET}u;
const DOMAIN_OFFSET: u32 = ${DOMAIN_OFFSET}u;
const STROKE_HEADER_OFFSET: u32 = ${STROKE_HEADER_OFFSET}u;
const STROKE_PALETTE_OFFSET: u32 = ${STROKE_PALETTE_OFFSET}u;
const STROKE_DOMAIN_OFFSET: u32 = ${STROKE_DOMAIN_OFFSET}u;
const WIDTH_OFFSET: u32 = ${WIDTH_OFFSET}u;
const RADIUS_OFFSET: u32 = ${RADIUS_OFFSET}u;
const MISSING_FIELD: u32 = 0xffffffffu;

@group(0) @binding(0) var<storage, read> parameters: array<u32>;
@group(0) @binding(1) var<storage, read> columns: array<f32>;
@group(0) @binding(2) var<storage, read_write> maskOutput: array<u32>;
@group(0) @binding(3) var<storage, read_write> colorOutput: array<u32>;
@group(0) @binding(4) var<storage, read_write> strokeOutput: array<u32>;
@group(0) @binding(5) var<storage, read_write> widthOutput: array<f32>;
@group(0) @binding(6) var<storage, read_write> radiusOutput: array<f32>;

fn isNotFinite(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u;
}

fn readParameterFloat(index: u32) -> f32 {
  return bitcast<f32>(parameters[index]);
}

fn readColumn(fieldSlot: u32, row: u32, rowCount: u32) -> f32 {
  // WGSL rejects a constant-expression NaN, so derive the bits from a runtime parameter word (filter counts stay far below the NaN mantissa bits).
  if (fieldSlot == MISSING_FIELD) { return bitcast<f32>(0x7fc00000u | parameters[0]); }
  return columns[fieldSlot * rowCount + row];
}

fn passesFilters(row: u32, rowCount: u32) -> bool {
  let filterCount = min(parameters[0], MAXIMUM_FILTERS);
  for (var filterIndex = 0u; filterIndex < filterCount; filterIndex++) {
    let base = FILTER_OFFSET + filterIndex * FILTER_WORD_COUNT;
    let value = readColumn(parameters[base + 1u], row, rowCount);
    if (isNotFinite(value)) { return false; }
    if (parameters[base] == 0u) {
      if (!(value >= readParameterFloat(base + 2u) && value <= readParameterFloat(base + 3u))) { return false; }
    } else {
      var found = false;
      for (var valueIndex = 0u; valueIndex < min(parameters[base + 4u], 7u); valueIndex++) {
        if (readParameterFloat(base + 5u + valueIndex) == value) { found = true; }
      }
      if (!found) { return false; }
    }
  }
  return true;
}

fn transformValue(scale: u32, value: f32) -> f32 {
  if (scale == 1u) { return sqrt(max(value, 0.0)); }
  if (scale == 2u) { return log(value); }
  return value;
}

fn mixChannel(first: u32, second: u32, amount: f32, shift: u32) -> u32 {
  let left = f32((first >> shift) & 255u);
  let right = f32((second >> shift) & 255u);
  return u32(floor(left + (right - left) * amount + 0.5)) & 255u;
}

fn mixColors(first: u32, second: u32, amount: f32) -> u32 {
  return mixChannel(first, second, amount, 0u)
    | (mixChannel(first, second, amount, 8u) << 8u)
    | (mixChannel(first, second, amount, 16u) << 16u)
    | (mixChannel(first, second, amount, 24u) << 24u);
}

fn readDomain(domainBase: u32, index: u32) -> f32 { return readParameterFloat(domainBase + index); }

// Color block header words: mode, field, scale, paletteCount, domainCount, interpolation, outside, constant, noData.
fn getFieldColor(header: u32, paletteBase: u32, domainBase: u32, row: u32, rowCount: u32) -> u32 {
  let noDataColor = parameters[header + 8u];
  let paletteCount = parameters[header + 3u];
  let domainCount = parameters[header + 4u];
  let scale = parameters[header + 2u];
  let value = readColumn(parameters[header + 1u], row, rowCount);
  if (isNotFinite(value) || paletteCount == 0u) { return noDataColor; }
  if (scale == 5u) {
    if (value < 0.0 || u32(value) >= paletteCount) { return noDataColor; }
    return parameters[paletteBase + u32(value)];
  }
  if (domainCount == 0u) { return noDataColor; }
  if (scale == 2u && value <= 0.0) { return noDataColor; }
  if (scale == 3u) {
    var classIndex = 0u;
    for (var index = 1u; index + 1u < domainCount; index++) {
      if (readDomain(domainBase, index) <= value) { classIndex++; }
    }
    return parameters[paletteBase + min(classIndex, paletteCount - 1u)];
  }
  let transformed = transformValue(scale, value);
  let first = transformValue(scale, readDomain(domainBase, 0u));
  let last = transformValue(scale, readDomain(domainBase, domainCount - 1u));
  var position = 0.5;
  if (last != first) { position = (transformed - first) / (last - first); }
  if ((position < 0.0 || position > 1.0) && parameters[header + 6u] == 2u) { return noDataColor; }
  position = clamp(position, 0.0, 1.0);
  if (scale == 4u || parameters[header + 5u] == 1u) {
    return parameters[paletteBase + min(u32(floor(position * f32(paletteCount))), paletteCount - 1u)];
  }
  if (domainCount > 2u && paletteCount == domainCount) {
    var segment = 0u;
    while (segment + 2u < domainCount && value >= readDomain(domainBase, segment + 1u)) { segment++; }
    let start = transformValue(scale, readDomain(domainBase, segment));
    let end = transformValue(scale, readDomain(domainBase, segment + 1u));
    var amount = 0.0;
    if (end != start) { amount = clamp((transformed - start) / (end - start), 0.0, 1.0); }
    return mixColors(parameters[paletteBase + segment], parameters[paletteBase + segment + 1u], amount);
  }
  let scaled = position * f32(paletteCount - 1u);
  let lower = min(u32(floor(scaled)), paletteCount - 1u);
  let upper = min(lower + 1u, paletteCount - 1u);
  return mixColors(parameters[paletteBase + lower], parameters[paletteBase + upper], scaled - f32(lower));
}

// Scalar block words: mode, field, scale, constant, domainStart, domainEnd, rangeStart, rangeEnd, nullValue.
fn getScalarValue(base: u32, row: u32, rowCount: u32) -> f32 {
  if (parameters[base] == 0u) { return readParameterFloat(base + 3u); }
  let nullValue = readParameterFloat(base + 8u);
  let value = readColumn(parameters[base + 1u], row, rowCount);
  let scale = parameters[base + 2u];
  if (isNotFinite(value)) { return nullValue; }
  if (scale == 2u && value <= 0.0) { return nullValue; }
  let first = transformValue(scale, readParameterFloat(base + 4u));
  let last = transformValue(scale, readParameterFloat(base + 5u));
  var position = 0.5;
  if (last != first) { position = (transformValue(scale, value) - first) / (last - first); }
  position = clamp(position, 0.0, 1.0);
  let low = readParameterFloat(base + 6u);
  let high = readParameterFloat(base + 7u);
  return low + (high - low) * position;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalInvocationId: vec3<u32>) {
  let row = globalInvocationId.x;
  let rowCount = parameters[10];
  if (row >= rowCount) { return; }
  maskOutput[row] = select(0u, 1u, passesFilters(row, rowCount));
  var fillColor = parameters[8];
  if (parameters[1] != 0u) {
    fillColor = getFieldColor(1u, PALETTE_OFFSET, DOMAIN_OFFSET, row, rowCount);
  }
  colorOutput[row] = fillColor;
  var strokeColor = fillColor;
  let strokeMode = parameters[STROKE_HEADER_OFFSET];
  if (strokeMode == 0u) {
    strokeColor = parameters[STROKE_HEADER_OFFSET + 7u];
  } else if (strokeMode == 1u) {
    strokeColor = getFieldColor(STROKE_HEADER_OFFSET, STROKE_PALETTE_OFFSET, STROKE_DOMAIN_OFFSET, row, rowCount);
  }
  strokeOutput[row] = strokeColor;
  widthOutput[row] = getScalarValue(WIDTH_OFFSET, row, rowCount);
  radiusOutput[row] = getScalarValue(RADIUS_OFFSET, row, rowCount);
}
`;
