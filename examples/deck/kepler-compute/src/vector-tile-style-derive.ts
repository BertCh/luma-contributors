// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Derives resident vector-tile style, draw parameters and filters from the props of the Deck
 * `MVTLayer` that Kepler's vectorTile layer renders (vector-tile-layer.js `renderLayer`).
 *
 * The wrapped Deck layer is the only Kepler object the live controller sees, but Kepler mirrors its
 * live visual channels into `updateTriggers` (`getFillColor`, `getLineColor`, `getLineWidth`,
 * `getPointRadius`, `getFilterValue`) and its draw settings into plain props. Reading them back gives
 * the same channel fields, scales, domains and palettes the native layer used, without a handle on
 * the Kepler layer instance. Pure TypeScript with no device access.
 */

import type {NativeStyleLayer} from './native-style-plan';
import type {DecodedTileFilter} from './vector-tile-decoded-batch';
import {VECTOR_TILE_MAXIMUM_FILTERS} from './vector-tile-resident-style';

export type VectorTileDeckProps = Readonly<Record<string, unknown>>;

/** Draw settings that bypass the GeoJSON-oriented native style render block (vector tiles use pixels). */
export type VectorTileDrawParameters = Readonly<{
  opacity: number;
  strokeOpacity: number;
  stroked: boolean;
  filled: boolean;
  pickable: boolean;
  lineWidthUnits: 'pixels' | 'meters';
  radiusUnits: 'pixels' | 'meters';
  radiusScale: number;
  radiusMinPixels: number;
}>;

export type VectorTileStyleFields = Readonly<{
  /** Property names read as numbers (numeric color/stroke/size/radius fields and filter fields). */
  numeric: readonly string[];
  /** Property names read as category codes (ordinal color scales). */
  categorical: readonly string[];
}>;

export const VECTOR_TILE_DEFAULT_BASE_COLOR: readonly number[] = [18, 147, 154];
const DEFAULT_OPACITY = 0.8;

function readRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' ? (value as Record<string, unknown>) : {};
}

function readField(value: unknown): {name: string} | null {
  const name = readRecord(value)['name'];
  return typeof name === 'string' ? {name} : null;
}

function readNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function readColorRange(value: unknown): NativeStyleLayer['config']['visConfig']['colorRange'] {
  const range = readRecord(value);
  return Array.isArray(range['colors'])
    ? (range as unknown as NativeStyleLayer['config']['visConfig']['colorRange'])
    : undefined;
}

/** True when the props carry Kepler's channel update triggers (otherwise nothing can be derived). */
export function hasVectorTileStyleTriggers(props: VectorTileDeckProps): boolean {
  const triggers = readRecord(props['updateTriggers']);
  return Boolean(triggers['getFillColor'] || triggers['getLineColor'] || triggers['getLineWidth']);
}

/** Builds the native style layer description from the Deck layer's update triggers and draw props. */
export function deriveNativeStyleLayerFromDeckProps(
  layerId: string,
  props: VectorTileDeckProps
): NativeStyleLayer | null {
  if (!hasVectorTileStyleTriggers(props)) return null;
  const triggers = readRecord(props['updateTriggers']);
  const fill = readRecord(triggers['getFillColor']);
  const line = readRecord(triggers['getLineColor']);
  const width = readRecord(triggers['getLineWidth']);
  const radius = readRecord(triggers['getPointRadius']);
  const baseColor = Array.isArray(fill['color']) ? (fill['color'] as number[]) : null;
  const strokeOpacity = readNumber(
    readRecord(readRecord(props['_subLayerProps'])['polygons-stroke'])['opacity']
  );
  return {
    id: layerId,
    type: 'vectorTile',
    config: {
      dataId: layerId,
      color: baseColor ?? VECTOR_TILE_DEFAULT_BASE_COLOR,
      colorField: readField(fill['colorField']),
      colorScale: fill['colorScale'] as string | undefined,
      colorDomain: fill['colorDomain'],
      strokeColorField: readField(line['strokeColorField']),
      strokeColorScale: line['strokeColorScale'] as string | undefined,
      strokeColorDomain: line['strokeColorDomain'],
      sizeField: readField(width['sizeField']),
      sizeScale: width['sizeScale'] as string | undefined,
      sizeDomain: width['sizeDomain'],
      radiusField: readField(radius['radiusField']),
      radiusScale: radius['radiusScale'] as string | undefined,
      radiusDomain: radius['radiusDomain'],
      visConfig: {
        // The kernel always plans the stroke channel; draw parameters decide whether outlines draw.
        stroked: true,
        filled: props['filled'] !== false,
        opacity: readNumber(props['opacity']) ?? DEFAULT_OPACITY,
        strokeOpacity: strokeOpacity ?? readNumber(props['opacity']) ?? DEFAULT_OPACITY,
        colorRange: readColorRange(fill['colorRange']),
        strokeColorRange: readColorRange(line['strokeColorRange']),
        strokeColor: Array.isArray(line['strokeColor']) ? (line['strokeColor'] as number[]) : null,
        sizeRange: Array.isArray(width['sizeRange']) ? (width['sizeRange'] as number[]) : undefined,
        radiusRange: Array.isArray(radius['radiusRange'])
          ? (radius['radiusRange'] as number[])
          : undefined,
        radius: readNumber(radius['radius'])
      }
    }
  };
}

/** Constant line width and point radius used when no field drives the channel. */
export function deriveVectorTileConstants(props: VectorTileDeckProps): {
  lineWidth?: number;
  radius?: number;
} {
  const triggers = readRecord(props['updateTriggers']);
  const lineWidth =
    readNumber(props['getLineWidth']) ??
    readNumber(readRecord(triggers['getLineWidth'])['strokeWidth']);
  const radius =
    readNumber(props['getPointRadius']) ??
    readNumber(readRecord(triggers['getPointRadius'])['radius']);
  return {
    ...(lineWidth !== undefined ? {lineWidth} : {}),
    ...(radius !== undefined ? {radius} : {})
  };
}

export function deriveVectorTileDrawParameters(
  props: VectorTileDeckProps
): VectorTileDrawParameters {
  const opacity = readNumber(props['opacity']) ?? DEFAULT_OPACITY;
  const strokeOpacity =
    readNumber(readRecord(readRecord(props['_subLayerProps'])['polygons-stroke'])['opacity']) ??
    opacity;
  return {
    opacity,
    strokeOpacity,
    stroked: props['stroked'] !== false,
    filled: props['filled'] !== false,
    pickable: props['pickable'] !== false,
    lineWidthUnits: props['lineWidthUnits'] === 'meters' ? 'meters' : 'pixels',
    radiusUnits: props['pointRadiusUnits'] === 'meters' ? 'meters' : 'pixels',
    radiusScale: readNumber(props['pointRadiusScale']) ?? 1,
    radiusMinPixels: readNumber(props['pointRadiusMinPixels']) ?? 0
  };
}

/**
 * Converts Kepler's GPU filter channels into host tile filters. Kepler stores each active filter as
 * `filterRange[i]` relative to `domain0` of its `gpuFilter_i` trigger (value - domain0), so the
 * absolute property range is `range + domain0`. Inactive channels have a null trigger. Time-interval
 * "end" channels are not representable as a property range and are skipped.
 */
export function deriveVectorTileFiltersFromDeckProps(
  props: VectorTileDeckProps
): DecodedTileFilter[] {
  const triggers = readRecord(readRecord(props['updateTriggers'])['getFilterValue']);
  const ranges = props['filterRange'];
  const filters: DecodedTileFilter[] = [];
  if (!Array.isArray(ranges)) return filters;
  for (let index = 0; index < VECTOR_TILE_MAXIMUM_FILTERS; index++) {
    const trigger = readRecord(triggers[`gpuFilter_${index}`]);
    const range = ranges[index];
    if (typeof trigger['name'] !== 'string' || !Array.isArray(range)) continue;
    const offset = readNumber(trigger['domain0']) ?? 0;
    const minimum = readNumber(range[0]);
    const maximum = readNumber(range[1]);
    if (minimum === undefined || maximum === undefined) continue;
    filters.push({
      kind: 'range',
      field: trigger['name'],
      min: minimum + offset,
      max: maximum + offset
    });
  }
  return filters;
}

/** Property names the derived style and filters read, split by how the tile column encodes them. */
export function collectVectorTileStyleFields(
  layer: NativeStyleLayer | null,
  filters: readonly DecodedTileFilter[]
): VectorTileStyleFields {
  const numeric = new Set<string>();
  const categorical = new Set<string>();
  if (layer) {
    const {config} = layer;
    for (const [field, scale] of [
      [config.colorField, config.colorScale],
      [config.strokeColorField, config.strokeColorScale]
    ] as const) {
      if (!field) continue;
      (scale === 'ordinal' || scale === 'customOrdinal' ? categorical : numeric).add(field.name);
    }
    if (config.sizeField) numeric.add(config.sizeField.name);
    if (config.radiusField) numeric.add(config.radiusField.name);
  }
  for (const filter of filters) numeric.add(filter.field);
  return {numeric: [...numeric], categorical: [...categorical]};
}

/**
 * Stable text of everything that feeds the derived style, draw parameters and filters. The live
 * controller compares it on every render and re-derives only when it changes.
 */
export function getVectorTileDeckStyleSignature(props: VectorTileDeckProps): string {
  const triggers = readRecord(props['updateTriggers']);
  return JSON.stringify([
    triggers['getFillColor'],
    triggers['getLineColor'],
    triggers['getLineWidth'],
    triggers['getPointRadius'],
    triggers['getFilterValue'],
    props['filterRange'],
    props['opacity'],
    props['stroked'],
    props['filled'],
    props['pickable'],
    props['lineWidthUnits'],
    props['pointRadiusUnits'],
    props['pointRadiusScale'],
    props['pointRadiusMinPixels'],
    props['getLineWidth'] === undefined || typeof props['getLineWidth'] === 'number'
      ? props['getLineWidth']
      : 'accessor',
    typeof props['getPointRadius'] === 'number' ? props['getPointRadius'] : 'accessor',
    readRecord(readRecord(props['_subLayerProps'])['polygons-stroke'])['opacity']
  ]);
}
