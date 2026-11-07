// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {isPolygonFootprintLayerType} from './cell-layer-types';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import {packGPUColor} from '@luma-contributors/analysis/dataframe-analysis';
import {
  validateKeplerResidentStylePlan,
  type KeplerResidentStyleDomain,
  type KeplerResidentStylePlan
} from '@luma-contributors/kepler';

type NativeField = {name: string};
type NativeMapColor = string | readonly number[];
type NativeColorRange = {
  colors: readonly string[];
  colorMap?: readonly (readonly [unknown, NativeMapColor])[] | null;
};
export type NativeStyleLayer = {
  id: string;
  type: string;
  /** Pinned Icon declares object|null; the icon-source adapter validates flattened mesh entries. */
  iconGeometry?: object | null;
  iconGeometryVersion?: number;
  config: {
    dataId?: string | null;
    columns?: Readonly<Record<string, {value: string | null; fieldIdx?: number}>>;
    color: readonly number[];
    colorField?: NativeField | null;
    colorScale?: PropertyKey | null;
    colorDomain?: unknown;
    strokeColorField?: NativeField | null;
    strokeColorScale?: PropertyKey | null;
    strokeColorDomain?: unknown;
    sizeField?: NativeField | null;
    sizeScale?: PropertyKey | null;
    sizeDomain?: unknown;
    radiusField?: NativeField | null;
    radiusScale?: PropertyKey | null;
    radiusDomain?: unknown;
    coverageField?: NativeField | null;
    coverageScale?: PropertyKey | null;
    coverageDomain?: unknown;
    heightField?: NativeField | null;
    heightScale?: PropertyKey | null;
    heightDomain?: unknown;
    elevationOffsetField?: NativeField | null;
    elevationOffsetScale?: PropertyKey | null;
    elevationOffsetDomain?: unknown;
    visConfig: {
      filled?: boolean;
      stroked?: boolean;
      outline?: boolean;
      billboard?: boolean;
      colorRange?: NativeColorRange;
      strokeColorRange?: NativeColorRange;
      strokeColor?: readonly number[] | null;
      targetColor?: readonly number[] | null;
      radius?: number;
      fixedRadius?: boolean;
      radiusRange?: readonly number[];
      sizeRange?: readonly number[];
      thickness?: number;
      opacity?: number;
      strokeOpacity?: number;
      coverage?: number;
      coverageRange?: readonly number[];
      enable3d?: boolean;
      fixedHeight?: boolean;
      invertHeight?: boolean;
      heightRange?: readonly number[];
      fixedElevation?: boolean;
      elevationOffset?: number | null;
      elevationOffsetRange?: readonly number[];
      elevationScale?: number;
      enableElevationZoomFactor?: boolean;
    };
  };
};

export type NativeStyleChannel =
  | 'fillColor'
  | 'targetColor'
  | 'strokeColor'
  | 'size'
  | 'pointSize'
  | 'coverage'
  | 'height'
  | 'elevationOffset';
type PropertyName =
  | 'fillColor'
  | 'lineColor'
  | 'lineWidth'
  | 'radius'
  | 'elevation'
  | 'elevationOffset';
export type NativeStylePropertySource<Format extends 'unorm8x4' | 'float32'> = {
  readonly values: GPUVector<Format>;
  /** Native property presence (finite numeric for elevationOffset, truthy for other properties). */
  readonly validity: GPUVector<'uint32'>;
};
/**
 * Declared output of the pinned native seeded Jenks oracle, never an extent or distribution.
 * This differs from exact Fisher-Jenks: [0..19] with five classes gives [4,8,11,16]
 * instead of [4,8,12,16]. Deriving native controls requires its sequential Float64
 * candidate objective; the resident exact natural-breaks analysis remains a separate method.
 * Controls support 1–63 requested classes and exactly representable Float32 boundaries.
 */
export type NativeJenksThresholds = {
  source: {datasetId: string; revision: number};
  column: string;
  classCount: number;
  values: readonly number[];
  /** Absent for a full source domain; otherwise must match the current DOMAIN selection. */
  selectionRevision?: number;
};
export type NativeStylePlanContext = {
  source: {datasetId: string; revision: number};
  /** Metadata from the upload boundary, never a scan of source rows. */
  columns: Readonly<Record<string, {format: string; dictionary?: readonly (string | number)[]}>>;
  /** Omit to translate the native domain. Fixed overrides are already resident stops/edges.
   * Selected domains use the DOMAIN mask, independently of display filtering.
   * Custom threshold scales always use their explicit native controls. */
  domains?: Partial<Record<NativeStyleChannel, KeplerResidentStyleDomain>>;
  /** Native seeded Jenks is not Fisher-Jenks DP. Plans consume revision-matched controls, supplied explicitly or derived by the host's bounded readback (native-jenks-reader). */
  jenksThresholds?: Partial<
    Record<'fillColor' | 'strokeColor' | 'targetColor', NativeJenksThresholds>
  >;
  zoom: number;
  zoomOffset?: number;
  geojsonPointFeatures?: boolean;
  geojsonPolygonFeatures?: boolean;
  geojsonFixedRadius?: boolean;
  /** Native H3 auto branch, determined from the current selected population. */
  h3Branch?: 'polygons' | 'columns';
  /** Immutable source properties. Missing/falsey rows use the current native layer fallback. */
  geojsonProperties?: {
    fillColor?: 'absent' | NativeStylePropertySource<'unorm8x4'>;
    lineColor?: 'absent' | NativeStylePropertySource<'unorm8x4'>;
    lineWidth?: 'absent' | NativeStylePropertySource<'float32'>;
    radius?: 'absent' | NativeStylePropertySource<'float32'>;
    elevation?: 'absent' | NativeStylePropertySource<'float32'>;
    elevationOffset?: 'absent' | NativeStylePropertySource<'float32'>;
  };
  /** Explicit opt-in for Trip sources that retain native GeoJSON line properties. */
  tripProperties?: {
    lineColor?: 'absent' | NativeStylePropertySource<'unorm8x4'>;
    lineWidth?: 'absent' | NativeStylePropertySource<'float32'>;
  };
};

export type NativeStyleChannelPlan =
  | {kind: 'constant'; value: number | readonly number[]}
  | {kind: 'resident'; plan: KeplerResidentStylePlan}
  | {
      kind: 'property';
      property: PropertyName;
      source: NativeStylePropertySource<'unorm8x4'> | NativeStylePropertySource<'float32'>;
      fallback: number | readonly number[];
    };

export type NativeStylePlan = {
  channels: Partial<Record<NativeStyleChannel, NativeStyleChannelPlan>>;
  /** Wrapper multipliers apply after encoding; they must never be baked into channel buffers. */
  render: {
    radiusUnits: 'meters';
    radiusScale: number;
    radiusMaxPixels: number;
    widthUnits: 'meters' | 'pixels';
    widthScale: number;
    opacity: number;
    strokeOpacity: number;
    /** H3 multipliers/visibility remain renderer controls, separate from source-row encodings. */
    coverage?: number;
    elevationScale?: number;
    extruded?: boolean;
    /** Native uniform/field activation; embedded offsets still apply when this is false. */
    elevationOffsetActive?: boolean;
    filled?: boolean;
    stroked?: boolean;
  };
};
export type NativeStylePlanResult =
  | {supported: true; plan: NativeStylePlan}
  | {supported: false; reasons: readonly string[]};

/** Pure pinned Kepler 3.3 translation. This does not upload, execute, or read back source data. */
export function createNativeStylePlan(
  layer: NativeStyleLayer,
  context: NativeStylePlanContext
): NativeStylePlanResult {
  const reasons: string[] = [];
  const channels: NativeStylePlan['channels'] = {};
  const {config} = layer;
  const {visConfig} = config;
  // S2, A5 and geohash layers share GeoJSON's channels and render published polygon footprints.
  const geojson = isPolygonFootprintLayerType(layer.type);
  const trip = layer.type === 'trip';
  const icon = layer.type === 'icon';
  const h3 = layer.type === 'hexagonId';
  if (!['point', 'line', 'arc', 'trip', 'icon', 'hexagonId'].includes(layer.type) && !geojson)
    return {supported: false, reasons: [`No resident style translator for ${layer.type}.`]};
  if (h3 && !context.h3Branch)
    return {
      supported: false,
      reasons: ['H3 styles require a current selected-population rendering branch.']
    };
  if (config.dataId !== context.source.datasetId)
    return {
      supported: false,
      reasons: ['The style source does not match the native layer dataset.']
    };

  function addChannel(channel: NativeStyleChannel, prepare: () => NativeStyleChannelPlan): void {
    try {
      channels[channel] = prepare();
    } catch (error) {
      reasons.push(`${channel}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  function constant(
    value: number | readonly number[],
    property?: PropertyName
  ): NativeStyleChannelPlan {
    if (!property) return {kind: 'constant', value};
    const binding =
      trip && property === 'lineColor'
        ? context.tripProperties?.lineColor
        : trip && property === 'lineWidth'
          ? context.tripProperties?.lineWidth
          : context.geojsonProperties?.[property];
    if (!binding)
      throw new Error(
        `${trip ? 'Trip' : 'GeoJSON'} ${property} property presence requires upload metadata.`
      );
    return binding === 'absent'
      ? {kind: 'constant', value}
      : {kind: 'property', property, source: binding, fallback: value};
  }
  function domain(channel: NativeStyleChannel, nativeDomain: unknown): KeplerResidentStyleDomain {
    const override = context.domains?.[channel];
    if (override) return override;
    if (!Array.isArray(nativeDomain) || nativeDomain.some(value => typeof value !== 'number'))
      throw new Error(
        'Zoom-dependent or missing domains require an explicit resident domain binding.'
      );
    return {kind: 'fixed', values: nativeDomain.map(normalizeFloat32)};
  }
  function sourceColumn(field: NativeField): NativeStylePlanContext['columns'][string] {
    const column = context.columns[field.name];
    if (!column) throw new Error(`No resident source column for ${field.name}.`);
    return column;
  }
  function makeResident(plan: KeplerResidentStylePlan): NativeStyleChannelPlan {
    validateKeplerResidentStylePlan(plan);
    return {kind: 'resident', plan};
  }
  function color(channel: 'fillColor' | 'strokeColor' | 'targetColor'): NativeStyleChannelPlan {
    const stroke = channel === 'strokeColor';
    const field = stroke ? config.strokeColorField : config.colorField;
    const fallback = stroke
      ? (visConfig.strokeColor ?? config.color)
      : channel === 'targetColor'
        ? (visConfig.targetColor ?? config.color)
        : config.color;
    if (!field)
      return constant(
        fallback,
        geojson
          ? stroke
            ? 'lineColor'
            : 'fillColor'
          : trip && context.tripProperties
            ? 'lineColor'
            : undefined
      );
    const column = sourceColumn(field);
    const scale = String((stroke ? config.strokeColorScale : config.colorScale) ?? 'quantile');
    const nativeDomain = stroke ? config.strokeColorDomain : config.colorDomain;
    const colorRange = stroke ? visConfig.strokeColorRange : visConfig.colorRange;
    if (
      !colorRange ||
      (!colorRange.colors.length && !(scale === 'custom' && colorRange.colorMap?.length))
    )
      throw new Error('A native palette is required.');
    const base = {id: `${layer.id}:${channel}`, source: context.source, column: field.name};
    if (scale === 'ordinal' || scale === 'customOrdinal') {
      if (column.format !== 'uint32' || !column.dictionary)
        throw new Error('Categorical styles require explicit dictionary codes and labels.');
      if (context.domains?.[channel]?.kind === 'selection')
        throw new Error('Selected categorical domains require explicit native labels.');
      // Trip layers keep Kepler's uncomputed numeric default domain ([0, 1]) for categorical
      // fields; no label matches it, so the domain is derived from the dictionary like native does.
      const dictionary = column.dictionary;
      const hasComputedDomain =
        !Array.isArray(nativeDomain) || nativeDomain.some(label => dictionary.includes(label));
      const nativeLabels = getCategoryDomain(
        hasComputedDomain ? nativeDomain : undefined,
        dictionary
      );
      if (scale === 'ordinal' && column.dictionary.some(label => !nativeLabels.includes(label)))
        throw new Error('Implicit ordinal category growth requires a complete native domain.');
      const mapping = makeCategoryColors(nativeLabels, colorRange, scale === 'customOrdinal');
      const labels = [...mapping.keys()];
      const codes = labels.map(label => column.dictionary!.indexOf(label));
      return makeResident({
        ...base,
        kind: 'color',
        scale: 'ordinal',
        domain: {kind: 'fixed', values: codes.filter(code => code >= 0)},
        palette: column.dictionary.map(label => mapping.get(label) ?? 0),
        legend: {labels: labels.map(String), codes},
        noDataColor: 0,
        outsideDomain: 'reject'
      });
    }
    if (column.format !== 'float32')
      throw new Error('Numeric styles require a prepared Float32 column.');
    if (scale === 'jenks') {
      const controls = context.jenksThresholds?.[channel];
      const population = context.domains?.[channel];
      const selectionRevision = population?.kind === 'selection' ? population.revision : undefined;
      if (!controls)
        throw new Error(
          'Native Jenks requires explicit or host-derived thresholds for the current revision.'
        );
      if (
        controls.source.datasetId !== context.source.datasetId ||
        controls.source.revision !== context.source.revision ||
        controls.column !== field.name ||
        controls.classCount !== colorRange.colors.length ||
        controls.selectionRevision !== selectionRevision
      )
        throw new Error(
          'Native Jenks thresholds do not match the source, field, palette or domain selection revision.'
        );
      if (
        !Number.isInteger(controls.classCount) ||
        controls.classCount < 1 ||
        controls.classCount > 63 ||
        controls.values.length >= controls.classCount
      )
        throw new Error(
          'Native Jenks controls support 1 to 63 classes and at most classCount minus one thresholds.'
        );
      const palette = colorRange.colors.slice(0, controls.values.length + 1).map(parseNativeColor);
      return makeResident({
        ...base,
        kind: 'color',
        scale: 'threshold',
        domain: {
          kind: 'fixed',
          values: [...controls.values],
          ...(selectionRevision !== undefined ? {selectionRevision} : {})
        },
        palette,
        noDataColor: 0,
        interpolation: 'step',
        outsideDomain: 'clamp'
      });
    }
    if (scale === 'custom' || scale === 'threshold') {
      // Kepler custom palettes own their thresholds. Source/selection domain overrides must not
      // replace these explicit controls or reinterpret them as a quantile sample distribution.
      const threshold = makeNativeThresholdControls(nativeDomain, colorRange, scale === 'custom');
      return makeResident({
        ...base,
        kind: 'color',
        scale: 'quantile',
        domain: {kind: 'fixed', values: threshold.edges},
        palette: threshold.palette,
        noDataColor: 0,
        interpolation: 'step',
        outsideDomain: 'clamp'
      });
    }
    if (!['linear', 'sqrt', 'log', 'quantile', 'quantize'].includes(scale))
      throw new Error(`Native ${scale} colors have no resident binding.`);
    let colorDomain = domain(channel, nativeDomain);
    let palette = colorRange.colors.map(parseNativeColor);
    if (scale === 'quantile' && colorDomain.kind === 'fixed' && !context.domains?.[channel]) {
      // Native quantile domain is a sample distribution, not class edges. Only these supplied
      // metadata values are sorted; this function never requests or scans source column rows.
      const samples = [...colorDomain.values].sort((left, right) => left - right);
      if (!samples.length) throw new Error('Empty quantile domains have no resident binding.');
      colorDomain = {
        kind: 'fixed',
        values: Array.from({length: palette.length + 1}, (_, index) =>
          normalizeFloat32(quantileSorted(samples, index / palette.length))
        )
      };
    } else if (scale === 'linear' || scale === 'sqrt' || scale === 'log') {
      // d3 uses min(domain.length, range.length) for piecewise continuous scales.
      const length =
        colorDomain.kind === 'fixed' ? Math.min(colorDomain.values.length, palette.length) : 2;
      palette = palette.slice(0, length);
      if (colorDomain.kind === 'fixed')
        colorDomain = {...colorDomain, values: colorDomain.values.slice(0, length)};
    }
    return makeResident({
      ...base,
      kind: 'color',
      scale:
        scale === 'quantize' ? 'equal-interval' : (scale as 'linear' | 'sqrt' | 'log' | 'quantile'),
      domain: colorDomain,
      palette,
      noDataColor: 0,
      interpolation: scale === 'quantile' || scale === 'quantize' ? 'step' : 'linear',
      outsideDomain: scale === 'quantile' || scale === 'quantize' ? 'clamp' : 'extrapolate'
    });
  }
  function scalar(
    channel: 'size' | 'pointSize' | 'coverage' | 'height' | 'elevationOffset'
  ): NativeStyleChannelPlan {
    const radius = channel === 'pointSize' || layer.type === 'point' || icon;
    const coverage = channel === 'coverage';
    const height = channel === 'height';
    const elevationOffset = channel === 'elevationOffset';
    // Native flat polygons ignore height; offset-only extrusion explicitly overrides it with zero.
    if (height && !visConfig.enable3d) return constant(0);
    // Native high-precision H3 polygons have no getCoverage accessor; only uniform coverage
    // reaches that branch. Do not validate or execute a coverage field that native ignores.
    if (coverage && context.h3Branch === 'polygons') return constant(1);
    const field = coverage
      ? config.coverageField
      : height
        ? config.heightField
        : elevationOffset
          ? config.elevationOffsetField
          : channel === 'pointSize'
            ? config.radiusField
            : config.sizeField;
    if (!field)
      return constant(
        height || (h3 && channel === 'size')
          ? 500
          : elevationOffset
            ? normalizeFloat32(visConfig.elevationOffset ?? 0)
            : 1,
        geojson
          ? height
            ? 'elevation'
            : elevationOffset
              ? 'elevationOffset'
              : radius
                ? 'radius'
                : 'lineWidth'
          : trip && context.tripProperties
            ? 'lineWidth'
            : undefined
      );
    const column = sourceColumn(field);
    const nativeScale = String(
      (coverage
        ? config.coverageScale
        : height
          ? config.heightScale
          : elevationOffset
            ? config.elevationOffsetScale
            : channel === 'pointSize'
              ? config.radiusScale
              : config.sizeScale) ?? 'linear'
    );
    const fixed =
      (layer.type === 'point' && visConfig.fixedRadius) ||
      ((height || (h3 && channel === 'size')) && visConfig.fixedHeight) ||
      (elevationOffset && visConfig.fixedElevation);
    const scale = fixed ? 'identity' : nativeScale;
    if (!['linear', 'sqrt', 'log', 'identity', 'point'].includes(scale))
      throw new Error(`Native ${scale} scalar encoding has no resident binding.`);
    let scalarDomain: KeplerResidentStyleDomain;
    const nativeDomain = coverage
      ? config.coverageDomain
      : height
        ? (config.heightDomain ?? [0, 1])
        : elevationOffset
          ? (config.elevationOffsetDomain ?? [0, 1])
          : channel === 'pointSize'
            ? config.radiusDomain
            : config.sizeDomain;
    let legend: KeplerResidentStylePlan['legend'];
    if (scale === 'point') {
      if (column.format !== 'uint32' || !column.dictionary)
        throw new Error('Point scales require explicit dictionary codes and labels.');
      if (context.domains?.[channel]?.kind === 'selection')
        throw new Error('Selected categorical domains require explicit native labels.');
      const labels = getCategoryDomain(nativeDomain, column.dictionary);
      const codes = labels.map(label => column.dictionary!.indexOf(label));
      if (codes.some(code => code < 0))
        throw new Error('Point domain contains labels absent from the source dictionary.');
      scalarDomain = {kind: 'fixed', values: codes};
      legend = {labels: labels.map(String), codes};
    } else {
      if (column.format !== 'float32')
        throw new Error('Numeric styles require a prepared Float32 column.');
      scalarDomain = domain(channel, nativeDomain);
    }
    const nativeRange = coverage
      ? (visConfig.coverageRange ?? [0, 1])
      : height
        ? (visConfig.heightRange ?? [0, 500])
        : elevationOffset
          ? (visConfig.elevationOffsetRange ?? [0, 500])
          : radius
            ? (visConfig.radiusRange ?? [0, 50])
            : (visConfig.sizeRange ?? (h3 ? [0, 500] : [0, 10]));
    if (nativeRange.length !== 2) throw new Error('Scalar ranges require two endpoints.');
    const range: [number, number] = [
      normalizeFloat32(nativeRange[0]),
      normalizeFloat32(nativeRange[1])
    ];
    if (height && visConfig.invertHeight && scale !== 'identity') range.reverse();
    return makeResident({
      id: `${layer.id}:${channel}`,
      source: context.source,
      column: field.name,
      kind: 'scalar',
      scale: scale as 'linear' | 'sqrt' | 'log' | 'identity' | 'point',
      domain: scalarDomain,
      range,
      ...(height && visConfig.invertHeight && scale === 'identity' ? {invert: true} : {}),
      nullValue: 0,
      ...(legend ? {legend} : {})
    });
  }

  if ((!geojson && !h3 && layer.type !== 'point') || visConfig.filled !== false)
    addChannel('fillColor', () => color('fillColor'));
  if (layer.type === 'arc' || layer.type === 'line')
    addChannel('targetColor', () => color('targetColor'));
  if (
    (geojson && visConfig.stroked !== false) ||
    (layer.type === 'point' && visConfig.outline) ||
    (h3 && visConfig.outline && !visConfig.enable3d)
  )
    addChannel('strokeColor', () => color('strokeColor'));
  // Pinned Trip declares a stroked condition for UI, but BaseLayer.getAttributeAccessors
  // never evaluates it: a sizeField always maps getWidth; without it lineWidth || 1 wins.
  if (!geojson || visConfig.stroked !== false) addChannel('size', () => scalar('size'));
  if (geojson && context.geojsonPointFeatures) addChannel('pointSize', () => scalar('pointSize'));
  if (h3) addChannel('coverage', () => scalar('coverage'));
  if (geojson && context.geojsonPolygonFeatures && visConfig.filled !== false) {
    addChannel('height', () => scalar('height'));
    addChannel('elevationOffset', () => scalar('elevationOffset'));
  }
  if (reasons.length) return {supported: false, reasons};
  const zoomFactor = 2 ** Math.max(14 - context.zoom + (context.zoomOffset ?? 0), 0);
  // Icon overrides the base zoom factor: its mesh radius continues shrinking past zoom 14.
  const radiusZoomFactor = icon ? 2 ** (14 - context.zoom + (context.zoomOffset ?? 0)) : zoomFactor;
  const fixedRadius = geojson
    ? context.geojsonFixedRadius
    : icon
      ? visConfig.fixedRadius
      : visConfig.fixedRadius && Boolean(config.sizeField);
  const radiusField = geojson ? config.radiusField : config.sizeField;
  const elevationOffsetActive = Boolean(
    config.elevationOffsetField || (visConfig.elevationOffset ?? 0) > 0
  );
  const geojsonExtruded = Boolean(visConfig.enable3d || elevationOffsetActive);
  return {
    supported: true,
    plan: {
      channels,
      render: {
        radiusUnits: 'meters',
        radiusScale: fixedRadius
          ? 1
          : (radiusField ? 1 : (visConfig.radius ?? 10)) * radiusZoomFactor,
        radiusMaxPixels: visConfig.fixedRadius ? Number.MAX_SAFE_INTEGER : 500,
        widthUnits: geojson || trip || h3 ? 'meters' : 'pixels',
        widthScale:
          (visConfig.thickness ?? (trip ? 0.5 : 2)) *
          (h3 ? 1 : geojson || trip ? zoomFactor * 8 : 2 / 3) *
          (trip && visConfig.billboard ? 2 / 3 : 1),
        opacity: visConfig.opacity ?? 0.8,
        strokeOpacity:
          h3 && context.h3Branch === 'polygons'
            ? (visConfig.opacity ?? 0.8)
            : (visConfig.strokeOpacity ?? 0.8),
        ...(geojson
          ? {
              elevationScale:
                (visConfig.elevationScale ?? 1) *
                (visConfig.fixedHeight
                  ? 1
                  : 2 ** Math.max(8 - context.zoom + (context.zoomOffset ?? 0), 0)),
              elevationOffsetActive,
              extruded: geojsonExtruded,
              filled: visConfig.filled !== false,
              stroked: visConfig.stroked !== false && !geojsonExtruded
            }
          : {}),
        ...(h3
          ? {
              coverage: config.coverageField ? 1 : (visConfig.coverage ?? 1),
              elevationScale:
                (visConfig.elevationScale ?? 5) *
                (visConfig.fixedHeight || visConfig.enableElevationZoomFactor === false
                  ? 1
                  : 2 ** Math.max(8 - context.zoom + (context.zoomOffset ?? 0), 0)),
              extruded: Boolean(visConfig.enable3d),
              filled: visConfig.filled !== false,
              stroked: Boolean(visConfig.outline) && !visConfig.enable3d
            }
          : {})
      }
    }
  };
}

function normalizeFloat32(value: number): number {
  const normalized = Math.fround(value);
  if (!Number.isFinite(normalized))
    throw new Error('Style controls must fit finite Float32 values.');
  return normalized;
}

function getCategoryDomain(
  nativeDomain: unknown,
  dictionary: readonly (string | number)[]
): (string | number)[] {
  if (nativeDomain !== undefined && !Array.isArray(nativeDomain))
    throw new Error('Zoom-dependent categorical domains have no resident binding.');
  const labels: unknown[] = nativeDomain === undefined ? [...dictionary].sort() : nativeDomain;
  if (labels.some(label => typeof label !== 'string' && typeof label !== 'number'))
    throw new Error('Categorical styles require string or numeric labels.');
  return [...new Set(labels)] as (string | number)[];
}

function makeCategoryColors(
  labels: readonly (string | number)[],
  range: NativeColorRange,
  custom: boolean
): Map<string | number, number> {
  const result = new Map<string | number, number>();
  if (custom && range.colorMap) {
    for (const [entry, color] of range.colorMap) {
      for (const label of Array.isArray(entry) ? entry : [entry]) {
        if ((typeof label === 'string' || typeof label === 'number') && !result.has(label))
          result.set(label, parseNativeColor(String(color)));
      }
    }
  } else {
    const ordered = custom ? [...labels].sort() : labels;
    ordered.forEach((label, index) => {
      const colorIndex = custom
        ? Math.min(index, range.colors.length - 1)
        : index % range.colors.length;
      result.set(label, parseNativeColor(range.colors[colorIndex]));
    });
  }
  return result;
}

function parseNativeColor(color: string): number {
  const match = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(color);
  // Matches Kepler hexToRgb, including its opaque-black fallback for invalid hex.
  return match
    ? packGPUColor(
        Number.parseInt(match[1], 16),
        Number.parseInt(match[2], 16),
        Number.parseInt(match[3], 16)
      )
    : packGPUColor(0, 0, 0);
}

/** d3 threshold compares with the active prefix only; a final null map key is not a boundary. */
function makeNativeThresholdControls(
  nativeDomain: unknown,
  range: NativeColorRange,
  custom: boolean
): {edges: number[]; palette: number[]} {
  let thresholds: readonly unknown[];
  let palette: number[];
  if (custom && range.colorMap?.length) {
    // Native Map preserves the first insertion position and the last color for duplicate keys.
    const colors = new Map(range.colorMap);
    const unknown = colors.get('__unknownColor__');
    if (unknown !== undefined && parseNativeMapColor(unknown) !== 0)
      throw new Error('Custom unknown colors require separate NaN and null output policies.');
    thresholds = [...colors.keys()];
    palette = [...colors.values()].map(parseNativeMapColor);
  } else {
    if (!Array.isArray(nativeDomain))
      throw new Error('Custom thresholds require an explicit ordered native domain.');
    thresholds = nativeDomain;
    palette = range.colors.map(parseNativeColor);
  }
  const count = Math.min(thresholds.length, palette.length - 1);
  const active = thresholds.slice(0, count);
  if (active.some(value => typeof value !== 'number' || !Number.isFinite(value)))
    throw new Error('Active custom thresholds must be finite numbers.');
  const boundaries = active as number[];
  if (boundaries.some((value, index) => index > 0 && value < boundaries[index - 1]))
    throw new Error('Custom thresholds must be in native ascending order.');
  if (boundaries.some(value => Math.fround(value) !== value))
    throw new Error('Custom thresholds must be exactly representable in Float32.');
  // The fixed quantile kernel counts inner edges with <=, exactly d3 bisectRight. Bookends
  // are classifier padding only: the legend reads the inner thresholds and unbounded tails.
  return {
    edges: count ? [boundaries[0], ...boundaries, boundaries[count - 1]] : [0, 0],
    palette: palette.slice(0, count + 1)
  };
}

function parseNativeMapColor(color: NativeMapColor): number {
  if (typeof color === 'string') return parseNativeColor(color);
  if (
    (color.length !== 3 && color.length !== 4) ||
    color.some(value => !Number.isInteger(value) || value < 0 || value > 255)
  )
    throw new Error('Custom map colors require RGB or RGBA bytes.');
  return packGPUColor(color[0], color[1], color[2], color[3] ?? 255);
}

function quantileSorted(values: readonly number[], probability: number): number {
  const position = (values.length - 1) * probability;
  const lower = Math.floor(position);
  return values[lower] + (values[Math.ceil(position)] - values[lower]) * (position - lower);
}
