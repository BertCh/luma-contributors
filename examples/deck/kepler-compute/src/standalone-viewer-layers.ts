// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ArcLayer, GeoJsonLayer, IconLayer, LineLayer, ScatterplotLayer} from '@deck.gl/layers';
import {GridLayer, HeatmapLayer, HexagonLayer} from '@deck.gl/aggregation-layers';
import {H3HexagonLayer, TripsLayer} from '@deck.gl/geo-layers';
import {createIndexedLayer} from './indexed-layer';
import {getConnectionLimitReason} from './resident-arc-line-plan';
import {getResidentChannelLegendModel} from './resident-channel-legend-model';
import {isCellLayerType} from './cell-layer-types';
import {cellToLatLng} from 'h3-js';
import {decodeCellFeatures} from './standalone-viewer-cells';
import {buildClusterLayer, type ViewerClusterView} from './standalone-viewer-cluster';
import {
  createCustomColorScale,
  createCustomOrdinalColorScale,
  isCustomColorScale,
  readColorMap
} from './standalone-viewer-color-map';
import type {ViewerIconAtlas} from './standalone-viewer-icons';
import {
  applyPolygonFilters,
  getPolygonFiltersForLayer,
  readGeoJsonGeometry,
  type GeoJsonLike
} from './standalone-viewer-polygon';
import {buildRasterLayer, findRasterTileTemplate, isViewerOnline} from './standalone-viewer-raster';
import {
  parseViewerColor,
  readColorRangeColors,
  type ViewerFilterSpec,
  type ViewerLayerSpec
} from './standalone-viewer-config';
import {selectFilteredRows} from './standalone-viewer-filters';
import type {ViewerDataset} from './standalone-viewer-data';
import {
  createContinuousColorScale,
  createOrdinalColorScale,
  createSizeScale,
  type ColorTriple
} from './standalone-viewer-scales';

import {clamp} from '@math.gl/core';

/** Builds deck.gl layers for every resident layer family from the saved Kepler configuration. */

export type ViewerLegendEntry =
  | {kind: 'ramp'; title: string; colors: ColorTriple[]; domain: number[]}
  | {kind: 'categories'; title: string; items: {label: string; color: ColorTriple}[]}
  | {kind: 'size'; title: string; ticks: {value: number; size: number}[]}
  | {kind: 'constant'; title: string; color: ColorTriple};

export type ViewerLegend = {layerId: string; label: string; entries: ViewerLegendEntry[]};

export type ViewerBuildContext = {
  datasets: ReadonlyMap<string, ViewerDataset>;
  filters: readonly ViewerFilterSpec[];
  /** Animation clock in the trip timestamp unit. */
  currentTime: number;
  /** Selects 'meters' for fixed point radius etc.; kept for symmetry with the host. */
  parseJson(datasetId: string, fieldName: string, row: number): unknown;
  /** Saved Kepler configuration, searched for raster tile templates. */
  nativeConfig?: unknown;
  /** Real icon meshes rasterized into an atlas; the pin mask is used without it. */
  iconAtlas?: ViewerIconAtlas | null;
  /** Current map view, needed by zoom-dependent cluster layers. */
  view?: ViewerClusterView;
};

export type ViewerBuildResult = {
  layers: any[];
  legends: ViewerLegend[];
  skipped: {layerId: string; label: string; reason: string}[];
  bounds: [number, number, number, number] | null;
  tripDomain: [number, number] | null;
  needsWebGL: boolean;
  /** Layers whose geometry depends on the zoom level (cluster). */
  isZoomDependent: boolean;
  /** Maps a picked deck object to dataset and source row. */
  pickTargets: Map<string, string>;
};

const BASIC_NUMBER_TYPES = new Set(['real', 'integer', 'timestamp', 'date']);

function toNumber(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

type ChannelRange = readonly [number, number];

class BoundsAccumulator {
  west = Infinity;
  south = Infinity;
  east = -Infinity;
  north = -Infinity;
  add(longitude: number, latitude: number): void {
    if (!Number.isFinite(longitude) || !Number.isFinite(latitude)) return;
    this.west = Math.min(this.west, longitude);
    this.east = Math.max(this.east, longitude);
    this.south = Math.min(this.south, latitude);
    this.north = Math.max(this.north, latitude);
  }
}

function getRowValues(values: ArrayLike<number>, rows: Uint32Array): Float64Array {
  const picked = new Float64Array(rows.length);
  for (let index = 0; index < rows.length; index++) picked[index] = values[rows[index]];
  return picked;
}

/** Color accessor (by source row) plus legend entry for a color channel. */
function createColorChannel(
  spec: ViewerLayerSpec,
  dataset: ViewerDataset,
  rows: Uint32Array,
  channel: 'color' | 'strokeColor',
  rangeKey: string,
  constant: ColorTriple,
  legends: ViewerLegendEntry[]
): (row: number) => ColorTriple {
  const field = spec.channels[channel].field;
  if (!field || !dataset.fieldNames.includes(field.name)) {
    if (channel === 'color') legends.push({kind: 'constant', title: 'Color', color: constant});
    return () => constant;
  }
  const colors = readColorRangeColors(spec.visConfig[rangeKey]);
  const scaleType = spec.channels[channel].scale;
  const customMap = isCustomColorScale(scaleType, spec.visConfig[rangeKey])
    ? readColorMap(spec.visConfig[rangeKey])
    : null;
  if (customMap && scaleType === 'customOrdinal') {
    const labels = dataset.readStrings(field.name);
    if (labels) {
      const scale = createCustomOrdinalColorScale(customMap);
      legends.push({
        kind: 'categories',
        title: field.name,
        items: customMap.map((entry, index) => ({
          label: Array.isArray(entry[0]) ? entry[0].join(', ') : String(entry[0] ?? 'Other'),
          color: scale.colors[index]
        }))
      });
      return row => (labels[row] === null ? constant : scale.getCategoryColor(labels[row]!));
    }
  }
  if (customMap && scaleType === 'custom') {
    const numbers = dataset.readNumbers(field.name);
    if (numbers) {
      const scale = createCustomColorScale(customMap);
      legends.push({kind: 'ramp', title: field.name, colors: scale.colors, domain: scale.domain});
      return row => (Number.isFinite(numbers[row]) ? scale.getColor(numbers[row]) : constant);
    }
  }
  if (
    field.type === 'string' ||
    field.type === 'boolean' ||
    !BASIC_NUMBER_TYPES.has(field.type ?? 'real')
  ) {
    const labels = dataset.readStrings(field.name)!;
    const scale = createOrdinalColorScale(
      Array.from(rows, row => labels[row]),
      colors
    );
    legends.push({
      kind: 'categories',
      title: field.name,
      items: scale.categories
        .slice(0, 24)
        .map(label => ({label, color: scale.getCategoryColor(label)}))
    });
    return row => (labels[row] === null ? constant : scale.getCategoryColor(labels[row]!));
  }
  const values = dataset.readNumbers(field.name)!;
  const scale = createContinuousColorScale(scaleType, colors, getRowValues(values, rows));
  legends.push({kind: 'ramp', title: field.name, colors: scale.colors, domain: scale.domain});
  return row => (Number.isFinite(values[row]) ? scale.getColor(values[row]) : constant);
}

/** Numeric size accessor (by source row); falls back to a constant without a field. */
function createSizeChannel(
  spec: ViewerLayerSpec,
  dataset: ViewerDataset,
  rows: Uint32Array,
  channel: string,
  range: ChannelRange,
  constant: number,
  legends: ViewerLegendEntry[]
): (row: number) => number {
  const field = spec.channels[channel]?.field;
  if (!field || !dataset.fieldNames.includes(field.name)) return () => constant;
  const values = dataset.readNumbers(field.name)!;
  const scaleType = spec.channels[channel].scale;
  const scale = createSizeScale(scaleType, getRowValues(values, rows), range);
  const model = getResidentChannelLegendModel({
    scale: scaleType ?? 'linear',
    domain: scale.domain,
    range
  });
  if (model.status === 'ready')
    legends.push({kind: 'size', title: field.name, ticks: model.ticks.map(tick => ({...tick}))});
  return row => (Number.isFinite(values[row]) ? scale.getSize(values[row]) : range[0]);
}

function readRange(value: unknown, fallback: ChannelRange): ChannelRange {
  return Array.isArray(value) && value.length === 2
    ? [Number(value[0]), Number(value[1])]
    : fallback;
}

type PointColumns = {longitudes: ArrayLike<number>; latitudes: ArrayLike<number>};

function readPointColumns(
  dataset: ViewerDataset,
  longitudeName: string | null | undefined,
  latitudeName: string | null | undefined
): PointColumns | null {
  const longitudes = longitudeName ? dataset.readNumbers(longitudeName) : null;
  const latitudes = latitudeName ? dataset.readNumbers(latitudeName) : null;
  return longitudes && latitudes ? {longitudes, latitudes} : null;
}

function parseFeatures(
  context: ViewerBuildContext,
  dataset: ViewerDataset,
  fieldName: string,
  rows: Uint32Array
): any[] {
  const features: any[] = [];
  for (const row of rows) {
    const parsed = context.parseJson(dataset.datasetId, fieldName, row);
    if (!parsed || typeof parsed !== 'object') continue;
    const geoJson: GeoJsonLike = parsed;
    const geometry = readGeoJsonGeometry(parsed);
    if (!geometry?.type) continue;
    features.push({
      type: 'Feature',
      geometry,
      properties: {...(geoJson.properties ?? {}), __row: row}
    });
  }
  return features;
}

function collectGeometryBounds(geometry: any, bounds: BoundsAccumulator): void {
  const visit = (coordinates: any): void => {
    if (typeof coordinates?.[0] === 'number') bounds.add(coordinates[0], coordinates[1]);
    else if (Array.isArray(coordinates)) for (const child of coordinates) visit(child);
  };
  if (geometry?.type === 'GeometryCollection')
    geometry.geometries?.forEach((child: any) => collectGeometryBounds(child, bounds));
  else visit(geometry?.coordinates);
}

const AGGREGATION_NAMES: Record<string, string> = {
  count: 'COUNT',
  average: 'MEAN',
  mean: 'MEAN',
  sum: 'SUM',
  minimum: 'MIN',
  maximum: 'MAX',
  min: 'MIN',
  max: 'MAX'
};

/** Reasons a saved layer type has no offline renderer. */
const UNSUPPORTED_TYPES: Record<string, string> = {
  mvt: 'Vector tiles need a network source.',
  '3d': 'Native 3D model layers are not part of the offline viewer.'
};

function describeSkip(spec: ViewerLayerSpec): string | null {
  if (UNSUPPORTED_TYPES[spec.type]) return UNSUPPORTED_TYPES[spec.type];
  if (spec.type === 'arc' || spec.type === 'line') {
    const reason = getConnectionLimitReason(spec.type, {
      columnMode: spec.columnMode,
      columns: Object.fromEntries(
        Object.entries(spec.columns)
          .filter(([role]) => !role.startsWith('alt'))
          .map(([role, value]) => [role, {value}])
      ),
      visConfig: spec.visConfig
    });
    if (reason) return reason;
  }
  return null;
}

export function buildViewerLayers(
  specs: readonly ViewerLayerSpec[],
  context: ViewerBuildContext
): ViewerBuildResult {
  const result: ViewerBuildResult = {
    layers: [],
    legends: [],
    skipped: [],
    bounds: null,
    tripDomain: null,
    needsWebGL: false,
    isZoomDependent: false,
    pickTargets: new Map()
  };
  const bounds = new BoundsAccumulator();
  for (const spec of specs) {
    if (!spec.isVisible) continue;
    const dataset = context.datasets.get(spec.datasetId);
    const skipReason = dataset ? describeSkip(spec) : 'The dataset is not in the bundle.';
    if (spec.type === 'rasterTile') {
      /* raster layers have no table; handled below */
    } else if (!dataset || skipReason) {
      result.skipped.push({layerId: spec.id, label: spec.label, reason: skipReason!});
      continue;
    }
    if (spec.type === 'rasterTile') {
      const template = findRasterTileTemplate(context.nativeConfig);
      const reason = !template
        ? 'Raster tiles come from a tile server and the bundle embeds no pixels or tile URL template.'
        : !isViewerOnline()
          ? 'Raster tiles need a network connection and the page is offline.'
          : null;
      if (reason) {
        result.skipped.push({layerId: spec.id, label: spec.label, reason});
        continue;
      }
      result.layers.push(
        buildRasterLayer({
          id: spec.id,
          opacity: toNumber(spec.visConfig['opacity'], 0.8),
          template: template!
        })
      );
      result.legends.push({layerId: spec.id, label: spec.label, entries: []});
      continue;
    }
    const rows = applyPolygonFilters(
      selectFilteredRows(spec.datasetId, context.filters, dataset!),
      getPolygonFiltersForLayer(context.filters, spec.id, spec.datasetId),
      row => getFilterPoints(spec, dataset!, context, row)
    );
    const legendEntries: ViewerLegendEntry[] = [];
    const layer = buildLayer(spec, dataset!, rows, context, legendEntries, bounds, result);
    if (!layer) {
      result.skipped.push({
        layerId: spec.id,
        label: spec.label,
        reason: `Layer type "${spec.type}" has no offline renderer or its columns are missing.`
      });
      continue;
    }
    result.layers.push(layer);
    result.pickTargets.set(spec.id, spec.datasetId);
    result.legends.push({layerId: spec.id, label: spec.label, entries: legendEntries});
  }
  if (bounds.west <= bounds.east)
    result.bounds = [bounds.west, bounds.south, bounds.east, bounds.north];
  return result;
}

function buildLayer(
  spec: ViewerLayerSpec,
  dataset: ViewerDataset,
  rows: Uint32Array,
  context: ViewerBuildContext,
  legendEntries: ViewerLegendEntry[],
  bounds: BoundsAccumulator,
  result: ViewerBuildResult
): any | null {
  const visConfig = spec.visConfig;
  const opacity = toNumber(visConfig['opacity'], 0.8);
  const baseColor = parseViewerColor(spec.color);
  const rowList = Array.from(rows);
  const common = {id: spec.id, pickable: true, opacity};

  switch (spec.type) {
    case 'point': {
      const columns = readPointColumns(dataset, spec.columns['lng'], spec.columns['lat']);
      if (!columns) return null;
      const altitude = spec.columns['altitude']
        ? dataset.readNumbers(spec.columns['altitude']!)
        : null;
      for (const row of rowList) bounds.add(columns.longitudes[row], columns.latitudes[row]);
      const getFill = createColorChannel(
        spec,
        dataset,
        rows,
        'color',
        'colorRange',
        baseColor,
        legendEntries
      );
      const getStroke = createColorChannel(
        spec,
        dataset,
        rows,
        'strokeColor',
        'strokeColorRange',
        parseViewerColor(visConfig['strokeColor'], baseColor),
        []
      );
      const fixedRadius = Boolean(visConfig['fixedRadius']);
      const getRadius = createSizeChannel(
        spec,
        dataset,
        rows,
        'radius',
        readRange(visConfig['radiusRange'], [1, 40]),
        toNumber(visConfig['radius'], 10),
        legendEntries
      );
      return createIndexedLayer(ScatterplotLayer, {
        ...common,
        data: rowList,
        radiusUnits: fixedRadius ? 'meters' : 'pixels',
        filled: visConfig['filled'] !== false,
        stroked: Boolean(visConfig['outline']),
        lineWidthUnits: 'pixels',
        getLineWidth: toNumber(visConfig['thickness'], 2),
        getPosition: (row: number) => [
          columns.longitudes[row],
          columns.latitudes[row],
          altitude ? altitude[row] || 0 : 0
        ],
        getRadius: (row: number) => getRadius(row),
        getFillColor: (row: number) => getFill(row),
        getLineColor: (row: number) => getStroke(row),
        updateTriggers: {}
      });
    }
    case 'arc':
    case 'line': {
      const longitudes0 = readPointColumns(dataset, spec.columns['lng0'], spec.columns['lat0']);
      const target = readPointColumns(dataset, spec.columns['lng1'], spec.columns['lat1']);
      if (!longitudes0 || !target) return null;
      for (const row of rowList) {
        bounds.add(longitudes0.longitudes[row], longitudes0.latitudes[row]);
        bounds.add(target.longitudes[row], target.latitudes[row]);
      }
      const getColor = createColorChannel(
        spec,
        dataset,
        rows,
        'color',
        'colorRange',
        baseColor,
        legendEntries
      );
      const targetColor = parseViewerColor(visConfig['targetColor'], baseColor);
      const getWidth = createSizeChannel(
        spec,
        dataset,
        rows,
        'size',
        readRange(visConfig['sizeRange'], [0, 10]),
        toNumber(visConfig['thickness'], 2),
        legendEntries
      );
      const hasColorField = Boolean(spec.channels['color'].field);
      const props = {
        ...common,
        data: rowList,
        widthUnits: 'pixels',
        getSourcePosition: (row: number) => [
          longitudes0.longitudes[row],
          longitudes0.latitudes[row]
        ],
        getTargetPosition: (row: number) => [target.longitudes[row], target.latitudes[row]],
        getWidth: (row: number) => getWidth(row)
      };
      return spec.type === 'arc'
        ? createIndexedLayer(ArcLayer, {
            ...props,
            greatCircle: Boolean(visConfig['greatCircle'] ?? false),
            getSourceColor: (row: number) => getColor(row),
            getTargetColor: (row: number) => (hasColorField ? getColor(row) : targetColor)
          })
        : createIndexedLayer(LineLayer, {...props, getColor: (row: number) => getColor(row)});
    }
    case 's2':
    case 'a5':
    case 'geohash':
    case 'geojson':
    case 'trip': {
      const isCell = isCellLayerType(spec.type);
      const fieldName = isCell ? spec.columns['token'] : spec.columns['geojson'];
      if (!fieldName || !dataset.fieldNames.includes(fieldName)) return null;
      const features = isCell
        ? decodeCellFeatures(spec.type as 's2' | 'a5' | 'geohash', dataset, fieldName, rows)
        : parseFeatures(context, dataset, fieldName, rows);
      for (const feature of features) collectGeometryBounds(feature.geometry, bounds);
      const getFill = createColorChannel(
        spec,
        dataset,
        rows,
        'color',
        'colorRange',
        baseColor,
        legendEntries
      );
      if (spec.type === 'trip')
        return buildTripsLayer(spec, features, getFill, context, result, common);
      const getStroke = createColorChannel(
        spec,
        dataset,
        rows,
        'strokeColor',
        'strokeColorRange',
        parseViewerColor(visConfig['strokeColor'], baseColor),
        []
      );
      const getWidth = createSizeChannel(
        spec,
        dataset,
        rows,
        'size',
        readRange(visConfig['sizeRange'], [0, 10]),
        toNumber(visConfig['thickness'], 0.5),
        legendEntries
      );
      const getRadius = createSizeChannel(
        spec,
        dataset,
        rows,
        'radius',
        readRange(visConfig['radiusRange'], [0, 50]),
        toNumber(visConfig['radius'], 10),
        legendEntries
      );
      const getHeight = createSizeChannel(
        spec,
        dataset,
        rows,
        'height',
        readRange(visConfig['heightRange'], [0, 500]),
        500,
        legendEntries
      );
      const fixedRadius = Boolean(visConfig['fixedRadius']);
      return createIndexedLayer(GeoJsonLayer, {
        ...common,
        data: features,
        filled: visConfig['filled'] !== false,
        stroked: visConfig['stroked'] !== false,
        extruded: Boolean(visConfig['enable3d']),
        wireframe: Boolean(visConfig['wireframe']),
        elevationScale: toNumber(visConfig['elevationScale'], 5),
        lineWidthUnits: 'pixels',
        pointRadiusUnits: fixedRadius ? 'meters' : 'pixels',
        getFillColor: (feature: any) => getFill(feature.properties.__row),
        getLineColor: (feature: any) => getStroke(feature.properties.__row),
        getLineWidth: (feature: any) => getWidth(feature.properties.__row),
        getPointRadius: (feature: any) => getRadius(feature.properties.__row),
        getElevation: (feature: any) => getHeight(feature.properties.__row)
      });
    }
    case 'hexagonId':
    case 'h3': {
      const fieldName = spec.columns['hex_id'];
      const hexagons = fieldName ? dataset.readStrings(fieldName) : null;
      if (!hexagons) return null;
      const getFill = createColorChannel(
        spec,
        dataset,
        rows,
        'color',
        'colorRange',
        baseColor,
        legendEntries
      );
      const getHeight = createSizeChannel(
        spec,
        dataset,
        rows,
        'height',
        readRange(visConfig['heightRange'], [0, 500]),
        500,
        legendEntries
      );
      const getStroke = createColorChannel(
        spec,
        dataset,
        rows,
        'strokeColor',
        'strokeColorRange',
        parseViewerColor(visConfig['strokeColor'], baseColor),
        []
      );
      return createIndexedLayer(H3HexagonLayer, {
        ...common,
        data: rowList.filter(row => hexagons[row]),
        extruded: Boolean(visConfig['enable3d']),
        coverage: toNumber(visConfig['coverage'], 1),
        elevationScale: toNumber(visConfig['elevationScale'], 5),
        stroked: Boolean(visConfig['outline']),
        lineWidthUnits: 'pixels',
        getHexagon: (row: number) => hexagons[row]!,
        getFillColor: (row: number) => getFill(row),
        getLineColor: (row: number) => getStroke(row),
        getElevation: (row: number) => getHeight(row),
        highPrecision: 'auto'
      });
    }
    case 'icon': {
      const columns = readPointColumns(dataset, spec.columns['lng'], spec.columns['lat']);
      const atlas = context.iconAtlas;
      if (!columns || !atlas) return null;
      for (const row of rowList) bounds.add(columns.longitudes[row], columns.latitudes[row]);
      const iconNames = spec.columns['icon'] ? dataset.readStrings(spec.columns['icon']!) : null;
      const getFill = createColorChannel(
        spec,
        dataset,
        rows,
        'color',
        'colorRange',
        baseColor,
        legendEntries
      );
      const getSize = createSizeChannel(
        spec,
        dataset,
        rows,
        'radius',
        readRange(visConfig['radiusRange'], [1, 4]),
        1,
        legendEntries
      );
      return createIndexedLayer(IconLayer, {
        ...common,
        data: rowList,
        iconAtlas: atlas.canvas,
        iconMapping: atlas.mapping,
        sizeUnits: 'pixels',
        sizeScale: clamp(toNumber(visConfig['radius'], 24) / 2, 12, 48),
        getIcon: (row: number) => {
          const name = iconNames?.[row];
          return name && atlas.hasIcon(name) ? name : '__fallback_pin';
        },
        getPosition: (row: number) => [columns.longitudes[row], columns.latitudes[row]],
        getSize: (row: number) => getSize(row),
        getColor: (row: number) => getFill(row),
        updateTriggers: {getIcon: [Object.keys(atlas.mapping).length]}
      });
    }
    case 'cluster': {
      const columns = readPointColumns(dataset, spec.columns['lng'], spec.columns['lat']);
      if (!columns || !context.view) return null;
      for (const row of rowList) bounds.add(columns.longitudes[row], columns.latitudes[row]);
      result.isZoomDependent = true;
      const colorField = spec.channels['color'].field;
      const colors = readColorRangeColors(visConfig['colorRange']);
      legendEntries.push({
        kind: 'ramp',
        title: colorField?.name ?? 'Point count',
        colors,
        domain: []
      });
      return buildClusterLayer({
        id: spec.id,
        opacity,
        rows,
        longitudes: columns.longitudes,
        latitudes: columns.latitudes,
        colorValues: colorField ? dataset.readNumbers(colorField.name) : null,
        colorAggregation: String(visConfig['colorAggregation'] ?? 'count'),
        visConfig,
        scaleType: spec.channels['color'].scale,
        colors,
        view: context.view
      }).layer;
    }
    case 'heatmap': {
      const columns = readPointColumns(dataset, spec.columns['lng'], spec.columns['lat']);
      if (!columns) return null;
      for (const row of rowList) bounds.add(columns.longitudes[row], columns.latitudes[row]);
      result.needsWebGL = true;
      const colors = readColorRangeColors(visConfig['colorRange']);
      const getWeight = createSizeChannel(spec, dataset, rows, 'weight', [0, 1], 1, []);
      legendEntries.push({kind: 'ramp', title: 'Density', colors, domain: [0, 1]});
      return createIndexedLayer(HeatmapLayer, {
        ...common,
        data: rowList,
        radiusPixels: toNumber(visConfig['radius'], 20),
        colorRange: colors,
        intensity: toNumber(visConfig['intensity'], 1),
        threshold: 0.05,
        getPosition: (row: number) => [columns.longitudes[row], columns.latitudes[row]],
        getWeight: (row: number) => getWeight(row)
      });
    }
    case 'grid':
    case 'hexagon': {
      const columns = readPointColumns(dataset, spec.columns['lng'], spec.columns['lat']);
      if (!columns) return null;
      for (const row of rowList) bounds.add(columns.longitudes[row], columns.latitudes[row]);
      result.needsWebGL = true;
      const colors = readColorRangeColors(visConfig['colorRange']);
      const colorField = spec.channels['color'].field;
      const heightField = spec.channels['size'].field ?? spec.channels['height'].field;
      const colorValues = colorField ? dataset.readNumbers(colorField.name) : null;
      const heightValues = heightField ? dataset.readNumbers(heightField.name) : null;
      const percentile = readRange(visConfig['percentile'], [0, 100]);
      const elevationPercentile = readRange(visConfig['elevationPercentile'], [0, 100]);
      legendEntries.push({kind: 'ramp', title: colorField?.name ?? 'Count', colors, domain: []});
      const size = toNumber(visConfig['worldUnitSize'], 1) * 1000;
      const props = {
        ...common,
        data: rowList,
        extruded: Boolean(visConfig['enable3d']),
        coverage: toNumber(visConfig['coverage'], 1),
        elevationScale: toNumber(visConfig['elevationScale'], 5),
        colorRange: colors,
        gpuAggregation: false,
        upperPercentile: percentile[1],
        lowerPercentile: percentile[0],
        elevationUpperPercentile: elevationPercentile[1],
        elevationLowerPercentile: elevationPercentile[0],
        colorAggregation:
          AGGREGATION_NAMES[String(visConfig['colorAggregation'] ?? 'count')] ?? 'COUNT',
        elevationAggregation:
          AGGREGATION_NAMES[String(visConfig['sizeAggregation'] ?? 'count')] ?? 'COUNT',
        getPosition: (row: number) => [columns.longitudes[row], columns.latitudes[row]],
        getColorWeight: (row: number) => (colorValues ? colorValues[row] : 1),
        getElevationWeight: (row: number) => (heightValues ? heightValues[row] : 1)
      };
      return spec.type === 'grid'
        ? createIndexedLayer(GridLayer, {...props, cellSize: size})
        : createIndexedLayer(HexagonLayer, {...props, radius: size});
    }
    default:
      return null;
  }
}

function buildTripsLayer(
  spec: ViewerLayerSpec,
  features: any[],
  getColor: (row: number) => ColorTriple,
  context: ViewerBuildContext,
  result: ViewerBuildResult,
  common: {id: string; pickable: boolean; opacity: number}
): any | null {
  const trips = features.filter(
    feature =>
      feature.geometry.type === 'LineString' && feature.geometry.coordinates[0]?.length >= 4
  );
  if (!trips.length) return null;
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const trip of trips)
    for (const coordinate of trip.geometry.coordinates) {
      minimum = Math.min(minimum, coordinate[3]);
      maximum = Math.max(maximum, coordinate[3]);
    }
  result.tripDomain = result.tripDomain
    ? [Math.min(result.tripDomain[0], minimum), Math.max(result.tripDomain[1], maximum)]
    : [minimum, maximum];
  const isMilliseconds = maximum > 1e11;
  const trailLength = toNumber(spec.visConfig['trailLength'], 180) * (isMilliseconds ? 1000 : 1);
  return createIndexedLayer(TripsLayer, {
    ...common,
    data: trips,
    widthUnits: 'pixels',
    widthMinPixels: toNumber(spec.visConfig['thickness'], 2),
    fadeTrail: spec.visConfig['fadeTrail'] !== false,
    trailLength,
    currentTime: context.currentTime,
    getPath: (trip: any) => trip.geometry.coordinates.map((c: number[]) => [c[0], c[1], c[2] || 0]),
    getTimestamps: (trip: any) => trip.geometry.coordinates.map((c: number[]) => c[3]),
    getColor: (trip: any) => getColor(trip.properties.__row)
  });
}

/** Positions native polygon filters test per layer family (cells and geometry use a center). */
function getFilterPoints(
  spec: ViewerLayerSpec,
  dataset: ViewerDataset,
  context: ViewerBuildContext,
  row: number
): [number, number][] {
  const read = (
    longitudeName?: string | null,
    latitudeName?: string | null
  ): [number, number][] => {
    const columns = readPointColumns(dataset, longitudeName, latitudeName);
    return columns ? [[columns.longitudes[row], columns.latitudes[row]]] : [];
  };
  switch (spec.type) {
    case 'arc':
    case 'line':
      return [
        ...read(spec.columns['lng0'], spec.columns['lat0']),
        ...read(spec.columns['lng1'], spec.columns['lat1'])
      ];
    case 'hexagonId':
    case 'h3': {
      const cell = spec.columns['hex_id']
        ? dataset.readStrings(spec.columns['hex_id']!)?.[row]
        : null;
      try {
        const [latitude, longitude] = cell ? cellToLatLng(cell) : [NaN, NaN];
        return [[longitude, latitude]];
      } catch {
        return [];
      }
    }
    case 's2':
    case 'a5':
    case 'geohash': {
      const field = spec.columns['token'];
      const feature = field
        ? decodeCellFeatures(
            spec.type as 's2' | 'a5' | 'geohash',
            dataset,
            field,
            Uint32Array.of(row)
          )[0]
        : null;
      return feature ? [getGeometryCenter(feature.geometry)] : [];
    }
    case 'geojson':
    case 'trip': {
      const field = spec.columns['geojson'];
      const geometry = field
        ? readGeoJsonGeometry(context.parseJson(dataset.datasetId, field, row))
        : null;
      return geometry?.type ? [getGeometryCenter(geometry)] : [];
    }
    default:
      return read(spec.columns['lng'], spec.columns['lat']);
  }
}

function getGeometryCenter(geometry: any): [number, number] {
  const bounds = new BoundsAccumulator();
  collectGeometryBounds(geometry, bounds);
  return [(bounds.west + bounds.east) / 2, (bounds.south + bounds.north) / 2];
}
