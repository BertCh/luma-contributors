// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Pure reader for the saved Kepler configuration (KeplerGlSchema.getConfigToSave) inside a bundle. */

type SavedRecord = Record<string, any>;

export type ViewerChannelField = {name: string; type?: string};

export type ViewerLayerSpec = {
  id: string;
  type: string;
  label: string;
  datasetId: string;
  isVisible: boolean;
  color: readonly number[];
  columns: Record<string, string | null>;
  columnMode?: string;
  visConfig: SavedRecord;
  channels: Record<string, {field: ViewerChannelField | null; scale: string | null}>;
};

export type ViewerFilterSpec = {
  id: string;
  datasetIds: string[];
  fieldNames: string[];
  type: string;
  value: unknown;
  /** Layers a polygon filter applies to. */
  layerIds: string[];
  enabled: boolean;
};

export type ViewerConfig = {
  layers: ViewerLayerSpec[];
  filters: ViewerFilterSpec[];
  tooltipFields: Record<string, string[]>;
  mapState: {
    longitude?: number;
    latitude?: number;
    zoom?: number;
    pitch?: number;
    bearing?: number;
    isSplit?: boolean;
    mapSplitMode?: string;
    swipeComparePercentage?: number;
    mapViewMode?: string;
    globe?: {enabled?: boolean} & Record<string, unknown>;
  } | null;
  /** Per split pane: layer id -> visible. Empty when the map is not split. */
  splitMapLayers: Record<string, boolean>[];
  /** Style JSON URL of the selected basemap style, if any (used only when online). */
  basemapStyleUrl: string | null;
  backgroundColor: string;
  splitMapsCount: number;
  animationCurrentTime: number | null;
};

const CHANNEL_NAMES = [
  'color',
  'strokeColor',
  'size',
  'height',
  'radius',
  'coverage',
  'weight',
  'elevationOffset'
] as const;

function isRecord(value: unknown): value is SavedRecord {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

/** Saved layer columns are plain field names (kepler.gl's saved form) or `{value}` objects (its runtime form). */
function readColumnName(column: unknown): string | null {
  if (typeof column === 'string') return column;
  return isRecord(column) && typeof column['value'] === 'string' ? column['value'] : null;
}

/** Convert one saved layer; null when it has no dataset reference. */
export function readViewerLayer(saved: unknown): ViewerLayerSpec | null {
  if (!isRecord(saved) || !isRecord(saved['config'])) return null;
  const config = saved['config'];
  if (typeof config['dataId'] !== 'string') return null;
  const channels: ViewerLayerSpec['channels'] = {};
  const visualChannels = isRecord(saved['visualChannels']) ? saved['visualChannels'] : {};
  for (const name of CHANNEL_NAMES) {
    const field = visualChannels[`${name}Field`];
    const scale = visualChannels[`${name}Scale`];
    channels[name] = {
      field:
        isRecord(field) && typeof field['name'] === 'string' ? (field as ViewerChannelField) : null,
      scale: typeof scale === 'string' ? scale : null
    };
  }
  const columns: Record<string, string | null> = {};
  if (isRecord(config['columns']))
    for (const [role, column] of Object.entries(config['columns']))
      columns[role] = readColumnName(column);
  return {
    id: String(saved['id']),
    type: String(saved['type']),
    label: typeof config['label'] === 'string' ? config['label'] : String(saved['id']),
    datasetId: config['dataId'],
    isVisible: config['isVisible'] !== false,
    color: Array.isArray(config['color']) ? config['color'] : [30, 150, 190],
    columns,
    ...(typeof config['columnMode'] === 'string' ? {columnMode: config['columnMode']} : {}),
    visConfig: isRecord(config['visConfig']) ? config['visConfig'] : {},
    channels
  };
}

export function readViewerConfig(nativeConfig: unknown): ViewerConfig {
  const root =
    isRecord(nativeConfig) && isRecord(nativeConfig['config'])
      ? nativeConfig['config']
      : nativeConfig;
  const visState = isRecord(root) && isRecord(root['visState']) ? root['visState'] : {};
  // Saved order is top-most first; deck draws the last layer on top.
  const layers = (Array.isArray(visState['layers']) ? visState['layers'] : [])
    .map(readViewerLayer)
    .filter((layer): layer is ViewerLayerSpec => layer !== null)
    .reverse();
  const filters: ViewerFilterSpec[] = [];
  for (const saved of Array.isArray(visState['filters']) ? visState['filters'] : []) {
    if (!isRecord(saved)) continue;
    const asList = (value: unknown) =>
      (Array.isArray(value) ? value : [value]).filter(item => typeof item === 'string');
    filters.push({
      id: String(saved['id']),
      datasetIds: asList(saved['dataId']),
      fieldNames: asList(saved['name']),
      type: String(saved['type']),
      value: saved['value'],
      layerIds: asList(saved['layerId']),
      enabled: saved['enabled'] !== false
    });
  }
  const tooltipFields: Record<string, string[]> = {};
  const fieldsToShow = visState['interactionConfig']?.tooltip?.fieldsToShow;
  if (isRecord(fieldsToShow))
    for (const [datasetId, fields] of Object.entries(fieldsToShow))
      if (Array.isArray(fields))
        tooltipFields[datasetId] = fields.map(field => String(field?.name ?? field));
  const mapState = isRecord(root) && isRecord(root['mapState']) ? root['mapState'] : null;
  const mapStyle = isRecord(root) && isRecord(root['mapStyle']) ? root['mapStyle'] : {};
  const backgroundColor =
    typeof mapStyle['backgroundColor'] === 'string'
      ? mapStyle['backgroundColor']
      : Array.isArray(mapStyle['backgroundColor'])
        ? `rgb(${mapStyle['backgroundColor'].slice(0, 3).join(',')})`
        : '#10151d';
  const animation = visState['animationConfig'];
  const splitMapLayers = (Array.isArray(visState['splitMaps']) ? visState['splitMaps'] : []).map(
    (pane: SavedRecord) => {
      const layersByPane: Record<string, boolean> = {};
      const savedLayers = isRecord(pane?.['layers']) ? pane['layers'] : {};
      for (const [layerId, setting] of Object.entries(savedLayers))
        layersByPane[layerId] = isRecord(setting)
          ? setting['isVisible'] !== false
          : Boolean(setting);
      return layersByPane;
    }
  );
  const styleId = typeof mapStyle['styleType'] === 'string' ? mapStyle['styleType'] : null;
  const styleEntry =
    styleId && isRecord(mapStyle['mapStyles']) && isRecord(mapStyle['mapStyles'][styleId])
      ? mapStyle['mapStyles'][styleId]
      : null;
  const basemapStyleUrl =
    styleEntry && typeof styleEntry['url'] === 'string' && /^https?:/.test(styleEntry['url'])
      ? styleEntry['url']
      : null;
  return {
    layers,
    filters,
    tooltipFields,
    mapState: mapState as ViewerConfig['mapState'],
    backgroundColor,
    splitMapLayers,
    basemapStyleUrl,
    splitMapsCount: Array.isArray(visState['splitMaps']) ? visState['splitMaps'].length : 0,
    animationCurrentTime: typeof animation?.currentTime === 'number' ? animation.currentTime : null
  };
}

/** Parse '#rrggbb', 'rgb(r,g,b)' or an [r,g,b] array into a byte triple. */
export function parseViewerColor(
  color: unknown,
  fallback: readonly number[] = [30, 150, 190]
): [number, number, number] {
  if (Array.isArray(color) && color.length >= 3) return [color[0], color[1], color[2]];
  if (typeof color === 'string') {
    const hex = /^#?([0-9a-f]{6})$/i.exec(color.trim());
    if (hex) {
      const value = parseInt(hex[1], 16);
      return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
    }
    const short = /^#([0-9a-f]{3})$/i.exec(color.trim());
    if (short) {
      const [red, green, blue] = short[1].split('').map(digit => parseInt(digit + digit, 16));
      return [red, green, blue];
    }
  }
  return [fallback[0], fallback[1], fallback[2]];
}

/** Colors of a saved Kepler colorRange, reversed when the range says so. */
export function readColorRangeColors(colorRange: unknown): [number, number, number][] {
  const colors =
    isRecord(colorRange) && Array.isArray(colorRange['colors']) ? colorRange['colors'] : [];
  const parsed = colors.map(color => parseViewerColor(color));
  return isRecord(colorRange) && colorRange['reversed'] ? parsed.reverse() : parsed;
}
