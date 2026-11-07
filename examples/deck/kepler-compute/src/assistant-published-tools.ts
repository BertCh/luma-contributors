// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Normalizes tool calls shaped like the tools published by @kepler.gl/ai-assistant@3.3.0-alpha.8
 * (loadData, addLayer, updateLayerColor, basemap, mapBoundary, saveDataToMap, SQL query tools,
 * echarts and geo tools) into the host-defined operation vocabulary of the assistant router.
 * Pure: no package import, model, network or provider. See ASSISTANT-TOOLS.md for the mapping.
 */

import type {AssistantContext} from './assistant-operation-router';

export type PublishedToolNormalization =
  | {kind: 'passthrough'}
  | {kind: 'operation'; operation: Record<string, unknown>}
  | {
      kind: 'rejection';
      code: 'malformed' | 'unknown-target' | 'unsupported-operation';
      reason: string;
    };

/** Base styles accepted by the published basemap tool (its zod enum). */
export const PUBLISHED_BASEMAP_STYLES: readonly string[] = [
  'no_map',
  'dark-matter',
  'positron',
  'voyager',
  'satellite',
  'dark',
  'light',
  'muted',
  'muted_night'
];

/** Published SQL tools; the value says whether the result becomes a new dataset. */
const PUBLISHED_SQL_TOOLS: Readonly<Record<string, boolean>> = {
  genericQuery: false,
  filterDataset: true,
  tableTool: true,
  mergeTables: true
};
/** Published echarts tools -> resident chart widget types. */
const PUBLISHED_CHART_TOOLS: Readonly<Record<string, string>> = {
  boxplotTool: 'boxplot',
  bubbleChartTool: 'bubble',
  histogramTool: 'histogram',
  pcpTool: 'pcp',
  scatterplotTool: 'scatterplot'
};
const PUBLISHED_ANALYSIS_TOOLS = [
  'classifyTool',
  'dataClassify',
  'weightsTool',
  'regressionTool',
  'lisaTool',
  'gridTool',
  'bufferTool',
  'centroidTool',
  'dissolveTool'
];

const UNSUPPORTED_PUBLISHED_REASONS: Readonly<Record<string, string>> = {
  saveDataToMap:
    'saveDataToMap saves geo-tool results as datasets; geo tools have no assistant route in the resident workspace, so only a url argument can be staged.'
};

function readStringList(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  return Array.isArray(value) ? value.filter(item => typeof item === 'string') : [];
}

/** Published colorMap arrives as {value: color} or [{value, color}] pairs. */
function readColorMap(value: unknown): Record<string, string> | undefined {
  if (isRecord(value)) return value as Record<string, string>;
  if (Array.isArray(value)) {
    const entries = value.filter(isRecord).map(entry => [String(entry['value']), entry['color']]);
    return Object.fromEntries(entries);
  }
  return undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function rejectOperation(
  code: 'malformed' | 'unknown-target' | 'unsupported-operation',
  reason: string
): PublishedToolNormalization {
  return {kind: 'rejection', code, reason};
}

/** Published addLayer identifies datasets by display name, not id; ids are accepted as a fallback. */
export function resolveDatasetIdentifier(
  datasetName: string,
  context: AssistantContext
): string | undefined {
  const byLabel = Object.entries(context.datasets).find(
    ([, dataset]) => dataset.label === datasetName
  );
  if (byLabel) return byLabel[0];
  return datasetName in context.datasets ? datasetName : undefined;
}

/** Accepts {toolName|tool|name, args|arguments|input|parameters} envelopes and flat {type: name, ...args}. */
function readToolCall(
  operation: Record<string, unknown>
): {toolName: string; args: Record<string, unknown>} | null {
  const toolName = [operation['toolName'], operation['tool'], operation['name']].find(
    value => typeof value === 'string'
  ) as string | undefined;
  if (toolName) {
    const args = [
      operation['args'],
      operation['arguments'],
      operation['input'],
      operation['parameters']
    ].find(isRecord);
    return {toolName, args: args ?? {}};
  }
  const flatName = operation['type'];
  if (typeof flatName !== 'string') return null;
  const {type: _type, ...args} = operation;
  return {toolName: flatName, args};
}

export function normalizePublishedToolCall(
  operation: Record<string, unknown>,
  context: AssistantContext
): PublishedToolNormalization {
  const call = readToolCall(operation);
  if (!call) return {kind: 'passthrough'};
  const {toolName, args} = call;
  const isEnvelope = operation['type'] === undefined;
  switch (toolName) {
    case 'loadData': {
      // Host-defined loadData carries sourceName and estimates; published carries only a url.
      if (typeof args['url'] !== 'string')
        return isEnvelope
          ? rejectOperation('malformed', 'loadData needs a url.')
          : {kind: 'passthrough'};
      // The published tool has no size estimate: assume the per-load maximum so memory guards stay conservative.
      return {
        kind: 'operation',
        operation: {
          type: 'loadData',
          sourceName: args['url'],
          estimatedRows: 0,
          estimatedBytes: context.limits.maximumLoadBytes
        }
      };
    }
    case 'addLayer': {
      if (typeof args['datasetName'] !== 'string')
        return isEnvelope
          ? rejectOperation('malformed', 'addLayer needs a datasetName.')
          : {kind: 'passthrough'};
      const datasetId = resolveDatasetIdentifier(args['datasetName'], context);
      if (!datasetId)
        return rejectOperation('unknown-target', `Unknown dataset: ${args['datasetName']}`);
      if (typeof args['layerType'] !== 'string')
        return rejectOperation('malformed', 'addLayer needs a layerType.');
      const colorField = args['colorBy'] ?? args['colorField'];
      if (colorField !== undefined && typeof colorField !== 'string')
        return rejectOperation('malformed', 'colorBy must be a field name.');
      const colorMap = args['colorMap'] === undefined ? undefined : readColorMap(args['colorMap']);
      if (args['colorMap'] !== undefined && !colorMap)
        return rejectOperation('malformed', 'colorMap must map values to colors.');
      const columns: Record<string, string> = {};
      if (typeof args['latitudeColumn'] === 'string') columns['lat'] = args['latitudeColumn'];
      if (typeof args['longitudeColumn'] === 'string') columns['lng'] = args['longitudeColumn'];
      // The published description tells the model to use the _geojson column for geojson layers.
      if (args['layerType'] === 'geojson' && context.datasets[datasetId]?.fields['_geojson'])
        columns['geojson'] = '_geojson';
      return {
        kind: 'operation',
        operation: {
          type: 'addLayer',
          datasetId,
          layerType: args['layerType'],
          ...(Object.keys(columns).length ? {columns} : {}),
          ...(typeof args['layerName'] === 'string' ? {layerName: args['layerName']} : {}),
          ...(colorField !== undefined ? {colorField} : {}),
          ...(colorMap ? {colorMap} : {})
        }
      };
    }
    case 'colorBy':
    case 'colorMap': {
      const field = args['colorBy'] ?? args['fieldName'] ?? args['field'];
      if (typeof args['layerId'] !== 'string' || typeof field !== 'string')
        return rejectOperation('malformed', `${toolName} needs layerId and a field name.`);
      const colorMap = readColorMap(args['colorMap']);
      return {
        kind: 'operation',
        operation: {
          type: 'setLayerColorField',
          layerId: args['layerId'],
          field,
          ...(colorMap ? {colorMap} : {})
        }
      };
    }
    case 'mapBoundary': {
      const datasetName = args['datasetName'];
      let datasetId: string | undefined;
      if (typeof datasetName === 'string') {
        datasetId = resolveDatasetIdentifier(datasetName, context);
        if (!datasetId) return rejectOperation('unknown-target', `Unknown dataset: ${datasetName}`);
      }
      return {
        kind: 'operation',
        operation: {type: 'readMapBoundary', ...(datasetId ? {datasetId} : {})}
      };
    }
    case 'updateLayerColor': {
      const colors = args['customColors'];
      if (
        typeof args['layerId'] !== 'string' ||
        !Array.isArray(colors) ||
        !colors.every(color => typeof color === 'string')
      )
        return rejectOperation('malformed', 'updateLayerColor needs layerId and customColors.');
      // Published tool rejects a count mismatch.
      if (args['numberOfColors'] !== colors.length)
        return rejectOperation(
          'malformed',
          `customColors array must contain exactly ${String(args['numberOfColors'])} colors`
        );
      return {
        kind: 'operation',
        operation: {
          type: 'setLayerStyle',
          layerId: args['layerId'],
          property: 'colorRange',
          value: {name: 'color.customPalette', type: 'custom', category: 'Custom', colors}
        }
      };
    }
    case 'basemap': {
      const styleType = args['styleType'];
      if (typeof styleType !== 'string' || !PUBLISHED_BASEMAP_STYLES.includes(styleType))
        return rejectOperation('malformed', `Invalid basemap style: ${String(styleType)}.`);
      return {kind: 'operation', operation: {type: 'setBasemap', styleType}};
    }
    default:
      break;
  }
  if (toolName === 'saveDataToMap' && typeof args['url'] === 'string')
    return normalizePublishedToolCall({type: 'loadData', url: args['url']}, context);
  if (toolName in PUBLISHED_SQL_TOOLS) {
    const query = [args['sql'], args['query'], args['sqlQuery']].find(
      value => typeof value === 'string'
    );
    if (typeof query !== 'string')
      return rejectOperation('malformed', `${toolName} needs a SQL query string.`);
    const datasetName = args['datasetName'];
    let datasetId: string | undefined;
    if (typeof datasetName === 'string') {
      datasetId = resolveDatasetIdentifier(datasetName, context);
      if (!datasetId) return rejectOperation('unknown-target', `Unknown dataset: ${datasetName}`);
    }
    return {
      kind: 'operation',
      operation: {
        type: 'runSql',
        query,
        createsDataset: PUBLISHED_SQL_TOOLS[toolName],
        ...(datasetId ? {datasetId} : {})
      }
    };
  }
  if (toolName in PUBLISHED_CHART_TOOLS) {
    const datasetName = args['datasetName'];
    const datasetId =
      typeof datasetName === 'string' ? resolveDatasetIdentifier(datasetName, context) : undefined;
    if (!datasetId)
      return rejectOperation('unknown-target', `Unknown dataset: ${String(datasetName)}`);
    const fields = [
      ...readStringList(args['variableName']),
      ...readStringList(args['variableNames']),
      ...readStringList(args['xVariable']),
      ...readStringList(args['yVariable']),
      ...readStringList(args['sizeVariable'])
    ];
    return {
      kind: 'operation',
      operation: {type: 'showChart', chartType: PUBLISHED_CHART_TOOLS[toolName], datasetId, fields}
    };
  }
  if (toolName in UNSUPPORTED_PUBLISHED_REASONS)
    return rejectOperation('unsupported-operation', UNSUPPORTED_PUBLISHED_REASONS[toolName]);
  if (PUBLISHED_ANALYSIS_TOOLS.includes(toolName))
    return rejectOperation(
      'unsupported-operation',
      `${toolName} is an OpenAssistant geo or chart tool with no resident executor; use runAnalysis recipes or the Analysis panel.`
    );
  return {kind: 'passthrough'};
}
