// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Pure router for approved AI-assistant structured operations. It maps each operation onto the same
 * native action or host capability route that the manual UI uses, after guards for approval,
 * target existence, population, memory, precision and export contracts. It never calls a model,
 * network or provider; execution happens through injected executors (see executeAssistantOperation).
 * The operation vocabulary is host-defined; calls shaped like the tools published by
 * @kepler.gl/ai-assistant@3.3.0-alpha.8 are normalized first (assistant-published-tools.ts).
 */

import {normalizePublishedToolCall} from './assistant-published-tools';

export type AssistantFieldContext = {
  type: 'integer' | 'real' | 'string' | 'timestamp' | 'boolean' | 'geojson' | 'point';
  /** Stored precision of resident numeric values. */
  precision?: 'float32' | 'float64';
};

export type AssistantDatasetContext = {
  /** Display name; published addLayer identifies datasets by name. */
  label?: string;
  resident: boolean;
  rowCount: number;
  population: 'complete' | 'partial' | 'streaming';
  fields: Readonly<Record<string, AssistantFieldContext>>;
};

export type AssistantContext = {
  datasets: Readonly<Record<string, AssistantDatasetContext>>;
  layers: Readonly<Record<string, {datasetId: string; type: string}>>;
  limits: {
    maximumExportRows: number;
    maximumLoadBytes: number;
    maximumGroupByAggregations: number;
  };
  memory: {residentBytes: number; budgetBytes: number};
  /** Layer types the resident renderer can draw. */
  supportedLayerTypes: readonly string[];
  /** Analysis recipes exposed by the Analysis panel. */
  supportedAnalyses: readonly string[];
  supportedExportFormats: readonly string[];
  /** Resident workspaces own one Deck device and one map view. */
  supportedViewCount: number;
};

export type AssistantOperation =
  | {type: 'loadData'; sourceName: string; estimatedRows: number; estimatedBytes: number}
  | {
      type: 'addLayer';
      datasetId: string;
      layerType: string;
      columns?: Record<string, string>;
      layerName?: string;
      /** Published colorBy: field that drives the color channel. */
      colorField?: string;
      /** Published colorMap: category value -> hex color; ordinal fields only. */
      colorMap?: Record<string, string>;
    }
  | {type: 'setLayerColorField'; layerId: string; field: string; colorMap?: Record<string, string>}
  | {
      type: 'runSql';
      query: string;
      datasetId?: string;
      /** True when the result becomes a new resident dataset (memory guarded like loadData). */
      createsDataset: boolean;
      maximumRows?: number;
    }
  | {type: 'showChart'; chartType: string; datasetId: string; fields: string[]}
  | {type: 'readMapBoundary'; datasetId?: string}
  | {type: 'setFilter'; datasetId: string; field: string; range: [number, number]}
  | {type: 'setLayerStyle'; layerId: string; property: string; value: unknown}
  | {
      type: 'runAggregation';
      datasetId: string;
      groupByField: string;
      aggregations: Record<string, 'count' | 'sum' | 'average' | 'minimum' | 'maximum'>;
      requiredPrecision?: 'float32' | 'float64';
    }
  | {
      type: 'runAnalysis';
      datasetId: string;
      recipe: string;
      fields: string[];
      requiredPrecision?: 'float32' | 'float64';
    }
  | {
      type: 'exportData';
      datasetId: string;
      format: string;
      scope: 'selected' | 'all';
      /** Required for resident data: full export is an explicit materialization. */
      confirmMaterialization?: boolean;
    }
  | {type: 'setMapSplit'; mode: string}
  | {type: 'setBasemap'; styleType: string};

export type AssistantRequest = {
  /** True only after the user approved this exact operation. */
  approved: boolean;
  operation: unknown;
};

export type AssistantRejectionCode =
  | 'not-approved'
  | 'malformed'
  | 'unknown-operation'
  | 'unsupported-operation'
  | 'unknown-target'
  | 'unsupported-option'
  | 'population'
  | 'memory'
  | 'precision'
  | 'export';

export type AssistantRoute =
  | {kind: 'native-action'; actionName: string; arguments: readonly unknown[]}
  | {kind: 'host-capability'; capability: string; request: Readonly<Record<string, unknown>>};

export type AssistantRouteResult =
  | {accepted: true; route: AssistantRoute}
  | {accepted: false; code: AssistantRejectionCode; reason: string};

/** Operations with a documented manual alternative; the assistant gets the same guidance. */
export const ASSISTANT_UNSUPPORTED_OPERATIONS: Readonly<Record<string, string>> = {
  extractData:
    'Native Extract data runs a retained-row CPU filter and is unavailable for resident sources. Use a filter, then export the selected rows.',
  runJoin: 'Join has no assistant route; use Add Data -> SQL or the Join panel manually.',
  runSpatialJoin: 'Spatial Join has no assistant route; use the Spatial Join panel manually.',
  deleteDataset: 'Dataset removal is not available to the assistant.'
};

const NUMERIC_TYPES = new Set(['integer', 'real']);

function reject(code: AssistantRejectionCode, reason: string): AssistantRouteResult {
  return {accepted: false, code, reason};
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export function routeAssistantOperation(
  request: AssistantRequest,
  context: AssistantContext
): AssistantRouteResult {
  if (request.approved !== true)
    return reject('not-approved', 'The operation needs explicit user approval before it can run.');
  const operation = request.operation;
  if (!isRecord(operation))
    return reject('malformed', 'The operation must be an object with a string type.');
  // Published-tool envelopes carry toolName instead of type; non-envelopes still need a string type.
  if (
    typeof operation['type'] !== 'string' &&
    typeof operation['toolName'] !== 'string' &&
    typeof operation['tool'] !== 'string' &&
    typeof operation['name'] !== 'string'
  )
    return reject('malformed', 'The operation must be an object with a string type.');
  const normalized = normalizePublishedToolCall(operation, context);
  if (normalized.kind === 'rejection') return reject(normalized.code, normalized.reason);
  const routed = normalized.kind === 'operation' ? normalized.operation : operation;
  const type = routed['type'];
  if (typeof type !== 'string') return reject('unknown-operation', 'Unknown assistant tool call.');
  // Host-defined executeSql is the read-only SQL route without dataset creation.
  if (type === 'executeSql')
    return routeRunSql(
      {
        type: 'runSql',
        query: routed['query'] as string,
        datasetId: routed['datasetId'] as string | undefined,
        createsDataset: false
      },
      context
    );
  if (type in ASSISTANT_UNSUPPORTED_OPERATIONS)
    return reject('unsupported-operation', ASSISTANT_UNSUPPORTED_OPERATIONS[type]);
  const typed = routed as AssistantOperation;
  switch (typed.type) {
    case 'loadData':
      return routeLoadData(typed, context);
    case 'addLayer':
      return routeAddLayer(typed, context);
    case 'setLayerColorField':
      return routeSetLayerColorField(typed, context);
    case 'runSql':
      return routeRunSql(typed, context);
    case 'showChart':
      return routeShowChart(typed, context);
    case 'readMapBoundary':
      return routeReadMapBoundary(typed, context);
    case 'setFilter':
      return routeSetFilter(typed, context);
    case 'setLayerStyle':
      return routeSetLayerStyle(typed, context);
    case 'runAggregation':
      return routeAggregation(typed, context);
    case 'runAnalysis':
      return routeAnalysis(typed, context);
    case 'exportData':
      return routeExport(typed, context);
    case 'setMapSplit':
      return typed.mode === 'SINGLE_MAP' || context.supportedViewCount >= 2
        ? {
            accepted: true,
            route: {kind: 'native-action', actionName: 'setMapSplitMode', arguments: [typed.mode]}
          }
        : reject(
            'unsupported-option',
            'This resident workspace owns one Deck device and supports one map view; keep the single view.'
          );
    case 'setBasemap':
      return {
        accepted: true,
        route: {kind: 'native-action', actionName: 'mapStyleChange', arguments: [typed.styleType]}
      };
    default:
      return reject('unknown-operation', `Unknown assistant operation: ${type}`);
  }
}

function getDataset(
  datasetId: unknown,
  context: AssistantContext
): AssistantDatasetContext | undefined {
  return typeof datasetId === 'string' ? context.datasets[datasetId] : undefined;
}

function checkComplete(
  dataset: AssistantDatasetContext,
  label: string
): AssistantRouteResult | null {
  return dataset.population === 'complete'
    ? null
    : reject(
        'population',
        `${label} needs a complete population; this dataset is ${dataset.population}.`
      );
}

function checkPrecision(
  dataset: AssistantDatasetContext,
  fields: readonly string[],
  required: 'float32' | 'float64' | undefined
): AssistantRouteResult | null {
  for (const name of fields) {
    const field = dataset.fields[name];
    if (!field) return reject('unknown-target', `Unknown field: ${name}`);
    if (!NUMERIC_TYPES.has(field.type))
      return reject('unsupported-option', `Field ${name} is not numeric.`);
    if (required === 'float64' && field.precision === 'float32')
      return reject(
        'precision',
        `Field ${name} is stored as float32 and cannot satisfy a float64 request.`
      );
  }
  return null;
}

function routeLoadData(
  operation: Extract<AssistantOperation, {type: 'loadData'}>,
  context: AssistantContext
): AssistantRouteResult {
  const bytes = operation.estimatedBytes;
  if (!Number.isFinite(bytes) || bytes < 0 || !Number.isFinite(operation.estimatedRows))
    return reject('malformed', 'loadData needs finite row and byte estimates.');
  if (bytes > context.limits.maximumLoadBytes)
    return reject('memory', 'The source exceeds the per-load byte budget.');
  if (context.memory.residentBytes + bytes > context.memory.budgetBytes)
    return reject('memory', 'Loading this source would exceed the resident memory budget.');
  return {
    accepted: true,
    route: {
      kind: 'host-capability',
      capability: 'source-workspace',
      request: {sourceName: operation.sourceName}
    }
  };
}

function routeAddLayer(
  operation: Extract<AssistantOperation, {type: 'addLayer'}>,
  context: AssistantContext
): AssistantRouteResult {
  const dataset = getDataset(operation.datasetId, context);
  if (!dataset) return reject('unknown-target', `Unknown dataset: ${operation.datasetId}`);
  if (!context.supportedLayerTypes.includes(operation.layerType))
    return reject(
      'unsupported-option',
      `Layer type ${operation.layerType} is not supported by the resident renderer.`
    );
  for (const column of Object.values(operation.columns ?? {}))
    if (!dataset.fields[column]) return reject('unknown-target', `Unknown field: ${column}`);
  const color = operation.colorField
    ? resolveColorChannel(dataset, operation.colorField, operation.colorMap)
    : null;
  if (color && 'accepted' in color) return color;
  if (!operation.colorField && operation.colorMap)
    return reject('malformed', 'colorMap needs a colorBy field.');
  return {
    accepted: true,
    route: {
      kind: 'native-action',
      actionName: 'addLayer',
      arguments: [
        {
          type: operation.layerType,
          dataId: operation.datasetId,
          columns: operation.columns,
          label: operation.layerName,
          ...(color ? {color} : {})
        }
      ]
    }
  };
}

export type AssistantColorChannel = {
  colorField: {name: string; type: string};
  colorScale: 'quantile' | 'ordinal';
  /** Category -> hex pairs applied as a custom ordinal palette. */
  colorRange?: {
    name: string;
    type: 'custom';
    category: 'Custom';
    colors: string[];
    colorMap: [string, string][];
  };
};

const HEX_COLOR = /^#[0-9a-fA-F]{6}$/;

/** Published colorBy/colorMap -> Kepler colorField, colorScale and colorRange. */
function resolveColorChannel(
  dataset: AssistantDatasetContext,
  fieldName: string,
  colorMap: Record<string, string> | undefined
): AssistantColorChannel | AssistantRouteResult {
  const field = dataset.fields[fieldName];
  if (!field) return reject('unknown-target', `Unknown field: ${fieldName}`);
  if (field.type === 'geojson' || field.type === 'point')
    return reject('unsupported-option', `Field ${fieldName} cannot drive a color channel.`);
  const categorical = field.type === 'string' || field.type === 'boolean';
  if (colorMap !== undefined) {
    // Assertion: colorMap is a category palette, so it only applies to categorical fields.
    if (!categorical)
      return reject('unsupported-option', 'colorMap needs a string or boolean colorBy field.');
    const entries = isRecord(colorMap) ? Object.entries(colorMap) : [];
    if (!entries.length || !entries.every(([, color]) => HEX_COLOR.test(String(color))))
      return reject('malformed', 'colorMap must map category values to #rrggbb colors.');
    return {
      colorField: {name: fieldName, type: field.type},
      colorScale: 'ordinal',
      colorRange: {
        name: 'color.customPalette',
        type: 'custom',
        category: 'Custom',
        colors: entries.map(([, color]) => color),
        colorMap: entries.map(([value, color]) => [value, color])
      }
    };
  }
  return {
    colorField: {name: fieldName, type: field.type},
    colorScale: categorical ? 'ordinal' : 'quantile'
  };
}

function routeSetLayerColorField(
  operation: Extract<AssistantOperation, {type: 'setLayerColorField'}>,
  context: AssistantContext
): AssistantRouteResult {
  const layer = context.layers[operation.layerId];
  if (!layer) return reject('unknown-target', `Unknown layer: ${operation.layerId}`);
  const dataset = getDataset(layer.datasetId, context);
  if (!dataset) return reject('unknown-target', `Unknown dataset: ${layer.datasetId}`);
  const color = resolveColorChannel(dataset, operation.field, operation.colorMap);
  if ('accepted' in color) return color;
  return {
    accepted: true,
    route: {
      kind: 'native-action',
      actionName: 'layerVisualChannelConfigChange',
      arguments: [operation.layerId, color]
    }
  };
}

const SQL_FORBIDDEN_KEYWORDS =
  /\b(insert|update|delete|drop|create|alter|attach|detach|copy|install|load|export|import|pragma|call|set|truncate|replace|read_csv|read_parquet|read_json|http_get)\b/i;

function routeRunSql(
  operation: Extract<AssistantOperation, {type: 'runSql'}>,
  context: AssistantContext
): AssistantRouteResult {
  if (typeof operation.query !== 'string' || !operation.query.trim())
    return reject('malformed', 'runSql needs a query string.');
  const text = operation.query
    .replace(/--[^\n]*/g, ' ')
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .trim()
    .replace(/;\s*$/, '');
  // Assertion: only one read-only statement may reach the DuckDB route.
  if (text.includes(';') || !/^(select|with)\b/i.test(text))
    return reject('unsupported-option', 'Assistant SQL must be a single SELECT or WITH statement.');
  if (SQL_FORBIDDEN_KEYWORDS.test(text.replace(/'(?:[^']|'')*'/g, "''")))
    return reject(
      'unsupported-option',
      'Assistant SQL may not modify state or read external files.'
    );
  if (operation.datasetId !== undefined) {
    const dataset = getDataset(operation.datasetId, context);
    if (!dataset) return reject('unknown-target', `Unknown dataset: ${operation.datasetId}`);
    const incomplete = checkComplete(dataset, 'SQL');
    if (incomplete) return incomplete;
  }
  if (operation.createsDataset) {
    // The result row count is unknown, so assume the per-load maximum like loadData.
    const bytes = context.limits.maximumLoadBytes;
    if (context.memory.residentBytes + bytes > context.memory.budgetBytes)
      return reject(
        'memory',
        'Materializing a SQL result could exceed the resident memory budget.'
      );
  }
  const maximumRows = Math.min(
    operation.maximumRows ?? context.limits.maximumExportRows,
    context.limits.maximumExportRows
  );
  return {
    accepted: true,
    route: {
      kind: 'host-capability',
      capability: 'sql-query',
      request: {
        query: text,
        datasetId: operation.datasetId,
        createsDataset: operation.createsDataset,
        maximumRows
      }
    }
  };
}

const CHART_FIELD_COUNTS: Readonly<Record<string, number>> = {
  histogram: 1,
  boxplot: 1,
  scatterplot: 2,
  bubble: 3,
  pcp: 2
};

function routeShowChart(
  operation: Extract<AssistantOperation, {type: 'showChart'}>,
  context: AssistantContext
): AssistantRouteResult {
  const dataset = getDataset(operation.datasetId, context);
  if (!dataset) return reject('unknown-target', `Unknown dataset: ${operation.datasetId}`);
  const minimum = CHART_FIELD_COUNTS[operation.chartType];
  if (minimum === undefined)
    return reject(
      'unsupported-option',
      `Chart type ${String(operation.chartType)} is not supported.`
    );
  const fields = operation.fields ?? [];
  if (fields.length < minimum || fields.length > 8)
    return reject('malformed', `${operation.chartType} needs at least ${minimum} fields.`);
  const incomplete = checkComplete(dataset, 'Chart');
  if (incomplete) return incomplete;
  const numeric = checkPrecision(dataset, fields, undefined);
  if (numeric) return numeric;
  return {
    accepted: true,
    route: {
      kind: 'host-capability',
      capability: 'chart-widget',
      request: {chartType: operation.chartType, datasetId: operation.datasetId, fields}
    }
  };
}

function routeReadMapBoundary(
  operation: Extract<AssistantOperation, {type: 'readMapBoundary'}>,
  context: AssistantContext
): AssistantRouteResult {
  if (operation.datasetId !== undefined && !getDataset(operation.datasetId, context))
    return reject('unknown-target', `Unknown dataset: ${operation.datasetId}`);
  return {
    accepted: true,
    route: {
      kind: 'host-capability',
      capability: 'map-boundary',
      request: {datasetId: operation.datasetId}
    }
  };
}

function routeSetFilter(
  operation: Extract<AssistantOperation, {type: 'setFilter'}>,
  context: AssistantContext
): AssistantRouteResult {
  const dataset = getDataset(operation.datasetId, context);
  if (!dataset) return reject('unknown-target', `Unknown dataset: ${operation.datasetId}`);
  const field = dataset.fields[operation.field];
  if (!field) return reject('unknown-target', `Unknown field: ${operation.field}`);
  if (field.type !== 'integer' && field.type !== 'real' && field.type !== 'timestamp')
    return reject('unsupported-option', 'Assistant filters support numeric and time ranges only.');
  const [low, high] = operation.range ?? [];
  if (!Number.isFinite(low) || !Number.isFinite(high) || low > high)
    return reject('malformed', 'Filter range must be a finite ascending pair.');
  return {
    accepted: true,
    route: {
      kind: 'native-action',
      actionName: 'setFilter',
      arguments: [operation.datasetId, operation.field, [low, high]]
    }
  };
}

function routeSetLayerStyle(
  operation: Extract<AssistantOperation, {type: 'setLayerStyle'}>,
  context: AssistantContext
): AssistantRouteResult {
  if (!context.layers[operation.layerId])
    return reject('unknown-target', `Unknown layer: ${operation.layerId}`);
  if (typeof operation.property !== 'string' || !operation.property)
    return reject('malformed', 'setLayerStyle needs a property name.');
  return {
    accepted: true,
    route: {
      kind: 'native-action',
      actionName: 'layerVisConfigChange',
      arguments: [operation.layerId, {[operation.property]: operation.value}]
    }
  };
}

function routeAggregation(
  operation: Extract<AssistantOperation, {type: 'runAggregation'}>,
  context: AssistantContext
): AssistantRouteResult {
  const dataset = getDataset(operation.datasetId, context);
  if (!dataset) return reject('unknown-target', `Unknown dataset: ${operation.datasetId}`);
  const incomplete = checkComplete(dataset, 'Aggregation');
  if (incomplete) return incomplete;
  if (!dataset.fields[operation.groupByField])
    return reject('unknown-target', `Unknown field: ${operation.groupByField}`);
  const entries = Object.entries(operation.aggregations ?? {});
  if (!entries.length || entries.length > context.limits.maximumGroupByAggregations)
    return reject('unsupported-option', 'Aggregation count is outside the supported range.');
  const numericFields = entries.filter(([, kind]) => kind !== 'count').map(([name]) => name);
  const precision = checkPrecision(dataset, numericFields, operation.requiredPrecision);
  if (precision) return precision;
  return {
    accepted: true,
    route: {
      kind: 'native-action',
      actionName: 'runGroupBy',
      arguments: [operation.datasetId, operation.groupByField, operation.aggregations]
    }
  };
}

function routeAnalysis(
  operation: Extract<AssistantOperation, {type: 'runAnalysis'}>,
  context: AssistantContext
): AssistantRouteResult {
  const dataset = getDataset(operation.datasetId, context);
  if (!dataset) return reject('unknown-target', `Unknown dataset: ${operation.datasetId}`);
  if (!context.supportedAnalyses.includes(operation.recipe))
    return reject(
      'unsupported-operation',
      `Analysis ${operation.recipe} has no resident executor.`
    );
  const incomplete = checkComplete(dataset, 'Analysis');
  if (incomplete) return incomplete;
  const precision = checkPrecision(dataset, operation.fields ?? [], operation.requiredPrecision);
  if (precision) return precision;
  return {
    accepted: true,
    route: {
      kind: 'host-capability',
      capability: 'analysis-panel',
      request: {datasetId: operation.datasetId, recipe: operation.recipe, fields: operation.fields}
    }
  };
}

function routeExport(
  operation: Extract<AssistantOperation, {type: 'exportData'}>,
  context: AssistantContext
): AssistantRouteResult {
  const dataset = getDataset(operation.datasetId, context);
  if (!dataset) return reject('unknown-target', `Unknown dataset: ${operation.datasetId}`);
  if (!context.supportedExportFormats.includes(operation.format))
    return reject('export', `Export format ${operation.format} is not supported.`);
  const incomplete = checkComplete(dataset, 'Export');
  if (incomplete) return incomplete;
  if (operation.scope !== 'selected' && operation.scope !== 'all')
    return reject('malformed', 'Export scope must be selected or all.');
  if (operation.scope === 'all') {
    if (dataset.resident && operation.confirmMaterialization !== true)
      return reject(
        'export',
        'Exporting all rows of a resident dataset materializes it and needs explicit confirmation.'
      );
    if (dataset.rowCount > context.limits.maximumExportRows)
      return reject('export', 'The dataset exceeds the export row budget.');
  }
  return {
    accepted: true,
    route: {
      kind: 'host-capability',
      capability: 'export-data',
      request: {datasetId: operation.datasetId, format: operation.format, scope: operation.scope}
    }
  };
}

export type AssistantExecutors = {
  executeNativeAction(actionName: string, args: readonly unknown[]): void | Promise<void>;
  executeHostCapability(
    capability: string,
    request: Readonly<Record<string, unknown>>
  ): void | Promise<void>;
};

/** Route then execute through injected executors; rejected operations never reach an executor. */
export async function executeAssistantOperation(
  request: AssistantRequest,
  context: AssistantContext,
  executors: AssistantExecutors
): Promise<AssistantRouteResult> {
  const result = routeAssistantOperation(request, context);
  if (!result.accepted) return result;
  try {
    if (result.route.kind === 'native-action')
      await executors.executeNativeAction(result.route.actionName, result.route.arguments);
    else await executors.executeHostCapability(result.route.capability, result.route.request);
  } catch (error) {
    return reject(
      'unsupported-operation',
      `Execution failed: ${error instanceof Error ? error.message : String(error)}`
    );
  }
  return result;
}
