// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  addGroupBy,
  addLayer,
  createOrUpdateFilter,
  layerVisConfigChange,
  layerVisualChannelConfigChange,
  mapStyleChange,
  runGroupBy,
  setGroupByConfig,
  setMapSplitMode,
  wrapTo
} from '@kepler.gl/actions';
import {
  executeAssistantOperation,
  type AssistantColorChannel,
  type AssistantContext,
  type AssistantExecutors,
  type AssistantRouteResult
} from './assistant-operation-router';

type AssistantFieldState = {name: string; type?: string};
type AssistantDatasetState = {
  id: string;
  label?: string;
  length?: number;
  fields: readonly AssistantFieldState[];
};
type AssistantVisState = {
  datasets: Readonly<Record<string, AssistantDatasetState>>;
  layers: readonly {id: string; type?: string | null; config: {dataId?: string | null}}[];
  groupBys?: readonly {id: string; dataId?: string | null}[];
};
type AssistantStore = {
  getState(): {keplerGl: Record<string, {visState?: unknown} | undefined>};
  dispatch(action: unknown): unknown;
};

export type AssistantHostOptions = {
  store: AssistantStore;
  instanceId: string;
  isResidentDataset(datasetId: string): boolean;
  /** Stored precision for resident numeric fields; defaults to float32 for resident datasets. */
  getFieldPrecision?(datasetId: string, fieldName: string): 'float32' | 'float64' | undefined;
  getPopulation?(datasetId: string): 'complete' | 'partial' | 'streaming';
  getMemory(): {residentBytes: number; budgetBytes: number};
  limits: AssistantContext['limits'];
  supportedLayerTypes: readonly string[];
  supportedAnalyses: readonly string[];
  supportedExportFormats: readonly string[];
  /**
   * Capability routes shared with the manual UI. Keys are router capability ids:
   * 'analysis-panel', 'export-data', 'source-workspace', 'sql-query', 'chart-widget',
   * 'map-boundary'. A missing capability rejects the operation with a reason.
   */
  hostCapabilities: Readonly<
    Record<string, (request: Readonly<Record<string, unknown>>) => void | Promise<void>>
  >;
};

export type AssistantHost = {
  getContext(): AssistantContext;
  /** Run one operation. approved must be true only after the user approved this exact operation. */
  runOperation(approved: boolean, operation: unknown): Promise<AssistantRouteResult>;
};

const NUMERIC_KEPLER_TYPES: Record<string, AssistantContextFieldType> = {
  integer: 'integer',
  real: 'real',
  string: 'string',
  timestamp: 'timestamp',
  date: 'timestamp',
  boolean: 'boolean',
  geojson: 'geojson',
  point: 'point'
};
type AssistantContextFieldType =
  | 'integer'
  | 'real'
  | 'string'
  | 'timestamp'
  | 'boolean'
  | 'geojson'
  | 'point';

/** Connects the pure operation router to the compute store so assistant and manual UI share routes. */
export function createAssistantHost(options: AssistantHostOptions): AssistantHost {
  const readVisState = (): AssistantVisState =>
    options.store.getState().keplerGl[options.instanceId]?.visState as AssistantVisState;
  const dispatch = (action: unknown) =>
    options.store.dispatch(wrapTo(options.instanceId, action as never));

  const getContext = (): AssistantContext => {
    const visState = readVisState();
    const datasets: Record<string, AssistantContext['datasets'][string]> = {};
    for (const dataset of Object.values(visState?.datasets ?? {})) {
      const resident = options.isResidentDataset(dataset.id);
      datasets[dataset.id] = {
        ...(dataset.label ? {label: dataset.label} : {}),
        resident,
        rowCount: dataset.length ?? 0,
        population: options.getPopulation?.(dataset.id) ?? 'complete',
        fields: Object.fromEntries(
          dataset.fields.map(field => {
            const precision = resident
              ? (options.getFieldPrecision?.(dataset.id, field.name) ?? 'float32')
              : undefined;
            return [
              field.name,
              {
                type: NUMERIC_KEPLER_TYPES[field.type ?? ''] ?? 'string',
                ...(precision ? {precision} : {})
              }
            ];
          })
        )
      };
    }
    return {
      datasets,
      layers: Object.fromEntries(
        (visState?.layers ?? []).map(layer => [
          layer.id,
          {datasetId: layer.config.dataId ?? '', type: layer.type ?? ''}
        ])
      ),
      limits: options.limits,
      memory: options.getMemory(),
      supportedLayerTypes: options.supportedLayerTypes,
      supportedAnalyses: options.supportedAnalyses,
      supportedExportFormats: options.supportedExportFormats,
      // The resident workspace owns one Deck device and one map view.
      supportedViewCount: 1
    };
  };

  const executors: AssistantExecutors = {
    executeNativeAction(actionName, args) {
      switch (actionName) {
        case 'addLayer': {
          const [spec] = args as [
            {
              type: string;
              dataId: string;
              columns?: Record<string, string>;
              label?: string;
              color?: AssistantColorChannel;
            }
          ];
          const {colorRange, ...colorChannel} = spec.color ?? {};
          dispatch(
            addLayer({
              type: spec.type,
              config: {
                dataId: spec.dataId,
                label: spec.label ?? spec.type,
                columns: spec.columns ?? {},
                ...colorChannel
              },
              ...(colorRange ? {visConfig: {colorRange}} : {})
            } as never)
          );
          return;
        }
        case 'layerVisualChannelConfigChange': {
          const [layerId, color] = args as [string, AssistantColorChannel];
          const layer = readVisState().layers.find(candidate => candidate.id === layerId);
          if (!layer) throw new Error(`Unknown layer ${layerId}`);
          const {colorRange, ...colorChannel} = color;
          // Same action the layer panel's color field selector dispatches.
          dispatch(
            layerVisualChannelConfigChange(
              layer as never,
              colorChannel as never,
              'color',
              colorRange ? ({colorRange} as never) : undefined
            )
          );
          return;
        }
        case 'setFilter': {
          const [datasetId, field, range] = args as [string, string, [number, number]];
          // Same action the published-source range control dispatches.
          dispatch(
            createOrUpdateFilter(`assistant-${datasetId}-${field}`, datasetId, field, range)
          );
          return;
        }
        case 'layerVisConfigChange': {
          const [layerId, config] = args as [string, Record<string, unknown>];
          const layer = readVisState().layers.find(candidate => candidate.id === layerId);
          if (!layer) throw new Error(`Unknown layer ${layerId}`);
          dispatch(layerVisConfigChange(layer as never, config as never));
          return;
        }
        case 'runGroupBy': {
          const [datasetId, groupByField, aggregations] = args as [
            string,
            string,
            Record<string, string>
          ];
          // Mirrors the manual Group By panel: add an operation, configure it, run it.
          const knownIds = new Set((readVisState().groupBys ?? []).map(operation => operation.id));
          dispatch(addGroupBy(datasetId));
          const created = (readVisState().groupBys ?? []).find(
            operation => !knownIds.has(operation.id)
          );
          if (!created) throw new Error('Group By operation was not created');
          dispatch(setGroupByConfig(created.id, {fieldName: groupByField, aggregations} as never));
          dispatch(runGroupBy(created.id));
          return;
        }
        case 'mapStyleChange': {
          const [styleType] = args as [string];
          dispatch(mapStyleChange(styleType as never));
          return;
        }
        case 'setMapSplitMode': {
          const [mode] = args as [string];
          options.store.dispatch(
            wrapTo(options.instanceId, setMapSplitMode({mapSplitMode: mode as never}) as never)
          );
          return;
        }
        default:
          throw new Error(`No executor for native action ${actionName}`);
      }
    },
    executeHostCapability(capability, request) {
      const handler = options.hostCapabilities[capability];
      if (!handler) throw new Error(`No host capability ${capability}`);
      return handler(request);
    }
  };

  return {
    getContext,
    runOperation: (approved, operation) =>
      executeAssistantOperation({approved, operation}, getContext(), executors)
  };
}
