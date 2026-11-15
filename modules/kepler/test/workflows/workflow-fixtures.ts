// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogParameterValue} from '../../src/analysis/catalog-binding-types';
import type {CatalogOutputTable} from '../../src/analysis/catalog-geojson';
import type {
  WorkflowData,
  WorkflowDatasetProfile,
  WorkflowDocument,
  WorkflowHost,
  WorkflowOperation,
  WorkflowOperationRegistry,
  WorkflowPortKind
} from '../../src/workflows/workflow-types';

// Hand-built fixtures that depend on the workflow types only, so every workflow spec can use
// them whatever the state of the other workflow modules.

export function makeFixtureTable(
  id: string,
  rows: readonly {id: string; value: number; geometry?: string}[]
): CatalogOutputTable {
  const hasGeometry = rows.some(row => row.geometry !== undefined);
  return {
    id,
    label: id,
    rowCount: rows.length,
    columns: {
      point_id: rows.map(row => row.id),
      value: rows.map(row => row.value),
      ...(hasGeometry ? {geometry: rows.map(row => row.geometry ?? null)} : {})
    },
    columnTypes: {
      point_id: 'string',
      value: 'float64',
      ...(hasGeometry ? {geometry: 'string' as const} : {})
    },
    ...(hasGeometry ? {geometryColumn: 'geometry'} : {})
  };
}

export function makeFixturePoints(count: number, id = 'points'): WorkflowData {
  return {
    kind: 'point',
    identityColumn: 'point_id',
    table: makeFixtureTable(
      id,
      Array.from({length: count}, (_, index) => ({
        id: `p${index}`,
        value: index * 1.5,
        geometry: JSON.stringify({type: 'Point', coordinates: [-87.6 + index * 0.01, 41.8]})
      }))
    )
  };
}

function normalize(
  operation: Pick<WorkflowOperation, 'parameters' | 'title'>,
  raw: Readonly<Record<string, unknown>>
): Record<string, CatalogParameterValue> {
  const result: Record<string, CatalogParameterValue> = {};
  for (const spec of operation.parameters) {
    const value = raw[spec.name] ?? spec.defaultValue;
    if (
      (spec.kind === 'number' || spec.kind === 'integer') &&
      (typeof value !== 'number' ||
        value < (spec.minimum ?? -Infinity) ||
        value > (spec.maximum ?? Infinity))
    )
      throw new Error(`${operation.title}: ${spec.label} is out of range`);
    result[spec.name] = value as CatalogParameterValue;
  }
  return result;
}

/** Call log of the fixture operations: `${operationId}:${stepId}` per run. */
export type FixtureCalls = string[];

/**
 * Three fake operations with the shape of real ones:
 * - `fixture-source` (no inputs, one `features` output, external revision key from the host),
 * - `fixture-scale` (points in, points out; multiplies `value` by `factor`),
 * - `fixture-count` (any geometry or table in, a one-row table out).
 */
export function makeFixtureRegistry(calls: FixtureCalls = []): WorkflowOperationRegistry {
  const base = {
    origin: 'builtin' as const,
    version: '1',
    capabilityIds: [],
    contributorNames: [],
    declaredBehavior: ['Fixture operation.'],
    limitation: 'Fixture only.'
  };
  const source: WorkflowOperation = {
    ...base,
    id: 'fixture-source',
    title: 'Fixture source',
    category: 'source',
    summary: 'Points from the fixture host.',
    engine: 'host',
    inputs: [],
    parameters: [
      {
        name: 'count',
        label: 'Point count',
        kind: 'integer',
        minimum: 1,
        maximum: 50,
        defaultValue: 4
      }
    ],
    outputs: [
      {
        name: 'data',
        label: 'Data',
        kind: 'features',
        identityColumns: ['point_id'],
        maximumRows: 50
      }
    ],
    normalizeParameters: raw => normalize(source, raw),
    getExternalRevisionKey: host => host.getDatasetRevisionKey(FIXTURE_REQUEST),
    run: async (context, _inputs, parameters) => {
      calls.push(`fixture-source:${context.stepId}`);
      return {
        outputs: {data: makeFixturePoints(parameters['count'] as number)},
        diagnostics: [],
        summary: {rows: parameters['count'] as number},
        engine: 'host',
        dispatchedContributors: []
      };
    }
  };
  const scale: WorkflowOperation = {
    ...base,
    id: 'fixture-scale',
    title: 'Fixture scale',
    category: 'geometry',
    summary: 'Multiplies the value column.',
    engine: 'gpu-or-cpu',
    inputs: [
      {name: 'points', label: 'Points', accepts: ['point'], maximumRows: 10, maximumVertices: 10}
    ],
    parameters: [
      {name: 'factor', label: 'Factor', kind: 'number', minimum: 0, maximum: 100, defaultValue: 2},
      {name: 'fail', label: 'Fail', kind: 'boolean', defaultValue: false}
    ],
    outputs: [
      {
        name: 'scaled',
        label: 'Scaled',
        kind: 'point',
        identityColumns: ['point_id'],
        maximumRows: 10
      }
    ],
    normalizeParameters: raw => normalize(scale, raw),
    run: async (context, inputs, parameters) => {
      calls.push(`fixture-scale:${context.stepId}`);
      if (parameters['fail']) throw new Error('Fixture scale: asked to fail');
      const table = inputs['points'].table;
      const factor = parameters['factor'] as number;
      return {
        outputs: {
          scaled: {
            kind: 'point',
            identityColumn: 'point_id',
            table: {
              ...table,
              id: 'scaled',
              label: 'Scaled',
              columns: {
                ...table.columns,
                value: table.columns['value'].map(value => (value as number) * factor)
              }
            }
          }
        },
        diagnostics: [],
        summary: {factor},
        engine: context.device && context.engine === 'auto' ? 'gpu' : 'cpu-reference',
        dispatchedContributors: [],
        ...(context.device ? {} : {fallbackReason: 'No GPU device supplied'})
      };
    }
  };
  const count: WorkflowOperation = {
    ...base,
    id: 'fixture-count',
    title: 'Fixture count',
    category: 'table',
    summary: 'Counts rows.',
    engine: 'cpu-reference',
    inputs: [
      {
        name: 'rows',
        label: 'Rows',
        accepts: ['point', 'line', 'polygon', 'table'],
        maximumRows: 100
      }
    ],
    parameters: [],
    outputs: [{name: 'count', label: 'Count', kind: 'table', identityColumns: [], maximumRows: 1}],
    normalizeParameters: raw => normalize(count, raw),
    run: async (context, inputs) => {
      calls.push(`fixture-count:${context.stepId}`);
      return {
        outputs: {
          count: {
            kind: 'table',
            table: {
              id: 'count',
              label: 'Count',
              rowCount: 1,
              columns: {rows: [inputs['rows'].table.rowCount]},
              columnTypes: {rows: 'int32'}
            }
          }
        },
        diagnostics: [],
        summary: {rows: inputs['rows'].table.rowCount},
        engine: 'cpu-reference',
        dispatchedContributors: []
      };
    }
  };
  const operations = [source, scale, count];
  return {
    list: () => operations,
    get: operationId => operations.find(operation => operation.id === operationId)
  };
}

const FIXTURE_REQUEST = {
  datasetId: 'fixture',
  geometry: {mode: 'none' as const},
  population: 'all' as const,
  maximumRows: 100
};

export type FixtureHost = WorkflowHost & {
  /** Bump to simulate a change of the host's data. */
  revision: number;
  published: Array<{stepId: string; output: string; datasetId: string; rowCount: number}>;
  retracted: string[];
};

export function makeFixtureHost(
  datasets: readonly WorkflowDatasetProfile[] = [],
  data: Readonly<Record<string, WorkflowData>> = {}
): FixtureHost {
  const host: FixtureHost = {
    revision: 1,
    published: [],
    retracted: [],
    getDevice: () => null,
    listDatasets: () => datasets,
    getDatasetRevisionKey: request => `${request.datasetId}@${host.revision}`,
    readDataset: async request => {
      const found = data[request.datasetId];
      if (!found) throw new Error(`Dataset ${request.datasetId} not found`);
      return found;
    },
    publishOutput: async request => {
      const datasetId = request.previousDatasetId ?? `published-${host.published.length + 1}`;
      host.published.push({
        stepId: request.stepId,
        output: request.output,
        datasetId,
        rowCount: request.data.table.rowCount
      });
      return {datasetId, label: request.label};
    },
    retractOutput: async datasetId => {
      host.retracted.push(datasetId);
    }
  };
  return host;
}

/** source -> scale -> count, with `scale` also displayed. */
export function makeFixtureDocument(): WorkflowDocument {
  return {
    schemaVersion: 1,
    id: 'fixture-workflow',
    title: 'Fixture workflow',
    inputs: [],
    settings: {engine: 'auto', autoRun: false},
    steps: [
      {
        id: 'source',
        operationId: 'fixture-source',
        inputs: {},
        parameters: {},
        position: {x: 0, y: 0}
      },
      {
        id: 'scale',
        operationId: 'fixture-scale',
        inputs: {points: {stepId: 'source', output: 'data'}},
        parameters: {factor: {from: 'static', value: 3}},
        display: ['scaled'],
        position: {x: 280, y: 0}
      },
      {
        id: 'count',
        operationId: 'fixture-count',
        inputs: {rows: {stepId: 'scale', output: 'scaled'}},
        parameters: {},
        position: {x: 560, y: 0}
      }
    ]
  };
}

export const FIXTURE_KINDS: readonly WorkflowPortKind[] = [
  'point',
  'line',
  'polygon',
  'features',
  'table'
];
