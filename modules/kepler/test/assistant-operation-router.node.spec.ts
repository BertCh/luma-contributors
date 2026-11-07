// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  executeAssistantOperation,
  routeAssistantOperation,
  type AssistantContext
} from '../../../examples/deck/kepler-compute/src/assistant-operation-router';

const context: AssistantContext = {
  datasets: {
    obs: {
      resident: true,
      rowCount: 1000,
      population: 'complete',
      fields: {
        value: {type: 'real', precision: 'float32'},
        group: {type: 'string'},
        time: {type: 'timestamp'}
      }
    },
    stream: {resident: true, rowCount: 10, population: 'streaming', fields: {value: {type: 'real'}}}
  },
  layers: {layer1: {datasetId: 'obs', type: 'point'}},
  limits: {maximumExportRows: 500, maximumLoadBytes: 1000, maximumGroupByAggregations: 4},
  memory: {residentBytes: 600, budgetBytes: 1000},
  supportedLayerTypes: ['point', 'hexagon'],
  supportedAnalyses: ['hotspot'],
  supportedExportFormats: ['csv', 'arrow'],
  supportedViewCount: 1
};

function route(operation: unknown, approved = true) {
  return routeAssistantOperation({approved, operation}, context);
}
function code(operation: unknown) {
  const result = route(operation);
  return result.accepted ? 'accepted' : result.code;
}

describe('routeAssistantOperation', () => {
  test('requires approval and well-formed operations', () => {
    expect(route({type: 'addLayer'}, false)).toMatchObject({accepted: false, code: 'not-approved'});
    expect(code('x')).toBe('malformed');
    expect(code({type: 'teleport'})).toBe('unknown-operation');
  });

  test('rejects unsupported native operations with guidance', () => {
    const result = route({type: 'extractData'});
    expect(result).toMatchObject({accepted: false, code: 'unsupported-operation'});
    expect(code({type: 'executeSql'})).toBe('malformed');
  });

  test('routes layers, filters and styles to native actions', () => {
    expect(route({type: 'addLayer', datasetId: 'obs', layerType: 'point'})).toMatchObject({
      accepted: true,
      route: {kind: 'native-action', actionName: 'addLayer'}
    });
    expect(code({type: 'addLayer', datasetId: 'obs', layerType: 'arc'})).toBe('unsupported-option');
    expect(code({type: 'addLayer', datasetId: 'nope', layerType: 'point'})).toBe('unknown-target');
    expect(code({type: 'setFilter', datasetId: 'obs', field: 'value', range: [1, 2]})).toBe(
      'accepted'
    );
    expect(code({type: 'setFilter', datasetId: 'obs', field: 'value', range: [3, 2]})).toBe(
      'malformed'
    );
    expect(code({type: 'setFilter', datasetId: 'obs', field: 'group', range: [1, 2]})).toBe(
      'unsupported-option'
    );
    expect(code({type: 'setLayerStyle', layerId: 'missing', property: 'opacity', value: 1})).toBe(
      'unknown-target'
    );
  });

  test('guards memory on load', () => {
    expect(code({type: 'loadData', sourceName: 'a', estimatedRows: 1, estimatedBytes: 300})).toBe(
      'accepted'
    );
    expect(code({type: 'loadData', sourceName: 'a', estimatedRows: 1, estimatedBytes: 500})).toBe(
      'memory'
    );
    expect(code({type: 'loadData', sourceName: 'a', estimatedRows: 1, estimatedBytes: 5000})).toBe(
      'memory'
    );
  });

  test('guards population and precision for aggregation and analysis', () => {
    const base = {datasetId: 'obs', groupByField: 'group', aggregations: {value: 'sum'}};
    expect(code({type: 'runAggregation', ...base})).toBe('accepted');
    expect(code({type: 'runAggregation', ...base, requiredPrecision: 'float64'})).toBe('precision');
    expect(code({type: 'runAggregation', ...base, datasetId: 'stream'})).toBe('population');
    expect(
      code({type: 'runAnalysis', datasetId: 'obs', recipe: 'hotspot', fields: ['value']})
    ).toBe('accepted');
    expect(
      code({type: 'runAnalysis', datasetId: 'obs', recipe: 'kriging', fields: ['value']})
    ).toBe('unsupported-operation');
    expect(
      code({type: 'runAnalysis', datasetId: 'obs', recipe: 'hotspot', fields: ['group']})
    ).toBe('unsupported-option');
  });

  test('enforces export contracts', () => {
    expect(code({type: 'exportData', datasetId: 'obs', format: 'csv', scope: 'selected'})).toBe(
      'accepted'
    );
    expect(code({type: 'exportData', datasetId: 'obs', format: 'pdf', scope: 'selected'})).toBe(
      'export'
    );
    expect(code({type: 'exportData', datasetId: 'obs', format: 'csv', scope: 'all'})).toBe(
      'export'
    );
    expect(
      code({
        type: 'exportData',
        datasetId: 'obs',
        format: 'csv',
        scope: 'all',
        confirmMaterialization: true
      })
    ).toBe('export');
    expect(code({type: 'exportData', datasetId: 'stream', format: 'csv', scope: 'selected'})).toBe(
      'population'
    );
  });

  test('keeps the single-view contract', () => {
    expect(code({type: 'setMapSplit', mode: 'DUAL_MAP'})).toBe('unsupported-option');
    expect(code({type: 'setMapSplit', mode: 'SINGLE_MAP'})).toBe('accepted');
  });
});

describe('executeAssistantOperation', () => {
  test('executes only accepted routes', async () => {
    const calls: string[] = [];
    const executors = {
      executeNativeAction: (name: string) => void calls.push(`native:${name}`),
      executeHostCapability: (name: string) => void calls.push(`host:${name}`)
    };
    await executeAssistantOperation(
      {
        approved: true,
        operation: {type: 'exportData', datasetId: 'obs', format: 'csv', scope: 'selected'}
      },
      context,
      executors
    );
    await executeAssistantOperation(
      {approved: true, operation: {type: 'extractData'}},
      context,
      executors
    );
    await executeAssistantOperation(
      {approved: false, operation: {type: 'addLayer'}},
      context,
      executors
    );
    expect(calls).toEqual(['host:export-data']);
  });

  test('reports executor failures as rejections', async () => {
    const result = await executeAssistantOperation(
      {approved: true, operation: {type: 'setMapSplit', mode: 'SINGLE_MAP'}},
      context,
      {
        executeNativeAction: () => {
          throw new Error('boom');
        },
        executeHostCapability: () => {}
      }
    );
    expect(result).toMatchObject({accepted: false});
  });
});
