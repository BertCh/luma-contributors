// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {createAssistantHost} from '../../../examples/deck/kepler-compute/src/assistant-host';

function createHostFixture() {
  const dispatched: unknown[] = [];
  const capabilities: Array<[string, unknown]> = [];
  const visState = {
    datasets: {
      data: {id: 'data', length: 10, fields: [{name: 'value', type: 'real'}]}
    },
    layers: [],
    groupBys: [] as Array<{id: string}>
  };
  const host = createAssistantHost({
    store: {
      getState: () => ({keplerGl: {compute: {visState}}}),
      dispatch: action => {
        dispatched.push(action);
        if ((action as {payload?: {type?: string}}).payload?.type === 'ADD_GROUP_BY_PLACEHOLDER')
          visState.groupBys.push({id: 'op'});
      }
    },
    instanceId: 'compute',
    isResidentDataset: id => id === 'data',
    getMemory: () => ({residentBytes: 0, budgetBytes: 1e9}),
    limits: {maximumExportRows: 100, maximumLoadBytes: 1e6, maximumGroupByAggregations: 4},
    supportedLayerTypes: ['point'],
    supportedAnalyses: ['quantile'],
    supportedExportFormats: ['csv'],
    hostCapabilities: {
      'analysis-panel': request => {
        capabilities.push(['analysis-panel', request]);
      }
    }
  });
  return {host, dispatched, capabilities};
}

describe('createAssistantHost', () => {
  test('builds a context from native state', () => {
    const {host} = createHostFixture();
    const context = host.getContext();
    expect(context.datasets['data']?.resident).toBe(true);
    expect(context.datasets['data']?.fields['value']).toEqual({type: 'real', precision: 'float32'});
    expect(context.supportedViewCount).toBe(1);
  });

  test('dispatches approved filters and refuses unapproved operations', async () => {
    const {host, dispatched} = createHostFixture();
    const operation = {type: 'setFilter', datasetId: 'data', field: 'value', range: [0, 5]};
    expect((await host.runOperation(false, operation)).accepted).toBe(false);
    expect(dispatched).toHaveLength(0);
    expect((await host.runOperation(true, operation)).accepted).toBe(true);
    expect(dispatched).toHaveLength(1);
  });

  test('routes analysis through the injected capability and rejects float64 on float32', async () => {
    const {host, capabilities} = createHostFixture();
    const analysis = {
      type: 'runAnalysis',
      datasetId: 'data',
      recipe: 'quantile',
      fields: ['value']
    };
    expect((await host.runOperation(true, analysis)).accepted).toBe(true);
    expect(capabilities).toHaveLength(1);
    const strict = await host.runOperation(true, {...analysis, requiredPrecision: 'float64'});
    expect(strict.accepted).toBe(false);
  });

  test('keeps the single-view guard for split maps', async () => {
    const {host, dispatched} = createHostFixture();
    const result = await host.runOperation(true, {type: 'setMapSplit', mode: 'SWIPE_COMPARE'});
    expect(result.accepted).toBe(false);
    expect(dispatched).toHaveLength(0);
  });
});
