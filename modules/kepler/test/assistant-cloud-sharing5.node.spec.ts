// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  routeAssistantOperation,
  type AssistantContext
} from '../../../examples/deck/kepler-compute/src/assistant-operation-router';
import {createAssistantNoticeCapabilities} from '../../../examples/deck/kepler-compute/src/assistant-host-capabilities';
import {
  LocalCloudService,
  createMemoryCloudStore,
  type LocalCloudHost
} from '../../../examples/deck/kepler-compute/src/local-cloud-service';
import {createLocalCloudProvider} from '../../../examples/deck/kepler-compute/src/local-cloud-provider';

const context: AssistantContext = {
  datasets: {
    'dataset-1': {
      label: 'Taxi trips',
      resident: true,
      rowCount: 100,
      population: 'complete',
      fields: {fare: {type: 'real'}, tip: {type: 'real'}, borough: {type: 'string'}}
    },
    streaming: {
      label: 'Live',
      resident: true,
      rowCount: 10,
      population: 'streaming',
      fields: {fare: {type: 'real'}}
    }
  },
  layers: {layer1: {datasetId: 'dataset-1', type: 'point'}},
  limits: {maximumExportRows: 500, maximumLoadBytes: 1000, maximumGroupByAggregations: 4},
  memory: {residentBytes: 0, budgetBytes: 5000},
  supportedLayerTypes: ['point'],
  supportedAnalyses: [],
  supportedExportFormats: ['csv'],
  supportedViewCount: 1
};

function route(operation: unknown, overrides: Partial<AssistantContext> = {}) {
  return routeAssistantOperation({approved: true, operation}, {...context, ...overrides});
}

describe('assistant published tools: color, SQL, chart, boundary', () => {
  test('colorMap on a categorical field yields a custom ordinal range', () => {
    const result = route({
      toolName: 'addLayer',
      args: {
        datasetName: 'Taxi trips',
        layerType: 'point',
        colorBy: 'borough',
        colorMap: {Bronx: '#ff0000'}
      }
    });
    expect(result).toMatchObject({
      accepted: true,
      route: {
        arguments: [
          {color: {colorScale: 'ordinal', colorRange: {colorMap: [['Bronx', '#ff0000']]}}}
        ]
      }
    });
  });

  test('colorMap on a numeric field and unknown color fields are rejected', () => {
    expect(
      route({
        toolName: 'addLayer',
        args: {
          datasetName: 'Taxi trips',
          layerType: 'point',
          colorBy: 'fare',
          colorMap: {a: '#000000'}
        }
      })
    ).toMatchObject({accepted: false, code: 'unsupported-option'});
    expect(route({toolName: 'colorBy', args: {layerId: 'layer1', colorBy: 'nope'}})).toMatchObject({
      accepted: false,
      code: 'unknown-target'
    });
  });

  test('colorBy on an existing layer routes to the visual channel action', () => {
    expect(route({toolName: 'colorBy', args: {layerId: 'layer1', colorBy: 'fare'}})).toMatchObject({
      accepted: true,
      route: {actionName: 'layerVisualChannelConfigChange'}
    });
  });

  test('read-only SQL routes; mutation, multi statement and streaming are refused', () => {
    expect(
      route({
        toolName: 'genericQuery',
        args: {sql: 'SELECT avg(fare) FROM t;', datasetName: 'Taxi trips'}
      })
    ).toMatchObject({accepted: true, route: {capability: 'sql-query'}});
    for (const sql of ['DROP TABLE t', 'SELECT 1; DELETE FROM t', "SELECT * FROM read_csv('x')"])
      expect(route({toolName: 'genericQuery', args: {sql}})).toMatchObject({
        accepted: false,
        code: 'unsupported-option'
      });
    expect(
      route({toolName: 'genericQuery', args: {sql: 'SELECT 1', datasetName: 'Live'}})
    ).toMatchObject({accepted: false, code: 'population'});
  });

  test('SQL that creates a dataset is memory guarded', () => {
    expect(
      route(
        {toolName: 'tableTool', args: {sql: 'SELECT * FROM t'}},
        {memory: {residentBytes: 4500, budgetBytes: 5000}}
      )
    ).toMatchObject({accepted: false, code: 'memory'});
  });

  test('chart tools need complete numeric fields', () => {
    expect(
      route({toolName: 'histogramTool', args: {datasetName: 'Taxi trips', variableName: 'fare'}})
    ).toMatchObject({accepted: true, route: {capability: 'chart-widget'}});
    expect(
      route({toolName: 'histogramTool', args: {datasetName: 'Taxi trips', variableName: 'borough'}})
    ).toMatchObject({accepted: false, code: 'unsupported-option'});
    expect(
      route({
        toolName: 'scatterplotTool',
        args: {datasetName: 'Live', xVariable: 'fare', yVariable: 'fare'}
      })
    ).toMatchObject({accepted: false, code: 'population'});
  });

  test('mapBoundary and saveDataToMap with a url route; geo tools stay rejected', () => {
    expect(route({toolName: 'mapBoundary', args: {}})).toMatchObject({
      accepted: true,
      route: {capability: 'map-boundary'}
    });
    expect(route({toolName: 'saveDataToMap', args: {url: 'https://x/a.csv'}})).toMatchObject({
      accepted: true,
      route: {capability: 'source-workspace'}
    });
    expect(route({toolName: 'bufferTool', args: {}})).toMatchObject({accepted: false});
  });

  test('notice capabilities stage a message', () => {
    const messages: string[] = [];
    createAssistantNoticeCapabilities(message => messages.push(message))['sql-query']({
      query: 'SELECT 1',
      createsDataset: false,
      maximumRows: 5
    });
    expect(messages[0]).toContain('SELECT 1');
  });
});

describe('local cloud provider contract', () => {
  function createInjected() {
    const host: LocalCloudHost = {
      captureBundle: async () => '{"b":1}',
      restoreBundle: async () => ({nativeConfig: {v: 1}, datasets: [{info: {id: 'd'}}]})
    };
    const service = new LocalCloudService(createMemoryCloudStore(), () => host);
    return createLocalCloudProvider({service});
  }
  const mapData = (title: string) =>
    ({
      map: {info: {title, description: 'd'}},
      thumbnail: new Blob([new Uint8Array([1, 2, 3])], {type: 'image/png'})
    }) as never;

  test('mock user, token and logout/login', async () => {
    const provider = createInjected();
    expect(provider.getUserName()).toBe('Local user');
    expect(await provider.getAccessToken()).toBe('local-token');
    await provider.logout();
    expect(await provider.getAccessToken()).toBeNull();
    expect(await provider.getUser()).toBeNull();
    await provider.login();
    expect(await provider.getUser()).toMatchObject({email: 'local@localhost'});
  });

  test('save stores a thumbnail, lists, loads datasets and deletes', async () => {
    const provider = createInjected();
    let changes = 0;
    provider.subscribeToChanges(() => changes++);
    const saved = await provider.uploadMap({mapData: mapData('A'), options: {}} as never);
    expect(provider.getMapUrl({id: saved.id} as never)).toBe(`local://map/${saved.id}`);
    expect(provider.getShareUrl()).toBe(`local://map/${saved.id}`);
    expect(provider.resolveMapUrl(`local://map/${saved.id}`)).toEqual({id: saved.id});
    expect(provider.resolveMapUrl('https://example.com')).toBeNull();
    const [item] = await provider.listMaps();
    expect(item.thumbnail).toMatch(/^data:image\/png;base64,/);
    expect(await provider.getMapThumbnail(item)).toBe(item.thumbnail);
    const loaded = await provider.downloadMap({id: saved.id});
    expect(loaded.map.datasets).toHaveLength(1);
    await provider.deleteMap(saved.id);
    expect(await provider.listMaps()).toHaveLength(0);
    expect(changes).toBe(2);
  });

  test('public publishing is refused', async () => {
    const provider = createInjected();
    expect(provider.hasSharingUrl()).toBe(false);
    await expect(
      provider.uploadMap({mapData: mapData('B'), options: {isPublic: true}} as never)
    ).rejects.toThrow('Public publishing is unavailable');
  });
});
