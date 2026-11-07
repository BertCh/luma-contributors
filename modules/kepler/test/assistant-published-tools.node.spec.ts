// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  routeAssistantOperation,
  type AssistantContext
} from '../../../examples/deck/kepler-compute/src/assistant-operation-router';

const context: AssistantContext = {
  datasets: {
    'dataset-1': {
      label: 'Taxi trips',
      resident: true,
      rowCount: 100,
      population: 'complete',
      fields: {pickupLat: {type: 'real'}, pickupLng: {type: 'real'}, fare: {type: 'real'}}
    }
  },
  layers: {layer1: {datasetId: 'dataset-1', type: 'point'}},
  limits: {maximumExportRows: 500, maximumLoadBytes: 1000, maximumGroupByAggregations: 4},
  memory: {residentBytes: 0, budgetBytes: 5000},
  supportedLayerTypes: ['point', 'hexagon'],
  supportedAnalyses: [],
  supportedExportFormats: ['csv'],
  supportedViewCount: 1
};

function route(operation: unknown, overrides: Partial<AssistantContext> = {}) {
  return routeAssistantOperation({approved: true, operation}, {...context, ...overrides});
}

describe('published @kepler.gl/ai-assistant tool call aliases', () => {
  test('addLayer by dataset name routes to the native addLayer action', () => {
    const result = route({
      toolName: 'addLayer',
      args: {
        datasetName: 'Taxi trips',
        layerType: 'point',
        latitudeColumn: 'pickupLat',
        longitudeColumn: 'pickupLng',
        layerName: 'Pickups'
      }
    });
    expect(result).toEqual({
      accepted: true,
      route: {
        kind: 'native-action',
        actionName: 'addLayer',
        arguments: [
          {
            type: 'point',
            dataId: 'dataset-1',
            columns: {lat: 'pickupLat', lng: 'pickupLng'},
            label: 'Pickups'
          }
        ]
      }
    });
  });

  test('addLayer flat shape works and unknown names, columns and layer types are rejected', () => {
    expect(
      route({type: 'addLayer', datasetName: 'Taxi trips', layerType: 'hexagon'}).accepted
    ).toBe(true);
    expect(route({type: 'addLayer', datasetName: 'Nope', layerType: 'point'})).toMatchObject({
      accepted: false,
      code: 'unknown-target'
    });
    expect(
      route({
        toolName: 'addLayer',
        args: {datasetName: 'Taxi trips', layerType: 'point', latitudeColumn: 'missing'}
      })
    ).toMatchObject({accepted: false, code: 'unknown-target'});
    expect(
      route({toolName: 'addLayer', args: {datasetName: 'Taxi trips', layerType: 'h3'}})
    ).toMatchObject({accepted: false, code: 'unsupported-option'});
  });

  test('addLayer colorBy maps to a color channel on the native addLayer action', () => {
    expect(
      route({
        toolName: 'addLayer',
        args: {datasetName: 'Taxi trips', layerType: 'point', colorBy: 'fare'}
      })
    ).toMatchObject({
      accepted: true,
      route: {arguments: [{color: {colorField: {name: 'fare', type: 'real'}}}]}
    });
  });

  test('host-defined addLayer still routes unchanged', () => {
    expect(route({type: 'addLayer', datasetId: 'dataset-1', layerType: 'point'})).toMatchObject({
      accepted: true,
      route: {actionName: 'addLayer'}
    });
  });

  test('updateLayerColor maps to a colorRange layerVisConfigChange', () => {
    const result = route({
      toolName: 'updateLayerColor',
      args: {layerId: 'layer1', numberOfColors: 2, customColors: ['#ff0000', '#00ff00']}
    });
    expect(result).toMatchObject({
      accepted: true,
      route: {
        actionName: 'layerVisConfigChange',
        arguments: ['layer1', {colorRange: {type: 'custom', colors: ['#ff0000', '#00ff00']}}]
      }
    });
    expect(
      route({
        toolName: 'updateLayerColor',
        args: {layerId: 'layer1', numberOfColors: 3, customColors: ['#ff0000']}
      })
    ).toMatchObject({accepted: false, code: 'malformed'});
    expect(
      route({
        toolName: 'updateLayerColor',
        args: {layerId: 'missing', numberOfColors: 1, customColors: ['#ff0000']}
      })
    ).toMatchObject({accepted: false, code: 'unknown-target'});
  });

  test('basemap maps to mapStyleChange and validates the published style enum', () => {
    expect(route({toolName: 'basemap', args: {styleType: 'dark-matter'}})).toMatchObject({
      accepted: true,
      route: {actionName: 'mapStyleChange', arguments: ['dark-matter']}
    });
    expect(route({toolName: 'basemap', args: {styleType: 'neon'}})).toMatchObject({
      accepted: false,
      code: 'malformed'
    });
  });

  test('loadData url routes to source-workspace under a worst-case size assumption', () => {
    expect(route({toolName: 'loadData', args: {url: 'https://example.com/a.csv'}})).toMatchObject({
      accepted: true,
      route: {capability: 'source-workspace', request: {sourceName: 'https://example.com/a.csv'}}
    });
    expect(
      route(
        {toolName: 'loadData', args: {url: 'https://example.com/a.csv'}},
        {memory: {residentBytes: 4500, budgetBytes: 5000}}
      )
    ).toMatchObject({accepted: false, code: 'memory'});
  });

  test('geo tools and saveDataToMap without a url are rejected with guidance', () => {
    for (const toolName of ['bufferTool', 'saveDataToMap'])
      expect(route({toolName, args: {}})).toMatchObject({
        accepted: false,
        code: 'unsupported-operation'
      });
  });

  test('unapproved published calls never route', () => {
    expect(
      routeAssistantOperation(
        {approved: false, operation: {toolName: 'basemap', args: {styleType: 'dark'}}},
        context
      )
    ).toMatchObject({accepted: false, code: 'not-approved'});
  });
});
