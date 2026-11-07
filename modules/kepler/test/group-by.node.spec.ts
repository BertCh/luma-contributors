// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  GPUCommandGraph,
  GraphVectorView,
  createTransientView
} from '@luma-contributors/graph/gpu-core';
import {
  addKeplerGroupBy,
  getKeplerGroupByMemoryEstimate,
  parseKeplerGroupBySettings,
  type KeplerGroupBySettings
} from '../src/aggregation/kepler-group-by';
import {createNullWebGPUDevice} from '../../graph/test/utils/gpu-contributor-test-utils';

const settings: KeplerGroupBySettings = {
  keyColumn: 'category',
  population: 'all',
  numericalPolicy: 'float32',
  aggregations: [{column: 'text', operation: 'count', name: 'text_count'}]
};

test('Group By planning admits distinct keys and explicit Float32/count policies within caps', () => {
  expect(parseKeplerGroupBySettings({...settings, aggregations: []}).aggregations).toEqual([]);
  expect(Object.isFrozen(parseKeplerGroupBySettings(settings).aggregations)).toBe(true);
  for (const override of [
    {population: 'selected'},
    {numericalPolicy: 'float64'},
    {aggregations: [{column: 'value', name: 'category', operation: 'sum'}]},
    {aggregations: [{column: 'value', name: 'out', operation: 'median'}]},
    {
      aggregations: Array.from({length: 17}, (_, index) => ({
        column: 'value',
        name: `out${index}`,
        operation: 'sum'
      }))
    }
  ])
    expect(() => parseKeplerGroupBySettings({...settings, ...override})).toThrow();
  const maximum = getKeplerGroupByMemoryEstimate({
    rowCount: 1_000_000,
    chunkCount: 4,
    dictionaryCount: 4095,
    measureCount: 16
  });
  expect(maximum.capacity).toBe(4096);
  expect(maximum.totalByteLength).toBe(maximum.outputByteLength + maximum.scratchByteLength);
  expect(
    getKeplerGroupByMemoryEstimate({
      rowCount: 0,
      chunkCount: 0,
      dictionaryCount: 0,
      measureCount: 0
    }).capacity
  ).toBe(1);
  for (const override of [
    {rowCount: 67_108_865},
    {dictionaryCount: 1_048_576},
    {measureCount: 17},
    {chunkCount: 0}
  ])
    expect(() =>
      getKeplerGroupByMemoryEstimate({
        rowCount: 2,
        chunkCount: 1,
        dictionaryCount: 3,
        measureCount: 1,
        ...override
      })
    ).toThrow();
});

test('Group By uses borrowed source chunks, dedicated outputs and plans scratch without compilation', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const data = [2, 0, 3].map((length, index) =>
    createTransientView(graph, `source-${index}`, 'uint32', length)
  );
  const key = new GraphVectorView({
    id: 'keys',
    name: 'keys',
    format: 'uint32',
    length: 5,
    valueLength: 5,
    stride: 1,
    byteStride: 4,
    rowByteLength: 4,
    data
  });
  const outputs = {
    keys: createTransientView(graph, 'keys-out', 'uint32', 4),
    keyValidity: createTransientView(graph, 'key-validity-out', 'uint32', 4),
    firstSourceRows: createTransientView(graph, 'first-out', 'uint32', 4),
    status: createTransientView(graph, 'status-out', 'uint32', 4),
    measures: {
      text_count: {
        values: createTransientView(graph, 'count-out', 'uint32', 4),
        validity: createTransientView(graph, 'count-validity-out', 'uint32', 4)
      }
    }
  };
  try {
    const result = addKeplerGroupBy(graph, {
      source: {datasetId: 'source', revision: 1},
      settings,
      sourceFields: ['category', 'text'],
      key,
      dictionaryCount: 3,
      columns: {},
      outputs
    });
    expect(result.outputs).toBe(outputs);
    expect(result.parameterByteLength).toBe(0);
    expect(result.capacity).toBe(4);
    expect(key.data).toEqual(data);
    expect(key.data.map(chunk => chunk.length)).toEqual([2, 0, 3]);
    expect(() => graph.planResources()).not.toThrow();
    result.destroyParameters();
  } finally {
    device.destroy();
  }
});
