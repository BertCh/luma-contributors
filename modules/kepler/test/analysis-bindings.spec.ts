// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import type {Buffer, Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GraphVectorView,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {addKeplerAnalysisRecipe} from '../src/analysis/recipe-bindings';
import type {KeplerAnalysisRecipe} from '../src/recipes/analysis-recipe';
import {importGraphBuffer} from '../../graph/src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../graph/test/utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';

const source = {datasetId: 'observations', revision: 1};
function makeRecipe(
  id: string,
  operation: KeplerAnalysisRecipe['operation']
): KeplerAnalysisRecipe {
  return {
    id,
    schemaVersion: 1,
    algorithmVersion: 1,
    source,
    population: 'all',
    backendPolicy: 'require-gpu',
    operation
  };
}
function makeBuffers(device: Device, graph: GPUCommandGraph) {
  const buffers: Buffer[] = [];
  return {
    buffers,
    input<Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format,
      data: Float32Array | Uint32Array
    ): GraphDataView<Format> {
      const buffer = createInputBuffer(device, data);
      buffers.push(buffer);
      return importGraphBuffer(graph, name, buffer, format, data.length);
    },
    output<Format extends 'float32' | 'uint32'>(name: string, format: Format, length: number) {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
    },
    destroy() {
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

it('saved standardization executes on chunked GPU data and separates full versus selected populations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const storage = makeBuffers(device, graph);
  const chunks = [
    storage.input('first', 'float32', new Float32Array([2, 4])),
    storage.input('second', 'float32', new Float32Array([6, 100]))
  ];
  const values = new GraphVectorView({
    id: 'values',
    name: 'values',
    format: 'float32',
    length: 4,
    valueLength: 4,
    stride: 1,
    byteStride: 4,
    rowByteLength: 4,
    data: chunks
  });
  const validityMask = storage.input('validity', 'uint32', new Uint32Array([1, 1, 1, 0]));
  const mask = storage.input('selection', 'uint32', new Uint32Array([0, 1, 1, 0]));
  const full = storage.output('full', 'float32', 4);
  const selected = storage.output('selected', 'float32', 4);
  const recipe = makeRecipe('full-analysis', {
    kind: 'standardize',
    column: 'value',
    method: 'deviation-from-mean',
    variance: 'sample'
  });
  const bindings = {source, columns: {value: values}, validityMask, mask, selectionRevision: 2};
  addKeplerAnalysisRecipe(graph, recipe, {...bindings, outputs: {values: full.view}});
  addKeplerAnalysisRecipe(
    graph,
    {...recipe, id: 'selected-analysis', population: 'selected', selectionRevision: 2},
    {...bindings, outputs: {values: selected.view}}
  );
  expect(() =>
    addKeplerAnalysisRecipe(graph, {...recipe, source: {...source, revision: 2}}, bindings)
  ).toThrow('stale');
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expectValues(await readFloat32(full.buffer, 4), [-2, 0, 2, NaN]);
    expectValues(await readFloat32(selected.buffer, 4), [NaN, -1, 1, NaN]);
    storage.buffers[3].write(new Uint32Array([1, 1, 0, 0]));
    submitGraph(device, compiled, undefined);
    expectValues(await readFloat32(selected.buffer, 4), [-1, 1, NaN, NaN]);
    expectValues(await readFloat32(full.buffer, 4), [-2, 0, 2, NaN]);
  } finally {
    compiled.destroy();
    storage.destroy();
  }
});

it('saved rate and classification recipes publish numerical GPU results without implicit submission', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const storage = makeBuffers(device, graph);
  const events = storage.input('events', 'float32', new Float32Array([10, 2, 20, 4]));
  const populations = storage.input('populations', 'float32', new Float32Array([100, 20, 100, 20]));
  const values = storage.input('values', 'float32', new Float32Array([1, 2, 10, 11]));
  const weights = {
    offsets: storage.input('offsets', 'uint32', new Uint32Array([0, 1, 2, 3, 4])),
    neighbors: storage.input('neighbors', 'uint32', new Uint32Array([1, 0, 3, 2])),
    weights: storage.input('weights', 'float32', new Float32Array([1, 1, 1, 1]))
  };
  const risk = storage.output('risk', 'float32', 4);
  const rates = storage.output('rates', 'float32', 4);
  const breaks = storage.output('breaks', 'float32', 3);
  const classCount = storage.output('class-count', 'uint32', 1);
  const classCounts = storage.output('class-counts', 'uint32', 2);
  const base = {source, columns: {events, populations, values}};
  addKeplerAnalysisRecipe(
    graph,
    makeRecipe('risk-analysis', {
      kind: 'excess-risk',
      events: 'events',
      populations: 'populations'
    }),
    {...base, outputs: {values: risk.view}}
  );
  addKeplerAnalysisRecipe(
    graph,
    makeRecipe('rate-analysis', {
      kind: 'spatial-rate',
      events: 'events',
      populations: 'populations',
      weightsId: 'neighbors',
      weightsRevision: 1,
      scale: 1000,
      islandPolicy: 'undefined'
    }),
    {
      ...base,
      weights: {id: 'neighbors', revision: 1, source, value: weights},
      outputs: {values: rates.view}
    }
  );
  const classification = addKeplerAnalysisRecipe(
    graph,
    makeRecipe('classify-analysis', {
      kind: 'classify',
      column: 'values',
      method: 'natural-breaks',
      classCount: 2,
      naturalBreaksMode: 'exact'
    }),
    {
      ...base,
      outputs: {breaks: breaks.view, classCount: classCount.view, classCounts: classCounts.view}
    }
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const observedRisk = await readFloat32(risk.buffer, 4);
    for (let row = 0; row < 4; row++)
      expect(observedRisk[row]).toBeCloseTo(row < 2 ? 2 / 3 : 4 / 3, 5);
    const observedRates = await readFloat32(rates.buffer, 4);
    for (let row = 0; row < 4; row++)
      expect(observedRates[row]).toBeCloseTo(row < 2 ? 100 : 200, 3);
    expect(Array.from(await readFloat32(breaks.buffer, 3))).toEqual([1, 10, 11]);
    expect(Array.from(await readUint32(classCount.buffer, 1))).toEqual([2]);
    expect(Array.from(await readUint32(classCounts.buffer, 2))).toEqual([2, 2]);
  } finally {
    compiled.destroy();
    classification.destroyParameters();
    storage.destroy();
  }
});

it('hotspot binding retains GPU inference outputs and matches the analytic corner Gi star statistic', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const storage = makeBuffers(device, graph);
  const values = storage.input('values', 'float32', new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]));
  const zScores = storage.output('z-scores', 'float32', 9);
  const result = addKeplerAnalysisRecipe(
    graph,
    makeRecipe('hotspots', {
      kind: 'hot-spots-lattice',
      column: 'value',
      width: 3,
      height: 3,
      criterion: 'queen',
      radius: 1,
      neighborCapacity: 72,
      minimumNeighbors: 1,
      selfWeight: 1,
      multipleTesting: 'none',
      inference: 'analytic-two-sided'
    }),
    {source, columns: {value: values}, outputs: {hotSpots: {zScores: zScores.view}}}
  );
  if (result.operation !== 'hot-spots') throw new Error('Expected hotspot result');
  expect(result.source).toEqual(source);
  expect(result.status.stages.length).toBeGreaterThan(0);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const observed = await readFloat32(zScores.buffer, 9);
    const expected = (12 - 4 * 5) / (Math.sqrt(60 / 9) * Math.sqrt((9 * 4 - 16) / 8));
    expect(observed[0]).toBeCloseTo(expected, 4);
    expect(observed[8]).toBeCloseTo(-expected, 4);
  } finally {
    compiled.destroy();
    result.destroyParameters();
    storage.destroy();
  }
});

function expectValues(actual: readonly number[], expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let row = 0; row < expected.length; row++) {
    if (Number.isNaN(expected[row])) expect(actual[row]).toBeNaN();
    else expect(actual[row]).toBeCloseTo(expected[row], 5);
  }
}
