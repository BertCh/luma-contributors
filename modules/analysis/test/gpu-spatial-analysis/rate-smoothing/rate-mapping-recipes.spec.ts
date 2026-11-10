// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  addExcessRiskRecipe,
  addSpatialRateRecipe
} from '../../../src/gpu-spatial-analysis/recipes/rate-mapping-recipes';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {RecipeTestFixture} from '../recipes/recipe-harness';

it('GeoDa pooled risk and binary spatial rates handle masks, self edges, islands and replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  // Independent hand arithmetic: total E=22, P=220 -> pooled=.1. The first neighborhood
  // pools 10/100 and 2/20 -> .1; averaging row rates would give the wrong third-row result.
  const events = new Float32Array([10, 2, 10, NaN, 99, 0, 1, 1]);
  const populations = new Float32Array([100, 20, 100, 10, 0, 30, Infinity, 10]);
  const mask = new Uint32Array([1, 1, 1, 1, 1, 0, 1, 0]);
  const buffers: ReturnType<typeof createInputBuffer>[] = [];
  const input = <Format extends 'float32' | 'uint32'>(
    id: string,
    data: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return importGraphBuffer(graph, id, buffer, format, data.length);
  };
  const eventView = input('events', events, 'float32');
  const populationView = input('populations', populations, 'float32');
  const maskView = input('mask', mask, 'uint32');
  const weights = {
    offsets: input('offsets', new Uint32Array([0, 2, 3, 4, 4, 4, 4, 4, 5]), 'uint32'),
    neighbors: input('neighbors', new Uint32Array([0, 1, 0, 7, 2]), 'uint32'),
    weights: input('weights', new Float32Array([99, 0.25, 7, -3, 100]), 'float32')
  };
  const riskBuffer = createOutputBuffer(device, events.length);
  const spatialBuffer = createOutputBuffer(device, events.length);
  const ownBuffer = createOutputBuffer(device, events.length);
  const props = {events: eventView, populations: populationView, mask: maskView};
  addExcessRiskRecipe(graph, {
    ...props,
    output: importGraphBuffer(graph, 'risk', riskBuffer, 'float32', events.length)
  });
  addSpatialRateRecipe(graph, {
    ...props,
    weights,
    scale: 1000,
    output: importGraphBuffer(graph, 'spatial', spatialBuffer, 'float32', events.length)
  });
  addSpatialRateRecipe(graph, {
    ...props,
    id: 'own',
    weights,
    islandPolicy: 'own-rate',
    output: importGraphBuffer(graph, 'own-output', ownBuffer, 'float32', events.length)
  });
  const compiled = graph.compile();
  try {
    for (let repetition = 0; repetition < 2; repetition++) {
      submitGraph(device, compiled, undefined);
      const risk = await readFloat32(riskBuffer, events.length);
      const spatial = await readFloat32(spatialBuffer, events.length);
      const own = await readFloat32(ownBuffer, events.length);
      for (let row = 0; row < events.length; row++) {
        if (row < 3) expect(risk[row]).toBeCloseTo(1, 5);
        else expect(risk[row]).toBeNaN();
        if (row < 2) expect(spatial[row]).toBeCloseTo(100, 4);
        else expect(spatial[row]).toBeNaN();
      }
      expect(own[2]).toBeCloseTo(0.1, 5);
      // Mutate inputs on the same compiled graph: pooled=32/220, row0 raw=.2;
      // first neighborhood rate=22/120. Distinguishes ratio of sums from mean(.2,.1)=.15.
      events[0] = 20;
      buffers[0].write(events);
      if (repetition === 0) {
        submitGraph(device, compiled, undefined);
        const changedRisk = await readFloat32(riskBuffer, events.length);
        const changedSpatial = await readFloat32(spatialBuffer, events.length);
        expect(changedRisk[0]).toBeCloseTo(20 / ((100 * 32) / 220), 5);
        expect(changedSpatial[0]).toBeCloseTo((1000 * 22) / 120, 4);
        events[0] = 10;
        buffers[0].write(events);
      }
    }
    events.fill(0);
    buffers[0].write(events);
    submitGraph(device, compiled, undefined);
    expect((await readFloat32(riskBuffer, events.length))[0]).toBeNaN();
    expect((await readFloat32(spatialBuffer, events.length))[0]).toBe(0);
  } finally {
    compiled.destroy();
    for (const buffer of [...buffers, riskBuffer, spatialBuffer, ownBuffer]) buffer.destroy();
  }
});

it('excess risk counts only mask-included rows with an undefined rate as invalid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'excess-risk-invalid-count');
  try {
    // 200 rows, only rows 0..4 selected. Row 3 (population 0) and row 4 (NaN events) are selected
    // but undefined, so the count is 2. The 195 deselected rows are NaN but must not be counted.
    const rowCount = 200;
    const events = new Float32Array(rowCount).fill(2);
    const populations = new Float32Array(rowCount).fill(20);
    const mask = new Uint32Array(rowCount);
    mask.fill(1, 0, 5);
    populations[3] = 0;
    events[4] = NaN;
    const result = addExcessRiskRecipe(fixture.graph, {
      events: fixture.input('events', events, 'float32', rowCount),
      populations: fixture.input('populations', populations, 'float32', rowCount),
      mask: fixture.input('mask', mask, 'uint32', rowCount),
      output: fixture.output('risk', 'float32', rowCount).view
    });
    const invalidCount = result.status.stages[0].status.invalidCount as GraphDataView<'uint32'>;
    const published = fixture.output('published-invalid-count', 'uint32', 1);
    fixture.graph.add(
      createWGSLKernelNode(fixture.graph, {
        id: 'publish-invalid-count',
        operation: 'excess-risk-test-readback',
        invocationCount: 1,
        bindings: [
          {name: 'source', view: invalidCount, type: 'u32', access: 'read'},
          {name: 'output', view: published.view, type: 'u32', access: 'read_write'}
        ],
        body: 'output[outputOffset] = source[sourceOffset];'
      })
    );
    fixture.run();
    expect(await fixture.readUint32(published, 1)).toEqual([2]);
  } finally {
    fixture.destroy();
  }
});
