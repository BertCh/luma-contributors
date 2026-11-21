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
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {RecipeTestFixture} from '../recipes/recipe-harness';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {GPUSpatialEmpiricalBayesRates} from '../../../src/gpu-spatial-analysis/rate-smoothing/index';

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

it.each([
  'excess-risk',
  'spatial-rate'
] as const)('%s can correct an invalid recipe and execute using the same IDs', async recipe => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, `retry-${recipe}`);
  try {
    const events = fixture.input('events', new Uint32Array([2, 6]), 'uint32', 2);
    const populations = fixture.input('populations', new Uint32Array([10, 10]), 'uint32', 2);
    const shortPopulations = fixture.input('short-populations', new Uint32Array([10]), 'uint32', 1);
    const output = fixture.output('output', 'float32', 2);
    const weights = {
      offsets: fixture.input('offsets', new Uint32Array([0, 1, 2]), 'uint32', 3),
      neighbors: fixture.input('neighbors', new Uint32Array([1, 0]), 'uint32', 2),
      weights: fixture.input('weights', new Float32Array([1, 1]), 'float32', 2)
    };
    const add = recipe === 'excess-risk' ? addExcessRiskRecipe : addSpatialRateRecipe;
    const props = {id: 'retry', events, populations, weights, output: output.view, scale: 100};
    expect(() => add(fixture.graph, {...props, populations: shortPopulations})).toThrow(
      /populations length/
    );
    add(fixture.graph, props);
    fixture.run();
    const actual = await fixture.readFloat32(output, 2);
    const expected = recipe === 'excess-risk' ? [0.5, 1.5] : [40, 40];
    for (let row = 0; row < 2; row++) {
      expect(Number.isFinite(actual[row])).toBe(true);
      expect(actual[row]).toBeCloseTo(expected[row], 5);
    }
  } finally {
    fixture.destroy();
  }
});

it('W5-15 final and candidate loss independently suppress spatial rates and recover', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const overflow = rig.input(Uint32Array.of(0), 'uint32', 1);
    const candidateOverflow = rig.input(Uint32Array.of(0), 'uint32', 1);
    const output = rig.output('float32', 2);
    const result = addSpatialRateRecipe(rig.graph, {
      events: rig.input(Float32Array.of(1, 3), 'float32', 2),
      populations: rig.input(Float32Array.of(10, 10), 'float32', 2),
      weights: {
        ...rig.uploadWeights({
          offsets: [0, 1, 2],
          neighbors: [1, 0],
          weights: [1, 1],
          distances: []
        }),
        status: {overflow, candidateOverflow}
      },
      output: output.view
    });
    expect(result.status.stages.map(stage => stage.stage)).toEqual(['weights']);
    expect(result.status.stages[0].status.overflow).toBe(overflow);
    expect(result.status.stages[0].status.candidateOverflow).toBe(candidateOverflow);
    const compiled = rig.graph.compile();
    try {
      for (const [finalLoss, candidateLoss] of [
        [1, 0],
        [0, 0],
        [0, 1],
        [0, 0]
      ]) {
        rig.bufferOf(overflow).write(Uint32Array.of(finalLoss));
        rig.bufferOf(candidateOverflow).write(Uint32Array.of(candidateLoss));
        submitGraph(device, compiled, undefined);
        const actual = await readFloat32(output.buffer, 2);
        if (finalLoss || candidateLoss) expect(actual.every(Number.isNaN)).toBe(true);
        else
          for (const value of actual) {
            // Hand pooling: (1 + 3)/(10 + 10) = .2 for either reciprocal neighborhood.
            expect(Number.isFinite(value)).toBe(true);
            expect(value).toBeCloseTo(0.2, 6);
          }
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});

it('W5-15 S1 a final-only producer keeps its generated candidate word clear during final loss', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const overflow = rig.input(Uint32Array.of(0), 'uint32', 1);
    const output = rig.output('float32', 2);
    const result = addSpatialRateRecipe(rig.graph, {
      events: rig.input(Float32Array.of(1, 3), 'float32', 2),
      populations: rig.input(Float32Array.of(10, 10), 'float32', 2),
      weights: {
        ...rig.uploadWeights({
          offsets: [0, 1, 2],
          neighbors: [1, 0],
          weights: [1, 1],
          distances: []
        }),
        status: {overflow}
      },
      output: output.view
    });
    const status = result.status.stages[0].status;
    expect(status.overflow).toBe(overflow);
    // The candidate word is generated: a fallback onto the final view would also be length one.
    expect(status.candidateOverflow).toBeDefined();
    expect(status.candidateOverflow).not.toBe(overflow);
    const candidateWord = rig.output('uint32', 1);
    rig.graph.add(
      createWGSLKernelNode(rig.graph, {
        id: 'retain-candidate-word',
        operation: 'RateLossTest',
        invocationCount: 1,
        bindings: [
          {name: 'input', view: status.candidateOverflow!, type: 'u32', access: 'read'},
          {name: 'output', view: candidateWord.view, type: 'u32', access: 'read_write'}
        ],
        body: 'output[outputOffset] = input[inputOffset];'
      })
    );
    const compiled = rig.graph.compile();
    try {
      for (const finalLoss of [1, 0, 1]) {
        rig.bufferOf(overflow).write(Uint32Array.of(finalLoss));
        submitGraph(device, compiled, undefined);
        // Final loss is not candidate loss: the generated word stays zero on every submission.
        expect(await readUint32(candidateWord.buffer, 1)).toEqual([0]);
        const actual = await readFloat32(output.buffer, 2);
        if (finalLoss) expect(actual.every(Number.isNaN)).toBe(true);
        else
          for (const value of actual) {
            // Hand pooling: (1 + 3)/(10 + 10) = .2 for either reciprocal neighborhood.
            expect(Number.isFinite(value)).toBe(true);
            expect(value).toBeCloseTo(0.2, 6);
          }
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});

it('W5-19 negative donors create recipe islands while own rates and zero events remain defined', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const events = rig.input(Float32Array.of(6, -2), 'float32', 2);
    const populations = rig.input(Float32Array.of(30, 10), 'float32', 2);
    const mask = rig.input(Uint32Array.of(1, 1), 'uint32', 2);
    const weights = rig.uploadWeights({
      offsets: [0, 1, 2],
      neighbors: [1, 0],
      weights: [1, 1],
      distances: []
    });
    const risk = rig.output('float32', 2);
    const pooled = rig.output('float32', 2);
    const excludedIslands = rig.output('float32', 2);
    const own = rig.output('float32', 2);
    const props = {events, populations, mask, weights};
    const recipe = addExcessRiskRecipe(rig.graph, {...props, output: risk.view});
    rig.graph.add(
      new GPUSpatialEmpiricalBayesRates({...props, output: {spatialRates: pooled.view}})
    );
    addSpatialRateRecipe(rig.graph, {...props, output: excludedIslands.view});
    addSpatialRateRecipe(rig.graph, {
      ...props,
      id: 'own-rate',
      islandPolicy: 'own-rate',
      output: own.view
    });
    const invalidCount = rig.output('uint32', 1);
    rig.graph.add(
      createWGSLKernelNode(rig.graph, {
        id: 'retain-invalid-count',
        operation: 'RateEligibilityTest',
        invocationCount: 1,
        bindings: [
          {
            name: 'input',
            view: recipe.status.stages[0].status.invalidCount!,
            type: 'u32',
            access: 'read'
          },
          {name: 'output', view: invalidCount.view, type: 'u32', access: 'read_write'}
        ],
        body: 'output[outputOffset] = input[inputOffset];'
      })
    );
    const compiled = rig.graph.compile();
    try {
      for (const fixture of [
        // Hand arithmetic: negative row excluded; own 6/30=.2, pooled risk=6/(30*.2)=1.
        {
          events: [6, -2],
          mask: [1, 1],
          risk: [1, NaN],
          rate: [0.2, NaN],
          excluded: [NaN, NaN],
          invalid: 1
        },
        // Zero is eligible: pooled 2/40=.05; row 1 excess risk=2/(10*.05)=4.
        {
          events: [0, 2],
          mask: [1, 1],
          risk: [0, 4],
          rate: [0.05, 0.05],
          excluded: [0.05, 0.05],
          invalid: 0
        },
        {
          events: [0, 0],
          mask: [1, 1],
          risk: [NaN, NaN],
          rate: [0, 0],
          excluded: [0, 0],
          invalid: 2
        },
        {
          events: [6, 2],
          mask: [1, 0],
          risk: [1, NaN],
          rate: [0.2, NaN],
          excluded: [NaN, NaN],
          invalid: 0
        },
        {
          events: [6, -2],
          mask: [1, 1],
          risk: [1, NaN],
          rate: [0.2, NaN],
          excluded: [NaN, NaN],
          invalid: 1
        }
      ]) {
        rig.bufferOf(events).write(Float32Array.from(fixture.events));
        rig.bufferOf(mask).write(Uint32Array.from(fixture.mask));
        submitGraph(device, compiled, undefined);
        for (const [output, expected] of [
          [risk, fixture.risk],
          [pooled, fixture.rate],
          [own, fixture.rate],
          [excludedIslands, fixture.excluded]
        ] as const) {
          const actual = await readFloat32(output.buffer, 2);
          expected.forEach((value, row) => {
            if (Number.isNaN(value)) expect(actual[row]).toBeNaN();
            else {
              expect(Number.isFinite(actual[row])).toBe(true);
              expect(actual[row]).toBeCloseTo(value, 6);
            }
          });
        }
        expect(await readUint32(invalidCount.buffer, 1)).toEqual([fixture.invalid]);
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});
