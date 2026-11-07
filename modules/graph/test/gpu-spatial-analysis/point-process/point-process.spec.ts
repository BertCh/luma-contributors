// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUOPTICS} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics';
import {GPUPointPatternEnvelope} from '../../../src/gpu-spatial-analysis/point-process/gpu-point-pattern-envelope';
import {GPUSpatialBalancedSample} from '../../../src/gpu-spatial-analysis/point-process/gpu-spatial-balanced-sample';
import {GPUNetworkKernelDensity} from '../../../src/gpu-network/gpu-network-kernel-density';
import {GPUInhomogeneousPointPattern} from '../../../src/gpu-spatial-analysis/point-process/gpu-inhomogeneous-point-pattern';
import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window/gpu-study-window-preparation';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import opticsFixture from './point-process-fixtures.json';

function createHarness(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    values: number[]
  ) => {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
    );
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `input-${buffers.length}`,
      buffer,
      format,
      format === 'float32x2' ? values.length / 2 : values.length
    );
  };
  const output = <Format extends 'float32' | 'uint32'>(format: Format, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {
      view: importGraphBuffer(graph, `output-${buffers.length}`, buffer, format, length),
      read: () => (format === 'uint32' ? readUint32(buffer, length) : readFloat32(buffer, length))
    };
  };
  return {
    graph,
    input,
    output,
    run: () => {
      const compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      return compiled;
    },
    destroy: () => {
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

it('OPTICS matches sklearn 1.7.2 ordering, predecessors, distances and DBSCAN extraction', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device);
  const count = opticsFixture.positions.length;
  const ordering = harness.output('uint32', count);
  const coreDistances = harness.output('float32', count);
  const reachability = harness.output('float32', count);
  const predecessors = harness.output('uint32', count);
  const labels = harness.output('uint32', count);
  const status = harness.output('uint32', 3);
  harness.graph.add(
    new GPUOPTICS({
      positions: harness.input('float32x2', opticsFixture.positions.flat()),
      minimumSamples: opticsFixture.minimumSamples,
      maximumDistance: opticsFixture.maximumDistance,
      extractionDistance: opticsFixture.extractionDistance,
      output: {
        ordering: ordering.view,
        coreDistances: coreDistances.view,
        reachability: reachability.view,
        predecessors: predecessors.view,
        labels: labels.view,
        status: status.view
      }
    })
  );
  const compiled = harness.run();
  expect(await ordering.read()).toEqual(opticsFixture.ordering);
  expect(await predecessors.read()).toEqual(
    opticsFixture.predecessors.map(value => (value < 0 ? 0xffffffff : value))
  );
  expect(await labels.read()).toEqual(
    opticsFixture.labels.map(value => (value < 0 ? 0xffffffff : value))
  );
  for (const [actual, expected] of [
    [await coreDistances.read(), opticsFixture.coreDistances],
    [await reachability.read(), opticsFixture.reachability]
  ] as const) {
    for (let row = 0; row < count; row++) {
      if (expected[row] === null) {
        expect(actual[row]).toBe(Infinity);
      } else {
        expect(actual[row]).toBeCloseTo(expected[row]!, 5);
      }
    }
  }
  expect(await status.read()).toEqual([count, 1, 0]);
  compiled.destroy();
  harness.destroy();
});

it('OPTICS exposes an exhausted frontier and never presents partial cluster labels as complete', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device);
  const labels = harness.output('uint32', 4);
  const status = harness.output('uint32', 3);
  harness.graph.add(
    new GPUOPTICS({
      positions: harness.input('float32x2', [0, 0, 0, 0, 0.1, 0, NaN, 0]),
      minimumSamples: 2,
      maximumDistance: 1,
      extractionDistance: 0.5,
      maximumIterations: 1,
      output: {
        ordering: harness.output('uint32', 4).view,
        coreDistances: harness.output('float32', 4).view,
        reachability: harness.output('float32', 4).view,
        predecessors: harness.output('uint32', 4).view,
        labels: labels.view,
        status: status.view
      }
    })
  );
  const compiled = harness.run();
  expect(await status.read()).toEqual([1, 0, 1]);
  expect(await labels.read()).toEqual([0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]);
  compiled.destroy();
  harness.destroy();
});

it('simulation curves produce pointwise envelopes and a separate conservative DCLF global p-value', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device);
  const lower = harness.output('float32', 2);
  const upper = harness.output('float32', 2);
  const statistics = harness.output('float32', 4);
  const status = harness.output('uint32', 1);
  harness.graph.add(
    new GPUPointPatternEnvelope({
      observed: harness.input('float32', [1, 2]),
      simulations: harness.input('float32', [0, 0, 1, 1, 1, 2, 3, 0]),
      reference: harness.input('float32', [0, 0]),
      integrationWeights: harness.input('float32', [1, 2]),
      simulationCount: 4,
      statistic: 'integrated-squared-deviation',
      nullModel: 'independent fixed-null curves',
      fittedParameters: 'none',
      output: {
        lower: lower.view,
        upper: upper.view,
        statistics: statistics.view,
        status: status.view
      }
    })
  );
  const compiled = harness.run();
  expect(await lower.read()).toEqual([0, 0]);
  expect(await upper.read()).toEqual([3, 2]);
  const summary = await statistics.read();
  expect(summary[0]).toBe(9);
  expect(summary[1]).toBeCloseTo(3 / 5);
  expect(summary[2]).toBeCloseTo(1 / 5);
  expect(summary[3]).toBe(4);
  expect(await status.read()).toEqual([0]);
  compiled.destroy();
  harness.destroy();
});

it('pointwise envelope rank selection handles negative values and ties like a sort', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device);
  const simulationCount = 9;
  const radiusCount = 4;
  const rank = 3;
  // Rows are simulations; columns are radii. Negative values, ties and tiny magnitudes.
  const simulations = [
    [-3, 0, 5, 1e-30],
    [2, 0, 5, -1e-30],
    [-3, -7.5, 5, 4],
    [8, 1, -2, 4],
    [0.5, 1, -2, 4],
    [-1, 0, 6, -4],
    [2, 3, 6, 0],
    [-8, 3, 7, 0],
    [4, 2, 7, 9]
  ];
  const lower = harness.output('float32', radiusCount);
  const upper = harness.output('float32', radiusCount);
  const statistics = harness.output('float32', 4);
  const status = harness.output('uint32', 1);
  harness.graph.add(
    new GPUPointPatternEnvelope({
      observed: harness.input('float32', [0, 0, 0, 0]),
      simulations: harness.input('float32', simulations.flat()),
      reference: harness.input('float32', [0, 0, 0, 0]),
      integrationWeights: harness.input('float32', [1, 1, 1, 1]),
      simulationCount,
      rank,
      nullModel: 'supplied curves',
      fittedParameters: 'none',
      output: {
        lower: lower.view,
        upper: upper.view,
        statistics: statistics.view,
        status: status.view
      }
    })
  );
  const compiled = harness.run();
  const expectedLower: number[] = [];
  const expectedUpper: number[] = [];
  for (let radius = 0; radius < radiusCount; radius++) {
    const column = simulations.map(row => Math.fround(row[radius])).sort((a, b) => a - b);
    expectedLower.push(column[rank - 1]);
    expectedUpper.push(column[simulationCount - rank]);
  }
  expect(await lower.read()).toEqual(expectedLower);
  expect(await upper.read()).toEqual(expectedUpper);
  expect(await status.read()).toEqual([0]);
  compiled.destroy();
  harness.destroy();
});

it('network intensity conserves event mass over branches and disconnected support', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device);
  const intensity = harness.output('float32', 4);
  const mass = harness.output('float32', 2);
  const status = harness.output('uint32', 4);
  // A branch represented by lengths 1,2,1, and a disconnected component represented by length 3.
  harness.graph.add(
    new GPUNetworkKernelDensity({
      distances: harness.input('float32', [0, Infinity, 1, Infinity, 1, Infinity, Infinity, 0]),
      eventCount: 2,
      quadratureLengths: harness.input('float32', [1, 2, 1, 3]),
      bandwidth: 2,
      output: {intensity: intensity.view, kernelMass: mass.view, status: status.view}
    })
  );
  const compiled = harness.run();
  const values = await intensity.read();
  expect(values[0]).toBeCloseTo(0.4);
  expect(values[1]).toBeCloseTo(0.2);
  expect(values[2]).toBeCloseTo(0.2);
  expect(values[3]).toBeCloseTo(1 / 3);
  expect(values[0] + 2 * values[1] + values[2] + 3 * values[3]).toBeCloseTo(2);
  expect(await mass.read()).toEqual([1.25, 1.5]);
  expect(await status.read()).toEqual([0, 0, 0, 0]);
  compiled.destroy();
  harness.destroy();
});

it('stratified spatial sampling returns unique frame IDs with exact design weights and repeatable provenance', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device);
  const frameIds = harness.output('uint32', 3);
  const probabilities = harness.output('float32', 3);
  const weights = harness.output('float32', 3);
  const status = harness.output('uint32', 4);
  const sample = new GPUSpatialBalancedSample({
    positions: harness.input('float32x2', [0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0]),
    frameIds: harness.input('uint32', [10, 11, 12, 13, 14, 15]),
    strata: harness.input('uint32', [0, 0, 0, 0, 1, 1]),
    allocations: [
      {stratumId: 0, sampleCount: 2},
      {stratumId: 1, sampleCount: 1}
    ],
    bounds: [0, 0, 6, 1],
    seed: 7,
    output: {
      frameIds: frameIds.view,
      inclusionProbabilities: probabilities.view,
      designWeights: weights.view,
      status: status.view
    }
  });
  harness.graph.add(sample);
  const compiled = harness.run();
  const selected = await frameIds.read();
  expect(new Set(selected).size).toBe(3);
  expect(selected.slice(0, 2).every(value => value >= 10 && value <= 13)).toBe(true);
  expect(selected[2]).toBeGreaterThanOrEqual(14);
  expect(await probabilities.read()).toEqual([0.5, 0.5, 0.5]);
  expect(await weights.read()).toEqual([2, 2, 2]);
  expect(await status.read()).toEqual([2, 0, 1, 0]);
  submitGraph(device, compiled, undefined);
  expect(await frameIds.read()).toEqual(selected);
  expect(sample.provenance.generator).toBe('Philox4x32-10');
  compiled.destroy();
  harness.destroy();
});

it('inhomogeneous polygon-border K respects supplied intensity and empty erosion', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device);
  const window = new GPUStudyWindowPreparation({
    positions: harness.input('float32x2', [0, 0, 10, 0, 10, 10, 0, 10, 0, 0]),
    ringOffsets: harness.input('uint32', [0, 5]),
    polygonOffsets: harness.input('uint32', [0, 1]),
    featureOffsets: harness.input('uint32', [0, 1]),
    samplingGrid: [4, 4]
  });
  harness.graph.add(window);
  const intensity = harness.output('float32', 3);
  const values = harness.output('float32', 3);
  const status = harness.output('uint32', 3);
  harness.graph.add(
    new GPUInhomogeneousPointPattern({
      studyWindow: window,
      positions: harness.input('float32x2', [4, 5, 5, 5, 6, 5]),
      suppliedIntensity: harness.input('float32', [0.5, 1, 2]),
      radii: harness.input('float32', [1, 2, 6]),
      output: {
        intensity: intensity.view,
        k: values.view,
        pairCorrelation: harness.output('float32', 3).view,
        status: status.view
      }
    })
  );
  const compiled = harness.run();
  expect(await intensity.read()).toEqual([0.5, 1, 2]);
  const results = await values.read();
  expect(results[0]).toBeCloseTo(5 / 3.5);
  expect(results[1]).toBeCloseTo(7 / 3.5);
  expect(Number.isNaN(results[2])).toBe(true);
  expect(await status.read()).toEqual([0, 0, 1]);
  compiled.destroy();
  window.destroy();
  harness.destroy();
});

it('unequal spatial inclusion tickets preserve certainty rows and reciprocal design weights', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device);
  const frameIds = harness.output('uint32', 2);
  const probabilities = harness.output('float32', 2);
  const weights = harness.output('float32', 2);
  const status = harness.output('uint32', 2);
  harness.graph.add(
    new GPUSpatialBalancedSample({
      positions: harness.input('float32x2', [0, 0, 1, 0, 2, 0]),
      frameIds: harness.input('uint32', [0, 1, 2]),
      inclusionTickets: harness.input('uint32', [1, 1, 2]),
      allocations: [{stratumId: 0, sampleCount: 2}],
      bounds: [0, 0, 3, 1],
      seed: 7,
      output: {
        frameIds: frameIds.view,
        inclusionProbabilities: probabilities.view,
        designWeights: weights.view,
        status: status.view
      }
    })
  );
  const compiled = harness.run();
  const selected = await frameIds.read();
  const inclusion = await probabilities.read();
  const designWeights = await weights.read();
  expect(selected).toContain(2);
  expect(new Set(selected).size).toBe(2);
  for (let sample = 0; sample < 2; sample++) {
    expect(inclusion[sample]).toBe(selected[sample] === 2 ? 1 : 0.5);
    expect(designWeights[sample]).toBe(selected[sample] === 2 ? 1 : 2);
  }
  expect(await status.read()).toEqual([2, 0]);
  compiled.destroy();
  harness.destroy();
});

it('invalid simulation curves and invalid sampling frames fail explicitly without dropping observations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device);
  const statistics = harness.output('float32', 4);
  const envelopeStatus = harness.output('uint32', 1);
  harness.graph.add(
    new GPUPointPatternEnvelope({
      observed: harness.input('float32', [1]),
      simulations: harness.input('float32', [NaN, 2]),
      reference: harness.input('float32', [0]),
      integrationWeights: harness.input('float32', [1]),
      simulationCount: 2,
      nullModel: 'invalid replicate fixture',
      fittedParameters: 'none',
      output: {
        lower: harness.output('float32', 1).view,
        upper: harness.output('float32', 1).view,
        statistics: statistics.view,
        status: envelopeStatus.view
      }
    })
  );
  const selected = harness.output('uint32', 1);
  const sampleStatus = harness.output('uint32', 2);
  harness.graph.add(
    new GPUSpatialBalancedSample({
      positions: harness.input('float32x2', [NaN, 0, 1, 0]),
      frameIds: harness.input('uint32', [0, 1]),
      allocations: [{stratumId: 0, sampleCount: 1}],
      bounds: [0, 0, 2, 1],
      seed: 1,
      output: {
        frameIds: selected.view,
        inclusionProbabilities: harness.output('float32', 1).view,
        designWeights: harness.output('float32', 1).view,
        status: sampleStatus.view
      }
    })
  );
  const compiled = harness.run();
  expect(await envelopeStatus.read()).toEqual([1]);
  expect(Number.isNaN((await statistics.read())[1])).toBe(true);
  expect(await selected.read()).toEqual([0xffffffff]);
  expect(await sampleStatus.read()).toEqual([0, 1]);
  compiled.destroy();
  harness.destroy();
});

it('estimated point intensity excludes self and uses the normalized compact kernel in an interior window', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device);
  const window = new GPUStudyWindowPreparation({
    positions: harness.input('float32x2', [0, 0, 10, 0, 10, 10, 0, 10, 0, 0]),
    ringOffsets: harness.input('uint32', [0, 5]),
    polygonOffsets: harness.input('uint32', [0, 1]),
    featureOffsets: harness.input('uint32', [0, 1]),
    samplingGrid: [4, 4]
  });
  harness.graph.add(window);
  const intensity = harness.output('float32', 3);
  const status = harness.output('uint32', 1);
  harness.graph.add(
    new GPUInhomogeneousPointPattern({
      studyWindow: window,
      positions: harness.input('float32x2', [4, 5, 5, 5, 6, 5]),
      bandwidth: 2,
      radii: harness.input('float32', [1]),
      output: {
        intensity: intensity.view,
        k: harness.output('float32', 1).view,
        pairCorrelation: harness.output('float32', 1).view,
        status: status.view
      }
    })
  );
  const compiled = harness.run();
  const estimates = await intensity.read();
  expect(estimates[0]).toBeCloseTo(0.75 / (2 * Math.PI), 5);
  expect(estimates[1]).toBeCloseTo(1.5 / (2 * Math.PI), 5);
  expect(estimates[2]).toBeCloseTo(estimates[0], 5);
  expect(await status.read()).toEqual([0]);
  compiled.destroy();
  window.destroy();
  harness.destroy();
});

it('resident survey replicates have empirical inclusion frequencies and rerun changed frame IDs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device);
  const parametersBuffer = createInputBuffer(device, Uint32Array.from([17, 0]));
  const frameIdsBuffer = createInputBuffer(
    device,
    Uint32Array.from({length: 16}, (_, row) => row)
  );
  const parameters = importGraphBuffer(
    harness.graph,
    'survey-parameters',
    parametersBuffer,
    'uint32',
    2
  );
  const frameIds = importGraphBuffer(
    harness.graph,
    'survey-frame-ids',
    frameIdsBuffer,
    'uint32',
    16
  );
  const selected = harness.output('uint32', 4);
  const sampleStatus = harness.output('uint32', 2);
  harness.graph.add(
    new GPUSpatialBalancedSample({
      positions: harness.input(
        'float32x2',
        Array.from({length: 16}, (_, row) => [row % 4, Math.floor(row / 4)]).flat()
      ),
      frameIds,
      parameters,
      allocations: [{stratumId: 0, sampleCount: 4}],
      bounds: [0, 0, 4, 4],
      seed: 0,
      output: {
        frameIds: selected.view,
        inclusionProbabilities: harness.output('float32', 4).view,
        designWeights: harness.output('float32', 4).view,
        status: sampleStatus.view
      }
    })
  );
  const compiled = harness.graph.compile();
  const frequencies = new Array<number>(16).fill(0);
  let firstSample: number[] = [];
  let geographicBalance = 0;
  let estimatedTotal = 0;
  for (let replicate = 0; replicate < 256; replicate++) {
    parametersBuffer.write(Uint32Array.from([17, replicate]));
    submitGraph(device, compiled, undefined);
    const result = await selected.read();
    expect(new Set(result).size).toBe(4);
    if (replicate === 0) {
      firstSample = result;
    }
    // Equal-area quadrants give an independent geographic balance diagnostic on this frame.
    const quadrants = [0, 0, 0, 0];
    for (const row of result) {
      frequencies[row]++;
      quadrants[(row % 4 >= 2 ? 1 : 0) + (Math.floor(row / 4) >= 2 ? 2 : 0)]++;
      estimatedTotal += 4 * row;
    }
    geographicBalance += quadrants.reduce((sum, count) => sum + (count - 1) ** 2, 0);
  }
  for (const frequency of frequencies) {
    expect(Math.abs(frequency / 256 - 0.25)).toBeLessThan(0.1);
  }
  // SRS without replacement has sum_h Var(n_h)=4*(4*0.25*0.75*12/15)=2.4.
  expect(geographicBalance / 256).toBeLessThan(2.4);
  expect(Math.abs(estimatedTotal / 256 - 120)).toBeLessThan(8);
  frameIdsBuffer.write(Uint32Array.from({length: 16}, (_, row) => row + 100));
  parametersBuffer.write(Uint32Array.from([17, 0]));
  submitGraph(device, compiled, undefined);
  expect(await selected.read()).toEqual(firstSample.map(row => row + 100));
  expect(await sampleStatus.read()).toEqual([4, 0]);
  compiled.destroy();
  parametersBuffer.destroy();
  frameIdsBuffer.destroy();
  harness.destroy();
});

it('finite survey bounds retain spatial ordering when their float32 extent overflows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device);
  const positions = Array.from({length: 16}, (_, row) => [
    (row % 4) - 1.5,
    Math.floor(row / 4) - 1.5
  ]).flat();
  const frameIds = harness.input(
    'uint32',
    Array.from({length: 16}, (_, row) => row)
  );
  const coordinateScale = 2 ** 127;
  const comparisons = [];
  for (let replicate = 0; replicate < 8; replicate++) {
    const outputs = [];
    for (const scale of [1, coordinateScale]) {
      const selected = harness.output('uint32', 4);
      const status = harness.output('uint32', 2);
      harness.graph.add(
        new GPUSpatialBalancedSample({
          id: `extent-${replicate}-${scale === 1 ? 'ordinary' : 'large'}`,
          positions: harness.input(
            'float32x2',
            positions.map(value => value * scale)
          ),
          frameIds,
          allocations: [{stratumId: 0, sampleCount: 4}],
          bounds: [-1.5 * scale, -1.5 * scale, 1.5 * scale, 1.5 * scale],
          seed: 42,
          replicate,
          output: {
            frameIds: selected.view,
            inclusionProbabilities: harness.output('float32', 4).view,
            designWeights: harness.output('float32', 4).view,
            status: status.view
          }
        })
      );
      outputs.push({selected, status});
    }
    comparisons.push(outputs);
  }
  const compiled = harness.run();
  try {
    for (const [ordinary, large] of comparisons) {
      expect(await large.status.read()).toEqual([4, 0]);
      expect(await ordinary.status.read()).toEqual([4, 0]);
      expect(await large.selected.read()).toEqual(await ordinary.selected.read());
    }
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});
