// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import type {GPUVectorFormat} from '@luma-contributors/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPULocalKriging,
  type GPULocalKrigingProps
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {
  GPULocalRBF,
  type GPULocalRBFProps
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-rbf';
import {GPUNaturalNeighbor} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-natural-neighbor';
import {GPUConditionalGaussianSimulation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation';
import {GPUVariogramModelFit} from '../../../src/gpu-dataframe/pair-statistics/gpu-variogram-model-fit';
import {
  evaluateVariogramModel,
  fitVariogramModel
} from '../../../src/gpu-dataframe/pair-statistics/variogram-model';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {INTERPOLATION_COVERAGE_REFERENCE as reference} from './interpolation-coverage-reference';

async function createHarness() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let nextBuffer = 0;
  return {
    graph,
    input<Format extends GPUVectorFormat>(data: readonly number[], format: Format, components = 1) {
      const buffer = createInputBuffer(
        device,
        format === 'uint32' ? new Uint32Array(data) : new Float32Array(data)
      );
      buffers.push(buffer);
      return importGraphBuffer(
        graph,
        `input-${nextBuffer++}`,
        buffer,
        format,
        data.length / components
      );
    },
    output<Format extends GPUVectorFormat>(length: number, format: Format, components = 1) {
      const buffer = createOutputBuffer(device, length * components);
      buffers.push(buffer);
      const view = importGraphBuffer(graph, `output-${nextBuffer++}`, buffer, format, length);
      return {
        view,
        read: () =>
          format === 'uint32'
            ? readUint32(buffer, length * components)
            : readFloat32(buffer, length * components)
      };
    },
    async run() {
      const compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      return () => {
        compiled.destroy();
        for (const buffer of buffers) {
          buffer.destroy();
        }
      };
    }
  };
}

for (const mode of ['simple', 'ordinary', 'universal', 'external-drift', 'co-kriging'] as const) {
  it(`local ${mode} kriging matches independent NumPy constrained covariance solves`, async () => {
    const harness = await createHarness();
    const queryCount = reference.queries.length / 2;
    const result = harness.output(queryCount, 'float32x4', 4);
    const status = harness.output(queryCount, 'uint32');
    const observations = reference.values.flatMap((value, row) => [
      value,
      reference.errors[row],
      mode === 'co-kriging' ? row % 2 : 0,
      1
    ]);
    const properties: GPULocalKrigingProps = {
      positions: harness.input(reference.positions, 'float32x2', 2),
      observations: harness.input(observations, 'float32x4', 4),
      supports: harness.input(reference.queries, 'float32x2', 2),
      mode,
      covariance: reference.covariance,
      means: [1.2, 2.3],
      neighborCount: 6,
      output: {values: result.view, status: status.view}
    };
    if (mode === 'co-kriging') {
      properties.coregionalization = [1, 0, 0.4, 0.8];
      properties.targetVariable = 1;
    }
    if (mode === 'external-drift') {
      properties.driftCount = 2;
      properties.sampleDrift = harness.input(
        reference.values.flatMap((_, row) => [1, reference.positions[row * 2] ** 2, 0, 0]),
        'float32x4',
        4
      );
      properties.queryDrift = harness.input(
        Array.from({length: queryCount}, (_, row) => [
          1,
          reference.queries[row * 2] ** 2,
          0,
          0
        ]).flat(),
        'float32x4',
        4
      );
    }
    harness.graph.add(new GPULocalKriging(properties));
    const cleanup = await harness.run();
    try {
      expect(Array.from(await status.read())).toEqual([0, 0, 0, 0]);
      const actual = await result.read();
      const expected = mode === 'co-kriging' ? reference.co : reference.kriging[mode];
      for (let query = 0; query < queryCount; query++) {
        expect(actual[query * 4]).toBeCloseTo(expected[query][0], 4);
        expect(actual[query * 4 + 1]).toBeCloseTo(expected[query][1], 4);
      }
    } finally {
      cleanup();
    }
  });
}

it('block-support kriging includes observation error and double support covariance quadrature', async () => {
  const harness = await createHarness();
  const result = harness.output(4, 'float32x4', 4);
  const status = harness.output(4, 'uint32');
  const supports = Array.from({length: 4}, (_, query) => [
    reference.queries[query * 2] - 0.1,
    reference.queries[query * 2 + 1],
    reference.queries[query * 2] + 0.1,
    reference.queries[query * 2 + 1]
  ]).flat();
  harness.graph.add(
    new GPULocalKriging({
      positions: harness.input(reference.positions, 'float32x2', 2),
      observations: harness.input(
        reference.values.flatMap((value, row) => [value, reference.errors[row], 0, 1]),
        'float32x4',
        4
      ),
      supports: harness.input(supports, 'float32x2', 2),
      supportCount: 2,
      neighborCount: 6,
      mode: 'ordinary',
      covariance: reference.covariance,
      output: {values: result.view, status: status.view}
    })
  );
  const cleanup = await harness.run();
  try {
    expect(Array.from(await status.read())).toEqual([0, 0, 0, 0]);
    const actual = await result.read();
    for (let query = 0; query < 4; query++) {
      expect(actual[query * 4]).toBeCloseTo(reference.block[query][0], 4);
      expect(actual[query * 4 + 1]).toBeCloseTo(reference.block[query][1], 4);
    }
  } finally {
    cleanup();
  }
});

for (const [kernel, oracleKernel] of [
  ['thin-plate-spline', 'thin_plate_spline'],
  ['cubic', 'cubic'],
  ['gaussian', 'gaussian'],
  ['linear', 'linear']
] as const) {
  it(`local ${kernel} RBF matches scipy.interpolate.RBFInterpolator`, async () => {
    const harness = await createHarness();
    const result = harness.output(4, 'float32x4', 4);
    const status = harness.output(4, 'uint32');
    const properties: GPULocalRBFProps = {
      positions: harness.input(reference.positions, 'float32x2', 2),
      values: harness.input(reference.values, 'float32'),
      queries: harness.input(reference.queries, 'float32x2', 2),
      kernel,
      epsilon: 0.8,
      smoothing: 0.02,
      neighborCount: 6,
      degree: kernel === 'gaussian' ? 0 : undefined,
      output: {values: result.view, status: status.view}
    };
    harness.graph.add(new GPULocalRBF(properties));
    const cleanup = await harness.run();
    try {
      expect(Array.from(await status.read())).toEqual([0, 0, 0, 0]);
      const actual = await result.read();
      for (let query = 0; query < 4; query++) {
        expect(actual[query * 4]).toBeCloseTo(reference.rbf[oracleKernel][query], 4);
      }
    } finally {
      cleanup();
    }
  });
}

it('RBF and universal kriging report collinear polynomial rank failure', async () => {
  const harness = await createHarness();
  const positions = harness.input([0, 0, 1, 0, 2, 0], 'float32x2', 2);
  const queries = harness.input([0.5, 0.2], 'float32x2', 2);
  const radialResult = harness.output(1, 'float32x4', 4);
  const radialStatus = harness.output(1, 'uint32');
  const krigingResult = harness.output(1, 'float32x4', 4);
  const krigingStatus = harness.output(1, 'uint32');
  harness.graph.add(
    new GPULocalRBF({
      positions,
      queries,
      values: harness.input([1, 2, 3], 'float32'),
      neighborCount: 3,
      output: {values: radialResult.view, status: radialStatus.view}
    })
  );
  harness.graph.add(
    new GPULocalKriging({
      positions,
      supports: queries,
      observations: harness.input([1, 0, 0, 1, 2, 0, 0, 1, 3, 0, 0, 1], 'float32x4', 4),
      mode: 'universal',
      neighborCount: 3,
      covariance: reference.covariance,
      output: {values: krigingResult.view, status: krigingStatus.view}
    })
  );
  const cleanup = await harness.run();
  try {
    expect((await radialStatus.read())[0]).toBe(2);
    expect((await krigingStatus.read())[0]).toBe(2);
    expect(Number.isNaN((await radialResult.read())[0])).toBe(true);
  } finally {
    cleanup();
  }
});

it('Sibson weights match independent Shapely stolen Voronoi areas and hull limits', async () => {
  const harness = await createHarness();
  const result = harness.output(4, 'float32x4', 4);
  const status = harness.output(4, 'uint32');
  const weights = harness.output(24, 'float32');
  harness.graph.add(
    new GPUNaturalNeighbor({
      positions: harness.input(reference.positions, 'float32x2', 2),
      values: harness.input(reference.values, 'float32'),
      queries: harness.input([0.8, 1.1, 0.6, 0.7, 3, 1, 1, 0], 'float32x2', 2),
      output: {values: result.view, status: status.view, weights: weights.view}
    })
  );
  const cleanup = await harness.run();
  try {
    expect(Array.from(await status.read())).toEqual([0, 0, 1, 0]);
    const actual = await result.read();
    const actualWeights = await weights.read();
    expect(actual[0]).toBeCloseTo(reference.natural.value, 4);
    expect(actual[4]).toBeCloseTo(2.2, 5);
    expect(actual[12]).toBeCloseTo(2, 5);
    for (let sample = 0; sample < 6; sample++) {
      expect(actualWeights[sample]).toBeCloseTo(reference.natural.weights[sample], 4);
      expect(actualWeights[sample]).toBeGreaterThanOrEqual(0);
    }
    expect(
      Array.from(actualWeights.slice(0, 6)).reduce((sum, weight) => sum + weight, 0)
    ).toBeCloseTo(1, 5);
    for (let coordinate = 0; coordinate < 2; coordinate++) {
      expect(
        Array.from(actualWeights.slice(0, 6)).reduce(
          (sum, weight, row) => sum + weight * reference.positions[row * 2 + coordinate],
          0
        )
      ).toBeCloseTo([0.8, 1.1][coordinate], 4);
    }
  } finally {
    cleanup();
  }
});

it('Sibson reports deliberate polygon capacity exhaustion', async () => {
  const harness = await createHarness();
  const result = harness.output(1, 'float32x4', 4);
  const status = harness.output(1, 'uint32');
  harness.graph.add(
    new GPUNaturalNeighbor({
      positions: harness.input([0, 0, 2, 0, 2, 2, 0, 2], 'float32x2', 2),
      values: harness.input([0, 2, 4, 2], 'float32'),
      queries: harness.input([1, 1], 'float32x2', 2),
      maximumVertexCount: 3,
      output: {values: result.view, status: status.view}
    })
  );
  const cleanup = await harness.run();
  try {
    expect((await status.read())[0]).toBe(3);
  } finally {
    cleanup();
  }
});

it('conditional simulation matches analytic joint conditional moments and ensemble covariance', async () => {
  const harness = await createHarness();
  const replicateCount = 4096;
  const values = harness.output(replicateCount * 2, 'float32');
  const moments = harness.output(2, 'float32x2', 2);
  const status = harness.output(replicateCount, 'uint32');
  harness.graph.add(
    new GPUConditionalGaussianSimulation({
      positions: harness.input(reference.positions, 'float32x2', 2),
      observations: harness.input(
        reference.values.flatMap((value, row) => [value, reference.errors[row], 0, 1]),
        'float32x4',
        4
      ),
      queries: harness.input(reference.queries.slice(0, 4), 'float32x2', 2),
      covariance: reference.covariance,
      mean: 1.2,
      seed: 143,
      replicateCount,
      output: {values: values.view, moments: moments.view, status: status.view}
    })
  );
  const cleanup = await harness.run();
  try {
    expect(Array.from(await status.read()).every(value => value === 0)).toBe(true);
    const actualMoments = await moments.read();
    const actual = await values.read();
    const means = [0, 0];
    for (let replicate = 0; replicate < replicateCount; replicate++) {
      for (let query = 0; query < 2; query++) {
        means[query] += actual[replicate * 2 + query] / replicateCount;
      }
    }
    for (let query = 0; query < 2; query++) {
      expect(actualMoments[query * 2]).toBeCloseTo(reference.simulation.mean[query], 4);
      expect(actualMoments[query * 2 + 1]).toBeCloseTo(
        reference.simulation.covariance[query][query],
        4
      );
      expect(Math.abs(means[query] - reference.simulation.mean[query])).toBeLessThan(0.05);
    }
    for (let first = 0; first < 2; first++) {
      for (let second = 0; second < 2; second++) {
        let covariance = 0;
        for (let replicate = 0; replicate < replicateCount; replicate++) {
          covariance +=
            ((actual[replicate * 2 + first] - means[first]) *
              (actual[replicate * 2 + second] - means[second])) /
            replicateCount;
        }
        expect(Math.abs(covariance - reference.simulation.covariance[first][second])).toBeLessThan(
          0.06
        );
      }
    }
  } finally {
    cleanup();
  }
});

for (const weighting of ['none', 'pairs', 'cressie'] as const) {
  it(`resident variogram fitting preserves ${weighting} fitting conventions`, async () => {
    const harness = await createHarness();
    const distances = Array.from({length: 20}, (_, row) => (row + 1) * 0.2);
    const pairCounts = distances.map((_, row) => 10 + row);
    const model = {model: 'exponential' as const, range: 2.2, nugget: 0.2, sill: 1.5};
    const semivariances = distances.map(distance => evaluateVariogramModel(model, distance));
    const parameters = harness.output(1, 'float32x4', 4);
    const diagnostics = harness.output(1, 'float32x4', 4);
    const status = harness.output(1, 'uint32');
    harness.graph.add(
      new GPUVariogramModelFit({
        distances: harness.input(distances, 'float32'),
        semivariances: harness.input(semivariances, 'float32'),
        pairCounts: harness.input(pairCounts, 'uint32'),
        weighting,
        models: ['exponential'],
        output: {parameters: parameters.view, diagnostics: diagnostics.view, status: status.view}
      })
    );
    const cleanup = await harness.run();
    try {
      expect((await status.read())[0]).toBe(0);
      const actual = await parameters.read();
      const expected = fitVariogramModel(
        {distances, semivariances, pairCounts},
        {model: 'exponential', weighting}
      );
      expect(actual[0]).toBeCloseTo(expected.nugget, 3);
      expect(actual[1]).toBeCloseTo(expected.sill, 3);
      expect(actual[2]).toBeCloseTo(expected.range, 3);
      expect((await diagnostics.read())[0]).toBe(20);
    } finally {
      cleanup();
    }
  });
}

it('conditional streams preserve replicate identity across batches and reproduce exact noiseless data', async () => {
  const harness = await createHarness();
  const positions = harness.input([0, 0], 'float32x2', 2);
  const observations = harness.input([5, 0, 0, 1], 'float32x4', 4);
  const queries = harness.input([0, 0, 1, 0], 'float32x2', 2);
  const full = {
    values: harness.output(16, 'float32'),
    moments: harness.output(2, 'float32x2', 2),
    status: harness.output(8, 'uint32')
  };
  const batch = {
    values: harness.output(8, 'float32'),
    moments: harness.output(2, 'float32x2', 2),
    status: harness.output(4, 'uint32')
  };
  for (const [output, replicateCount, replicateOffset] of [
    [full, 8, 0],
    [batch, 4, 4]
  ] as const) {
    harness.graph.add(
      new GPUConditionalGaussianSimulation({
        id: `simulation-${replicateOffset}`,
        positions,
        observations,
        queries,
        covariance: {model: 'exponential', range: 2, sill: 1},
        seed: 347,
        replicateCount,
        replicateOffset,
        output: {
          values: output.values.view,
          moments: output.moments.view,
          status: output.status.view
        }
      })
    );
  }
  const cleanup = await harness.run();
  try {
    expect(Array.from(await full.status.read())).toEqual(Array(8).fill(0));
    expect(Array.from(await batch.status.read())).toEqual(Array(4).fill(0));
    const values = await full.values.read();
    expect(Array.from(await batch.values.read())).toEqual(Array.from(values.slice(8)));
    for (let replicate = 0; replicate < 8; replicate++) {
      expect(values[replicate * 2]).toBe(5);
    }
    expect((await full.moments.read())[1]).toBe(0);
  } finally {
    cleanup();
  }
});

it('variogram fitting distinguishes missing bins and exhausted refinement from convergence', async () => {
  const harness = await createHarness();
  const distances = harness.input([0.2, 0.4, 0.8, 1.2], 'float32');
  const semivariances = harness.input([0.3, 0.7, 1, 1.3], 'float32');
  const missing = {
    parameters: harness.output(1, 'float32x4', 4),
    diagnostics: harness.output(1, 'float32x4', 4),
    status: harness.output(1, 'uint32')
  };
  const exhausted = {
    parameters: harness.output(1, 'float32x4', 4),
    diagnostics: harness.output(1, 'float32x4', 4),
    status: harness.output(1, 'uint32')
  };
  for (const [name, counts, output, maximumIterations] of [
    ['missing', [0, 0, 0, 0], missing, 32],
    ['exhausted', [10, 10, 10, 10], exhausted, 0]
  ] as const) {
    harness.graph.add(
      new GPUVariogramModelFit({
        id: `fit-${name}`,
        distances,
        semivariances,
        pairCounts: harness.input(counts, 'uint32'),
        models: ['gaussian'],
        maximumIterations,
        output: {
          parameters: output.parameters.view,
          diagnostics: output.diagnostics.view,
          status: output.status.view
        }
      })
    );
  }
  const cleanup = await harness.run();
  try {
    expect((await missing.status.read())[0]).toBe(1);
    expect(Number.isNaN((await missing.parameters.read())[0])).toBe(true);
    expect((await exhausted.status.read())[0]).toBe(3);
    expect(Number.isFinite((await exhausted.parameters.read())[0])).toBe(true);
  } finally {
    cleanup();
  }
});

it('natural neighbors reject duplicate and collinear samples away from exact hits', async () => {
  const harness = await createHarness();
  const queries = harness.input([0.4, 0.3], 'float32x2', 2);
  const values = harness.input([1, 2, 3, 4], 'float32');
  const duplicate = {
    result: harness.output(1, 'float32x4', 4),
    status: harness.output(1, 'uint32')
  };
  const collinear = {
    result: harness.output(1, 'float32x4', 4),
    status: harness.output(1, 'uint32')
  };
  for (const [name, positions, output] of [
    ['duplicate', [0, 0, 1, 0, 0, 1, 0, 0], duplicate],
    ['collinear', [0, 0, 1, 0, 2, 0, 3, 0], collinear]
  ] as const) {
    harness.graph.add(
      new GPUNaturalNeighbor({
        id: name,
        positions: harness.input(positions, 'float32x2', 2),
        values,
        queries,
        output: {values: output.result.view, status: output.status.view}
      })
    );
  }
  const cleanup = await harness.run();
  try {
    expect((await duplicate.status.read())[0]).toBe(2);
    expect((await collinear.status.read())[0]).toBe(2);
  } finally {
    cleanup();
  }
});

it('local kriging graph replays changed observations and resets invalid outputs and weights', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const positions = createInputBuffer(device, new Float32Array([0, 0, 1, 0, 0, 1]));
  const observations = createInputBuffer(
    device,
    new Float32Array([1, 0.1, 0, 1, 2, 0.1, 0, 1, 3, 0.1, 0, 1])
  );
  const supports = createInputBuffer(device, new Float32Array([0.3, 0.3]));
  const result = createOutputBuffer(device, 4);
  const status = createOutputBuffer(device, 1);
  const weights = createOutputBuffer(device, 3);
  graph.add(
    new GPULocalKriging({
      positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', 3),
      observations: importGraphBuffer(graph, 'observations', observations, 'float32x4', 3),
      supports: importGraphBuffer(graph, 'supports', supports, 'float32x2', 1),
      covariance: reference.covariance,
      mode: 'ordinary',
      neighborCount: 3,
      output: {
        values: importGraphBuffer(graph, 'result', result, 'float32x4', 1),
        status: importGraphBuffer(graph, 'status', status, 'uint32', 1),
        weights: importGraphBuffer(graph, 'weights', weights, 'float32', 3)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const first = await readFloat32(result, 4);
    expect((await readUint32(status, 1))[0]).toBe(0);
    observations.write(new Float32Array([8, 0.1, 0, 1, 9, 0.1, 0, 1, 10, 0.1, 0, 1]));
    submitGraph(device, compiled, undefined);
    const second = await readFloat32(result, 4);
    expect(second[0] - first[0]).toBeCloseTo(7, 4);
    expect(second[1]).toBe(first[1]);
    observations.write(new Float32Array(12));
    submitGraph(device, compiled, undefined);
    expect((await readUint32(status, 1))[0]).toBe(1);
    expect(Number.isNaN((await readFloat32(result, 4))[0])).toBe(true);
    expect(Array.from(await readFloat32(weights, 3))).toEqual([0, 0, 0]);
  } finally {
    compiled.destroy();
    for (const buffer of [positions, observations, supports, result, status, weights]) {
      buffer.destroy();
    }
  }
});

it('co-kriging reports singular supplied PSD models and external drift reports missing query drift', async () => {
  const harness = await createHarness();
  const positions = harness.input([0, 0, 0, 0], 'float32x2', 2);
  const queries = harness.input([0.5, 0.5], 'float32x2', 2);
  const coResult = harness.output(1, 'float32x4', 4);
  const coStatus = harness.output(1, 'uint32');
  harness.graph.add(
    new GPULocalKriging({
      id: 'singular-co',
      positions,
      supports: queries,
      observations: harness.input([1, 0, 0, 1, 2, 0, 1, 1], 'float32x4', 4),
      mode: 'co-kriging',
      covariance: reference.covariance,
      coregionalization: [1, 0, 1, 0],
      neighborCount: 2,
      output: {values: coResult.view, status: coStatus.view}
    })
  );
  const driftResult = harness.output(1, 'float32x4', 4);
  const driftStatus = harness.output(1, 'uint32');
  harness.graph.add(
    new GPULocalKriging({
      id: 'missing-drift',
      positions,
      supports: queries,
      observations: harness.input([1, 0.1, 0, 1, 2, 0.1, 0, 1], 'float32x4', 4),
      mode: 'external-drift',
      covariance: reference.covariance,
      neighborCount: 2,
      sampleDrift: harness.input([1, 0, 0, 0, 1, 0, 0, 0], 'float32x4', 4),
      queryDrift: harness.input([NaN, 0, 0, 0], 'float32x4', 4),
      output: {values: driftResult.view, status: driftStatus.view}
    })
  );
  const cleanup = await harness.run();
  try {
    expect((await coStatus.read())[0]).toBe(2);
    expect((await driftStatus.read())[0]).toBe(1);
  } finally {
    cleanup();
  }
});

for (const model of ['spherical', 'exponential', 'gaussian'] as const) {
  it(`resident ${model} variogram fits match independent scipy.optimize.least_squares`, async () => {
    const harness = await createHarness();
    const parameters = harness.output(1, 'float32x4', 4);
    const diagnostics = harness.output(1, 'float32x4', 4);
    const status = harness.output(1, 'uint32');
    const fixture = reference.variogram.fits[model];
    harness.graph.add(
      new GPUVariogramModelFit({
        distances: harness.input(reference.variogram.distances, 'float32'),
        semivariances: harness.input(fixture.semivariances, 'float32'),
        pairCounts: harness.input(reference.variogram.pairCounts, 'uint32'),
        models: [model],
        weighting: 'pairs',
        output: {parameters: parameters.view, diagnostics: diagnostics.view, status: status.view}
      })
    );
    const cleanup = await harness.run();
    try {
      expect((await status.read())[0]).toBe(0);
      const actual = await parameters.read();
      for (let parameter = 0; parameter < 3; parameter++) {
        expect(Math.abs(actual[parameter] - fixture.parameters[parameter])).toBeLessThan(0.002);
      }
      expect(Math.abs(actual[3] - fixture.residual)).toBeLessThan(1e-4);
    } finally {
      cleanup();
    }
  });
}
