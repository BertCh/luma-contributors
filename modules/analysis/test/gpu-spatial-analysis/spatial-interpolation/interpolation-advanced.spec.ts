// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPULocalKriging,
  type GPUPredictionCovariance
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {
  GPULocalRBF,
  type GPULocalRBFProps
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-rbf';
import {GPUNaturalNeighbor} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-natural-neighbor';
import {GPUConditionalGaussianSimulation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation';
import {GPUVariogramModelFit} from '../../../src/dataframe-analysis/pair-statistics/gpu-variogram-model-fit';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import reference from './interpolation-advanced-reference.json';
import gstatReference from './interpolation-gstat-reference.json';
import defaultDegreeReference from '../../oracles/fixtures/interpolation-rbf-default-degree.json';
const covariance: GPUPredictionCovariance = {...reference.covariance, model: 'exponential'};
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

it('natural-neighbor duplicate hits average finite large values without overflow', async () => {
  const harness = await createHarness();
  const values = harness.output(1, 'float32x4', 4);
  const status = harness.output(1, 'uint32');
  harness.graph.add(
    new GPUNaturalNeighbor({
      positions: harness.input([0, 0, 0, 0, 1, 0, 0, 1], 'float32x2', 2),
      values: harness.input([3e38, 3e38, 0, 0], 'float32'),
      queries: harness.input([0, 0], 'float32x2', 2),
      output: {values: values.view, status: status.view}
    })
  );
  const cleanup = await harness.run();
  try {
    expect(Array.from(await status.read())).toEqual([0]);
    expect((await values.read())[0]).toBe(Math.fround(3e38));
  } finally {
    cleanup();
  }
});

it('variogram fitting reports arithmetic failure when every candidate residual overflows', async () => {
  const harness = await createHarness();
  const parameters = harness.output(1, 'float32x4', 4);
  const diagnostics = harness.output(1, 'float32x4', 4);
  const status = harness.output(1, 'uint32');
  harness.graph.add(
    new GPUVariogramModelFit({
      distances: harness.input([1, 2, 3, 4], 'float32'),
      semivariances: harness.input([1e30, 3e30, 2e30, 4e30], 'float32'),
      pairCounts: harness.input([10, 10, 10, 10], 'uint32'),
      models: ['exponential'],
      weighting: 'none',
      output: {parameters: parameters.view, diagnostics: diagnostics.view, status: status.view}
    })
  );
  const cleanup = await harness.run();
  try {
    expect(Array.from(await status.read())).toEqual([2]);
    expect(Array.from(await parameters.read()).every(Number.isNaN)).toBe(true);
    expect(Array.from(await diagnostics.read()).every(Number.isNaN)).toBe(true);
  } finally {
    cleanup();
  }
});

for (const mode of ['simple', 'ordinary', 'universal', 'external-drift'] as const) {
  it(`${mode} prediction and latent variance match direct GSTools1.7 and PyKrige1.7.3`, async () => {
    const harness = await createHarness();
    const values = harness.output(3, 'float32x4', 4);
    const status = harness.output(3, 'uint32');
    harness.graph.add(
      new GPULocalKriging({
        positions: harness.input(reference.positions.flat(), 'float32x2', 2),
        observations: harness.input(
          reference.values.flatMap(value => [value, 0, 0, 1]),
          'float32x4',
          4
        ),
        supports: harness.input(reference.queries.flat(), 'float32x2', 2),
        covariance,
        mode,
        means: [1.2],
        neighborCount: 6,
        ...(mode === 'external-drift'
          ? {
              driftCount: 2,
              sampleDrift: harness.input(
                reference.positions.flatMap(point => [1, point[0] ** 2]),
                'float32'
              ),
              queryDrift: harness.input(
                reference.queries.flatMap(point => [1, point[0] ** 2]),
                'float32'
              )
            }
          : {}),
        output: {values: values.view, status: status.view}
      })
    );
    const cleanup = await harness.run();
    try {
      expect(Array.from(await status.read())).toEqual([0, 0, 0]);
      const actual = await values.read();
      for (const expected of Object.values(reference.kriging[mode])) {
        for (let row = 0; row < 3; row++) {
          expect(actual[row * 4]).toBeCloseTo(expected[row][0], 4);
          expect(actual[row * 4 + 1]).toBeCloseTo(expected[row][1], 4);
        }
      }
    } finally {
      cleanup();
    }
  });
}

it('32-neighbor weighted blocks with8 arbitrary supplied drift columns match independent constrained solves', async () => {
  const harness = await createHarness();
  const data = reference.wide;
  const values = harness.output(3, 'float32x4', 4);
  const status = harness.output(3, 'uint32');
  const weights = harness.output(96, 'float32');
  harness.graph.add(
    new GPULocalKriging({
      positions: harness.input(data.positions.flat(), 'float32x2', 2),
      observations: harness.input(
        data.values.flatMap(value => [value, 0.07, 0, 1]),
        'float32x4',
        4
      ),
      supports: harness.input(
        data.supports.flatMap(support =>
          support.flatMap((point, index) => [...point, data.supportWeights[index], 0])
        ),
        'float32x4',
        4
      ),
      supportCount: 4,
      mode: 'universal',
      covariance,
      neighborCount: 32,
      driftCount: 8,
      sampleDrift: harness.input(data.sampleDesign.flat(), 'float32'),
      queryDrift: harness.input(data.queryDesign.flat(), 'float32'),
      output: {values: values.view, status: status.view, weights: weights.view}
    })
  );
  const cleanup = await harness.run();
  try {
    expect(Array.from(await status.read())).toEqual([0, 0, 0]);
    const actual = await values.read();
    for (let row = 0; row < 3; row++) {
      expect(actual[row * 4]).toBeCloseTo(data.expected[row][0], 3);
      expect(actual[row * 4 + 1]).toBeCloseTo(data.expected[row][1], 3);
    }
    const coefficients = await weights.read();
    for (let row = 0; row < 3; row++) {
      expect(
        Array.from(coefficients.slice(row * 32, row * 32 + 32)).reduce(
          (sum, value) => sum + value,
          0
        )
      ).toBeCloseTo(1, 4);
    }
  } finally {
    cleanup();
  }
});

it('three-variable unknown-mean two-structure PSD LMC honors masks, weighted supports and unbiasedness', async () => {
  const harness = await createHarness();
  const data = reference.lmc;
  const values = harness.output(1, 'float32x4', 4);
  const status = harness.output(1, 'uint32');
  const weights = harness.output(32, 'float32');
  harness.graph.add(
    new GPULocalKriging({
      positions: harness.input(data.positions.flat(), 'float32x2', 2),
      observations: harness.input(data.observations.flat(), 'float32x4', 4),
      supports: harness.input(
        data.supports.flatMap((point, index) => [...point, data.supportWeights[index], 0]),
        'float32x4',
        4
      ),
      supportCount: 3,
      mode: 'co-kriging',
      coKrigingMean: 'ordinary',
      variableCount: 3,
      targetVariable: 2,
      coregionalizationModels: data.structures.map(structure => ({
        ...structure,
        covariance: {
          ...structure.covariance,
          model: structure.covariance.model as GPUPredictionCovariance['model']
        }
      })),
      covariance,
      neighborCount: 32,
      output: {values: values.view, status: status.view, weights: weights.view}
    })
  );
  const cleanup = await harness.run();
  try {
    expect(Array.from(await status.read())).toEqual([0]);
    const actual = await values.read();
    expect(actual[0]).toBeCloseTo(data.expected[0], 4);
    expect(actual[1]).toBeCloseTo(data.expected[1], 4);
    expect(actual[0]).toBeCloseTo(gstatReference.lmc[0], 4);
    expect(actual[1]).toBeCloseTo(gstatReference.lmc[1], 4);
    expect(actual[3]).toBe(29);
    const center = [0, 1].map(axis =>
      data.supports.reduce((sum, point, index) => sum + point[axis] * data.supportWeights[index], 0)
    );
    const order = data.positions
      .map((point, index) => ({
        index,
        distance: Math.hypot(point[0] - center[0], point[1] - center[1])
      }))
      .filter(item => data.observations[item.index][3] > 0)
      .sort((first, second) => first.distance - second.distance || first.index - second.index);
    const coefficients = await weights.read();
    for (let variable = 0; variable < 3; variable++) {
      expect(
        order.reduce(
          (sum, item, index) =>
            sum + (data.observations[item.index][2] === variable ? coefficients[index] : 0),
          0
        )
      ).toBeCloseTo(variable === 2 ? 1 : 0, 4);
    }
  } finally {
    cleanup();
  }
});

for (const kernel of Object.keys(reference.radial.expected) as NonNullable<
  GPULocalRBFProps['kernel']
>[]) {
  it(`${kernel} indexed32-neighbor quadratic RBF with per-sample smoothing matches SciPy`, async () => {
    const harness = await createHarness();
    const data = reference.radial;
    const values = harness.output(4, 'float32x4', 4);
    const status = harness.output(4, 'uint32');
    harness.graph.add(
      new GPULocalRBF({
        positions: harness.input(data.positions.flat(), 'float32x2', 2),
        values: harness.input(data.values, 'float32'),
        queries: harness.input(data.queries.flat(), 'float32x2', 2),
        sampleSmoothing: harness.input(data.smoothing, 'float32'),
        neighborhoods: harness.input(data.neighborhoods.flat(), 'uint32'),
        neighborhoodCapacity: 32,
        neighborCount: 32,
        kernel,
        degree: 2,
        epsilon: 0.9,
        output: {values: values.view, status: status.view}
      })
    );
    const cleanup = await harness.run();
    try {
      expect(Array.from(await status.read())).toEqual([0, 0, 0, 0]);
      const actual = await values.read();
      const expected = reference.radial.expected[kernel];
      for (let row = 0; row < 4; row++) {
        expect(actual[row * 4]).toBeCloseTo(expected[row], 3);
        expect(actual[row * 4 + 2]).toBe(32);
      }
    } finally {
      cleanup();
    }
  });
}

// Math correctness review, interpolation-density-12: the SciPy fixture above always passes
// degree 2. This one leaves `degree` out, against SciPy fitted at the default GPULocalRBF
// documents for each kernel (`test/oracles/interpolation-rbf-default-degree.py`).
for (const kernel of Object.keys(defaultDegreeReference.expected) as NonNullable<
  GPULocalRBFProps['kernel']
>[]) {
  it(`${kernel} RBF at its default polynomial degree matches SciPy`, async () => {
    const harness = await createHarness();
    const data = reference.radial;
    const values = harness.output(4, 'float32x4', 4);
    const status = harness.output(4, 'uint32');
    const contributor = new GPULocalRBF({
      positions: harness.input(data.positions.flat(), 'float32x2', 2),
      values: harness.input(data.values, 'float32'),
      queries: harness.input(data.queries.flat(), 'float32x2', 2),
      sampleSmoothing: harness.input(data.smoothing, 'float32'),
      neighborhoods: harness.input(data.neighborhoods.flat(), 'uint32'),
      neighborhoodCapacity: 32,
      neighborCount: 32,
      kernel,
      epsilon: defaultDegreeReference.epsilon,
      output: {values: values.view, status: status.view}
    });
    expect(contributor.degree).toBe(defaultDegreeReference.degrees[kernel]);
    harness.graph.add(contributor);
    const cleanup = await harness.run();
    try {
      expect(Array.from(await status.read())).toEqual([0, 0, 0, 0]);
      const actual = await values.read();
      const expected = defaultDegreeReference.expected[kernel];
      for (let row = 0; row < 4; row++) {
        expect(actual[row * 4]).toBeCloseTo(expected[row], 3);
        expect(actual[row * 4 + 2]).toBe(32);
      }
    } finally {
      cleanup();
    }
  });
}

it('indexed Delaunay cavities reproduce CGAL6.0.1 Sibson coordinates across128 sites including sample/hull/outside queries', async () => {
  const harness = await createHarness();
  const data = reference.natural;
  const queryCount = data.queries.length;
  const values = harness.output(queryCount, 'float32x4', 4);
  const status = harness.output(queryCount, 'uint32');
  const weights = harness.output(queryCount * 64, 'float32x2', 2);
  harness.graph.add(
    new GPUNaturalNeighbor({
      positions: harness.input(data.positions.flat(), 'float32x2', 2),
      values: harness.input(data.values, 'float32'),
      queries: harness.input(data.queries.flat(), 'float32x2', 2),
      triangles: harness.input(data.triangles.flat(), 'uint32'),
      queryTriangles: harness.input(data.seeds, 'uint32'),
      maximumNeighborCount: 64,
      output: {values: values.view, status: status.view, weights: weights.view}
    })
  );
  const cleanup = await harness.run();
  try {
    const actual = await values.read();
    const statuses = await status.read();
    const coefficients = await weights.read();
    for (let query = 0; query < queryCount; query++) {
      const expected = data.expected[query];
      expect(statuses[query]).toBe(expected ? 0 : 1);
      if (!expected) {
        expect(Number.isNaN(actual[query * 4])).toBe(true);
        continue;
      }
      expect(actual[query * 4]).toBeCloseTo(expected.prediction, 3);
      let total = 0;
      const rows = new Map<number, number>();
      for (let slot = 0; slot < 64; slot++) {
        const offset = (query * 64 + slot) * 2;
        const weight = coefficients[offset + 1];
        expect(weight).toBeGreaterThanOrEqual(0);
        total += weight;
        if (coefficients[offset] >= 0) {
          rows.set(coefficients[offset], weight);
        }
      }
      expect(total).toBeCloseTo(1, 5);
      for (const [sample, weight] of expected.weights) {
        expect(rows.get(sample) ?? 0).toBeCloseTo(weight, 3);
      }
    }
  } finally {
    cleanup();
  }
});

it('shared GPU factor conditions40 observations and64 joint sites against direct GSTools', async () => {
  const harness = await createHarness();
  const data = reference.simulation;
  const values = harness.output(64 * 8, 'float32');
  const moments = harness.output(64, 'float32x2', 2);
  const status = harness.output(8, 'uint32');
  harness.graph.add(
    new GPUConditionalGaussianSimulation({
      positions: harness.input(data.positions.flat(), 'float32x2', 2),
      observations: harness.input(
        data.values.flatMap(value => [value, 0.07, 0, 1]),
        'float32x4',
        4
      ),
      queries: harness.input(data.queries.flat(), 'float32x2', 2),
      covariance,
      mean: 1.2,
      seed: 77,
      replicateCount: 8,
      output: {values: values.view, moments: moments.view, status: status.view}
    })
  );
  const cleanup = await harness.run();
  try {
    expect(Array.from(await status.read())).toEqual(Array(8).fill(0));
    const actual = await moments.read();
    for (let query = 0; query < 64; query++) {
      expect(actual[query * 2]).toBeCloseTo(data.mean[query], 3);
      expect(actual[query * 2 + 1]).toBeCloseTo(data.variance[query], 3);
    }
    expect(Array.from(await values.read()).every(Number.isFinite)).toBe(true);
  } finally {
    cleanup();
  }
});

for (const scenario of [
  {name: 'centering', observation: 3e38, mean: -3e38, sill: 1},
  {name: 'whitening', observation: 3e38, mean: 0, sill: 0.01}
]) {
  it(`conditional simulation reports ${scenario.name} overflow instead of successful nonfinite draws`, async () => {
    const harness = await createHarness();
    const values = harness.output(4, 'float32');
    const moments = harness.output(2, 'float32x2', 2);
    const status = harness.output(2, 'uint32');
    harness.graph.add(
      new GPUConditionalGaussianSimulation({
        positions: harness.input([0, 0], 'float32x2', 2),
        observations: harness.input([scenario.observation, 0, 0, 1], 'float32x4', 4),
        queries: harness.input([0, 0, 1, 0], 'float32x2', 2),
        covariance: {model: 'exponential', range: 1, sill: scenario.sill},
        mean: scenario.mean,
        seed: 11,
        replicateCount: 2,
        output: {values: values.view, moments: moments.view, status: status.view}
      })
    );
    const cleanup = await harness.run();
    try {
      expect(Array.from(await status.read())).toEqual([2, 2]);
      expect(Array.from(await values.read()).every(Number.isNaN)).toBe(true);
      expect(Array.from(await moments.read()).every(Number.isNaN)).toBe(true);
    } finally {
      cleanup();
    }
  });
}

it('conditioned ensemble matches GSTools marginal moments and gstat joint covariance', async () => {
  const harness = await createHarness();
  const count = 4096;
  const values = harness.output(count * 2, 'float32');
  const moments = harness.output(2, 'float32x2', 2);
  const status = harness.output(count, 'uint32');
  harness.graph.add(
    new GPUConditionalGaussianSimulation({
      positions: harness.input(reference.positions.flat(), 'float32x2', 2),
      observations: harness.input(
        reference.values.flatMap(value => [value, 0.07, 0, 1]),
        'float32x4',
        4
      ),
      queries: harness.input(reference.queries.slice(0, 2).flat(), 'float32x2', 2),
      covariance,
      mean: 1.2,
      seed: 191,
      replicateCount: count,
      output: {values: values.view, moments: moments.view, status: status.view}
    })
  );
  const cleanup = await harness.run();
  try {
    expect(Array.from(await status.read()).every(value => value === 0)).toBe(true);
    const draws = await values.read();
    const mean = [0, 1].map(
      component =>
        Array.from({length: count}, (_, row) => draws[row * 2 + component]).reduce(
          (sum, value) => sum + value,
          0
        ) / count
    );
    for (let component = 0; component < 2; component++) {
      expect(Math.abs(mean[component] - reference.simulation.ensembleMean[component])).toBeLessThan(
        0.13
      );
      for (let other = 0; other < 2; other++) {
        const covarianceValue =
          Array.from(
            {length: count},
            (_, row) =>
              (draws[row * 2 + component] - mean[component]) *
              (draws[row * 2 + other] - mean[other])
          ).reduce((sum, value) => sum + value, 0) /
          (count - 1);
        expect(
          Math.abs(covarianceValue - gstatReference.ensembleCovariance[component][other])
        ).toBeLessThan(0.15);
        if (component === other) {
          expect(
            Math.abs(covarianceValue - reference.simulation.ensembleCovariance[component][other])
          ).toBeLessThan(0.2);
        }
      }
    }
  } finally {
    cleanup();
  }
});

it('resident variogram fitting agrees with direct GSTools exact-model parameter fitting', async () => {
  const harness = await createHarness();
  const data = reference.fit;
  const parameters = harness.output(1, 'float32x4', 4);
  const diagnostics = harness.output(1, 'float32x4', 4);
  const status = harness.output(1, 'uint32');
  harness.graph.add(
    new GPUVariogramModelFit({
      distances: harness.input(data.lags, 'float32'),
      semivariances: harness.input(data.semivariances, 'float32'),
      pairCounts: harness.input(
        data.lags.map(() => 20),
        'uint32'
      ),
      models: ['exponential'],
      weighting: 'none',
      output: {parameters: parameters.view, diagnostics: diagnostics.view, status: status.view}
    })
  );
  const cleanup = await harness.run();
  try {
    expect(Array.from(await status.read())).toEqual([0]);
    const actual = await parameters.read();
    for (let parameter = 0; parameter < 3; parameter++) {
      expect(actual[parameter]).toBeCloseTo(data.parameters[parameter], 3);
    }
  } finally {
    cleanup();
  }
});

it('weighted measurement-error block prediction matches direct gstat irregular-block quadrature', async () => {
  const harness = await createHarness();
  const data = reference.lmc;
  const values = harness.output(1, 'float32x4', 4);
  const status = harness.output(1, 'uint32');
  harness.graph.add(
    new GPULocalKriging({
      positions: harness.input(reference.positions.flat(), 'float32x2', 2),
      observations: harness.input(
        reference.values.flatMap(value => [value, 0.07, 0, 1]),
        'float32x4',
        4
      ),
      supports: harness.input(
        data.supports.flatMap((point, index) => [...point, data.supportWeights[index], 0]),
        'float32x4',
        4
      ),
      supportCount: 3,
      mode: 'ordinary',
      covariance,
      neighborCount: 6,
      output: {values: values.view, status: status.view}
    })
  );
  const cleanup = await harness.run();
  try {
    expect(Array.from(await status.read())).toEqual([0]);
    const actual = await values.read();
    expect(actual[0]).toBeCloseTo(gstatReference.block[0], 4);
    expect(actual[1]).toBeCloseTo(gstatReference.block[1], 4);
  } finally {
    cleanup();
  }
});

it('resident grid neighbor search feeds CSR RBF directly without CPU repacking or readback', async () => {
  const {GPUNeighborSearch, getGPUNeighborSearchParameterValues} = await import(
    '../../../src/gpu-spatial-analysis/neighbor-search/index'
  );
  const harness = await createHarness();
  const data = reference.radial;
  const positions = harness.input(data.positions.flat(), 'float32x2', 2);
  const queries = harness.input(data.queries.flat(), 'float32x2', 2);
  const offsets = harness.output(5, 'uint32');
  const neighbors = harness.output(128, 'uint32');
  const values = harness.output(4, 'float32x4', 4);
  const status = harness.output(4, 'uint32');
  const overflow = harness.output(1, 'uint32');
  harness.graph.add(
    new GPUNeighborSearch({
      positions,
      queryPositions: queries,
      parameters: harness.input(
        Array.from(getGPUNeighborSearchParameterValues({bounds: [-2, -2, 2, 2]})),
        'float32'
      ),
      mode: 'knn',
      k: 32,
      gridSize: [8, 8],
      output: {
        offsets: offsets.view,
        neighbors: neighbors.view,
        weights: harness.output(128, 'float32').view,
        status: {overflow: overflow.view}
      }
    })
  );
  harness.graph.add(
    new GPULocalRBF({
      positions,
      values: harness.input(data.values, 'float32'),
      queries,
      neighborhoods: neighbors.view,
      neighborhoodOffsets: offsets.view,
      neighborCount: 32,
      kernel: 'thin-plate-spline',
      degree: 2,
      epsilon: 0.9,
      sampleSmoothing: harness.input(data.smoothing, 'float32'),
      output: {values: values.view, status: status.view}
    })
  );
  const cleanup = await harness.run();
  try {
    expect(Array.from(await overflow.read())).toEqual([0]);
    expect(Array.from(await status.read())).toEqual([0, 0, 0, 0]);
    const actual = await values.read();
    for (let row = 0; row < 4; row++) {
      expect(actual[row * 4]).toBeCloseTo(data.expected['thin-plate-spline'][row], 3);
    }
  } finally {
    cleanup();
  }
});

it('weighted support rejects negative/zero weights and indexed cavities report capacity exhaustion', async () => {
  const harness = await createHarness();
  const predictions = harness.output(2, 'float32x4', 4);
  const statuses = harness.output(2, 'uint32');
  harness.graph.add(
    new GPULocalKriging({
      positions: harness.input(reference.positions.flat(), 'float32x2', 2),
      observations: harness.input(
        reference.values.flatMap(value => [value, 0.07, 0, 1]),
        'float32x4',
        4
      ),
      supports: harness.input([0, 0, -1, 0, 0, 0, 0, 0], 'float32x4', 4),
      mode: 'ordinary',
      covariance,
      output: {values: predictions.view, status: statuses.view}
    })
  );
  const natural = reference.natural;
  const values = harness.output(1, 'float32x4', 4);
  const status = harness.output(1, 'uint32');
  harness.graph.add(
    new GPUNaturalNeighbor({
      positions: harness.input(natural.positions.flat(), 'float32x2', 2),
      values: harness.input(natural.values, 'float32'),
      queries: harness.input(natural.queries[0], 'float32x2', 2),
      triangles: harness.input(natural.triangles.flat(), 'uint32'),
      queryTriangles: harness.input([natural.seeds[0]], 'uint32'),
      maximumNeighborCount: 3,
      output: {values: values.view, status: status.view}
    })
  );
  const cleanup = await harness.run();
  try {
    expect(Array.from(await statuses.read())).toEqual([1, 1]);
    expect(Array.from(await status.read())).toEqual([3]);
  } finally {
    cleanup();
  }
});
