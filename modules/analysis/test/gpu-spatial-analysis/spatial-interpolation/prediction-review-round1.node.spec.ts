// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {validateWeightsLossSources} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights-loss';
import {getGPUInverseDistanceWeightingParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/spatial-interpolation-parameters';
import {getGPUKrigingParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/kriging-parameters';
import {
  getGPULocalKrigingParameterValues,
  getGPUSpaceTimeKrigingParameterValues,
  type GPUPredictionCovariance
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/prediction-covariance';
import {
  GPULocalKriging,
  decodeGPULocalKrigingStatus
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {
  GPULocalRBF,
  decodeGPULocalRBFStatus
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-rbf';
import {GPUGaussianRandomField} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-gaussian-random-field';
import {GPUConditionalGaussianSimulation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation';

const covariance = {model: 'spherical', range: 1, sill: 1} as const;
const grid = makeGPURasterMetadata({width: 2, height: 1});
function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const positions = createTransientView(graph, 'positions', 'float32x2', 3);
  const observations = createTransientView(graph, 'observations', 'float32x4', 3);
  const queries = createTransientView(graph, 'queries', 'float32x2', 2);
  const output = {
    values: createTransientView(graph, 'output', 'float32x4', 2),
    status: createTransientView(graph, 'status', 'uint32', 2)
  };
  const kriging = {
    id: 'review-kriging',
    positions,
    observations,
    supports: queries,
    covariance,
    mode: 'ordinary' as const,
    neighborCount: 2,
    output
  };
  const rbf = {
    id: 'review-rbf',
    positions,
    queries,
    values: createTransientView(graph, 'data', 'float32', 3),
    neighborCount: 2,
    kernel: 'gaussian' as const,
    degree: 0 as const,
    output
  };
  return {device, graph, positions, observations, queries, output, kriging, rbf};
}

it('B1 preserves raw IDW nodata radius and power cases while rejecting encoding overflow', () => {
  for (const settings of [
    {searchRadius: -1, power: 2},
    {searchRadius: NaN, power: 2},
    {searchRadius: 2, power: -1}
  ]) {
    const result = getGPUInverseDistanceWeightingParameterValues(settings, grid);
    expect(Object.is(result[4], settings.searchRadius)).toBe(true);
    expect(result[5]).toBe(settings.power);
  }
  for (const settings of [{searchRadius: 1e50}, {searchRadius: 2, power: 1e50}]) {
    const target = new Float32Array(11).fill(77);
    expect(() => getGPUInverseDistanceWeightingParameterValues(settings, grid, target)).toThrow(
      /float32/
    );
    expect(Array.from(target)).toEqual(Array(11).fill(77));
  }
});

for (const route of [
  'kriging-csr',
  'rbf-csr',
  'kriging-legacy',
  'rbf-legacy',
  'conditioning'
] as const) {
  it(`B2 rejects complete-only producer polarity through ${route} snapshots`, () => {
    const fixture = createFixture();
    const {device, graph} = fixture;
    try {
      const status = {complete: createTransientView(graph, 'complete', 'uint32', 1)};
      // A described resident CSR: producer completeness must not disappear during capture.
      const weights = {
        offsets: createTransientView(graph, 'offsets', 'uint32', 3),
        neighbors: createTransientView(graph, 'neighbors', 'uint32', 4),
        status
      };
      expect(() => validateWeightsLossSources('direct', [weights])).toThrow(
        /complete.*candidateOverflow/
      );
      const candidates = route.endsWith('csr')
        ? {neighborhoodWeights: weights}
        : {
            neighborhoods: weights.neighbors,
            neighborhoodOffsets: weights.offsets,
            neighborhoodStatus: status
          };
      const construct = () =>
        route.startsWith('kriging')
          ? new GPULocalKriging({...fixture.kriging, ...candidates})
          : route.startsWith('rbf')
            ? new GPULocalRBF({...fixture.rbf, ...candidates})
            : new GPUGaussianRandomField({
                id: 'review-field',
                positions: fixture.positions,
                parameters: createTransientView(graph, 'params', 'float32', 16),
                conditioning: {
                  indices: createTransientView(graph, 'indices', 'uint32', 1),
                  values: createTransientView(graph, 'conditioning-data', 'float32', 1),
                  neighborhoods: createTransientView(graph, 'conditioning-neighbors', 'uint32', 3),
                  neighborCount: 1,
                  status
                },
                output: {
                  values: createTransientView(graph, 'field-values', 'float32', 3),
                  validity: createTransientView(graph, 'field-validity', 'uint32', 3),
                  status: {
                    approximation: createTransientView(graph, 'approximation', 'uint32', 1),
                    candidateOverflow: createTransientView(graph, 'candidate-overflow', 'uint32', 1)
                  },
                  clippedEigenvalueMass: createTransientView(graph, 'mass', 'float32', 1)
                }
              });
      expect(construct).toThrow(/complete.*candidateOverflow/);
    } finally {
      device.destroy();
    }
  });
}

it('B3 rejects vectors on RBF packed-only sample values at construction', () => {
  const fixture = createFixture();
  const {device, graph} = fixture;
  try {
    const values = createVectorView('bad-sample-vector', 'float32', [
      createTransientView(graph, 'sample-0', 'float32', 1),
      createTransientView(graph, 'sample-1', 'float32', 2)
    ]);
    const queries = createVectorView('queries-vector', 'float32x2', [
      createTransientView(graph, 'query-0', 'float32x2', 1),
      createTransientView(graph, 'query-1', 'float32x2', 1)
    ]);
    const output = {
      values: createVectorView('output-vector', 'float32x4', [
        createTransientView(graph, 'out-0', 'float32x4', 1),
        createTransientView(graph, 'out-1', 'float32x4', 1)
      ]),
      status: createVectorView('status-vector', 'uint32', [
        createTransientView(graph, 'status-0', 'uint32', 1),
        createTransientView(graph, 'status-1', 'uint32', 1)
      ])
    };
    // @ts-expect-error Sample values are a single packed view, even when queries are vectors.
    expect(() => new GPULocalRBF({...fixture.rbf, values, queries, output})).toThrow(
      /review-rbf.*values.*data graph view/
    );
  } finally {
    device.destroy();
  }
});

it('B3 names the actual kriging parameter format before route-specific checks', () => {
  const fixture = createFixture();
  try {
    const parameters = createTransientView(fixture.graph, 'params', 'sint32', 8);
    // @ts-expect-error Deliberately invalid fixed parameter format.
    expect(() => new GPULocalKriging({...fixture.kriging, parameters})).toThrow(
      /review-kriging.*parameters.*float32.*sint32/
    );
  } finally {
    fixture.device.destroy();
  }
});

for (const kind of ['gneiting', 'metric', 'sum-metric'] as const) {
  it(`B4 rejects unusable encoded ${kind} coefficients without publishing`, () => {
    const unit = {...covariance, nugget: 0};
    const settings = {
      searchAnisotropy: 1,
      covariance:
        kind === 'gneiting'
          ? {
              type: kind,
              sill: 1,
              nugget: 0,
              spatialRange: 1,
              temporalRange: 1,
              alpha: 1,
              gamma: 1,
              beta: 0.01,
              tau: 0.02
            }
          : kind === 'metric'
            ? {type: kind, joint: unit, anisotropy: 1e-50}
            : {type: kind, joint: unit, spatial: unit, temporal: unit, anisotropy: 1e-50}
    };
    const arena = new Float32Array(32).fill(77);
    expect(() => getGPUSpaceTimeKrigingParameterValues(settings, arena.subarray(2))).toThrow(
      kind === 'gneiting' ? /Gneiting.*temporal.*float32/ : /covariance.anisotropy.*float32/
    );
    expect(Array.from(arena)).toEqual(Array(32).fill(77));
  });
}

it('B4 legal Gneiting encoding retains exp(-3) at one temporal range', () => {
  const packed = getGPUSpaceTimeKrigingParameterValues({
    searchAnisotropy: 1,
    covariance: {
      type: 'gneiting',
      sill: 1,
      nugget: 0,
      spatialRange: 1,
      temporalRange: 1,
      alpha: 1,
      gamma: 1,
      beta: 0.5,
      tau: 1
    }
  });
  // At spatial lag 0 and temporal lag 1, (1 + a)^(-tau) = exp(-3).
  const correlation = (1 + packed[18]) ** -packed[23];
  expect(Number.isFinite(correlation)).toBe(true);
  expect(correlation).toBeCloseTo(0.049787068367863944, 7);
  expect(packed[19]).toBe(3);
});

it('B5 distinguishes graph insertion failure after retained node generation', () => {
  const fixture = createFixture();
  const {device, graph} = fixture;
  const producer = new GPUConditionalGaussianSimulation({
    id: 'lif',
    positions: fixture.positions,
    observations: fixture.observations,
    queries: fixture.queries,
    covariance,
    seed: 1,
    replicateCount: 1,
    reusePreparation: true,
    output: {
      values: createTransientView(graph, 'draws', 'float32', 2),
      moments: createTransientView(graph, 'moments', 'float32x2', 2),
      status: createTransientView(graph, 'draw-status', 'uint32', 1)
    }
  });
  const existing = createTransientView(graph, 'existing', 'uint32', 1);
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'lif-prepare-conditioning',
      operation: 'Fixture',
      invocationCount: 1,
      bindings: [{name: 'existing', view: existing, type: 'u32', access: 'read_write'}],
      body: 'existing[existingOffset] = 0u;'
    })
  );
  const original = device.createBuffer.bind(device);
  const acquired: ReturnType<typeof original>[] = [];
  const allocation = vi.spyOn(device, 'createBuffer').mockImplementation(options => {
    const buffer = original(options);
    acquired.push(buffer);
    return buffer;
  });
  try {
    expect(() => graph.add(producer)).toThrow(/lif-prepare-conditioning/);
    expect(acquired.length).toBe(4);
    // The public graph API has no insertion-failure callback: the producer still owns them.
    expect(acquired.every(buffer => !buffer.destroyed)).toBe(true);
    expect(producer.preparationStatistics).toBeDefined();
    producer.destroy();
    expect(acquired.every(buffer => buffer.destroyed)).toBe(true);
    expect(producer.preparationStatistics).toBeUndefined();
  } finally {
    allocation.mockRestore();
    producer.destroy();
    device.destroy();
  }
});

for (const kind of ['total', 'model'] as const) {
  it(`S1 rejects raster kriging ${kind} before publishing`, () => {
    const variogram =
      kind === 'total'
        ? {...covariance, nugget: 2e38, sill: 2e38}
        : {...covariance, nugget: 0, model: 'constructor' as typeof covariance.model};
    const target = new Float32Array(14).fill(77);
    expect(() => getGPUKrigingParameterValues({searchRadius: 1, variogram}, grid, target)).toThrow(
      kind === 'total' ? /finite.*total|total.*finite/ : /variogram.model/
    );
    expect(Array.from(target)).toEqual(Array(14).fill(77));
  });
}

for (const field of ['range', 'sill', 'nugget'] as const) {
  it(`S2 rejects a coercible string ${field} in prediction covariance`, () => {
    const invalid = {
      ...covariance,
      nugget: 0.25,
      [field]: field === 'nugget' ? '0.25' : field === 'range' ? '2' : '1'
    } as unknown as GPUPredictionCovariance;
    const target = new Float32Array(10).fill(77);
    expect(() => getGPULocalKrigingParameterValues({covariance: invalid}, target)).toThrow(
      new RegExp(field)
    );
    expect(Array.from(target)).toEqual(Array(10).fill(77));
  });
}

it('S5 decodes one word without copying the full status arena and rejects wrong element types', () => {
  for (const decode of [decodeGPULocalKrigingStatus, decodeGPULocalRBFStatus]) {
    const buffer = new ArrayBuffer(1 + 4 * 4096);
    const bytes = new Uint8Array(buffer, 1, 4 * 4096);
    new DataView(buffer).setUint32(1 + 4 * 2048, 3, true);
    const copy = vi.spyOn(Uint8Array.prototype, 'slice');
    try {
      expect(decode(bytes, 2048).candidateOverflow).toBe(1);
      expect(copy.mock.calls.every(arguments_ => arguments_.length > 0)).toBe(true);
    } finally {
      copy.mockRestore();
    }
    // @ts-expect-error Float status storage is outside the documented readback types.
    expect(() => decode(Float32Array.of(0))).toThrow(/Uint32Array.*Uint8Array.*ArrayBuffer/);
  }
});

it('Earlier co-kriging fix accepts coregionalizationModels as its sole covariance source', () => {
  const fixture = createFixture();
  try {
    const {covariance: unused, ...base} = fixture.kriging;
    void unused;
    const coregionalizationModels = [{covariance, loadings: [1, 0, 0, 1]}];
    expect(Array.from(getGPULocalKrigingParameterValues({coregionalizationModels}))).toEqual([
      1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1
    ]);
    const contributor = new GPULocalKriging({
      ...base,
      mode: 'co-kriging',
      variableCount: 2,
      coregionalizationModels
    });
    expect(() => contributor.getCommandNodes(fixture.graph)).not.toThrow();
  } finally {
    fixture.device.destroy();
  }
});

it('S4 forwards unequal field CSR row lengths to the local solve', () => {
  const fixture = createFixture();
  const {device, graph} = fixture;
  try {
    const field = new GPUGaussianRandomField({
      id: 'csr-field',
      positions: fixture.positions,
      parameters: createTransientView(graph, 'params', 'float32', 16),
      modeCount: 1,
      conditioning: {
        indices: createTransientView(graph, 'indices', 'uint32', 2),
        values: createTransientView(graph, 'observed-values', 'float32', 2),
        neighborhoods: createTransientView(graph, 'neighbors', 'uint32', 6),
        neighborhoodOffsets: createTransientView(graph, 'offsets', 'uint32', 4),
        neighborCount: 2
      },
      output: {
        values: createTransientView(graph, 'field-values', 'float32', 3),
        validity: createTransientView(graph, 'validity', 'uint32', 3),
        status: {approximation: createTransientView(graph, 'approximation', 'uint32', 1)},
        clippedEigenvalueMass: createTransientView(graph, 'mass', 'float32', 1)
      }
    });
    expect(
      getGPUCommandNodeKernelSources(field.getCommandNodes(graph)).some(({source}) =>
        source.includes('neighborhoodOffsets[neighborhoodOffsetsOffset')
      )
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

it('S5 rejects float storage even when its bits encode success', () => {
  for (const decode of [decodeGPULocalKrigingStatus, decodeGPULocalRBFStatus]) {
    // @ts-expect-error The runtime boundary must enforce the declared element types.
    expect(() => decode(Float32Array.of(0))).toThrow(/Uint32Array.*Uint8Array.*ArrayBuffer/);
  }
});

it('B1 preserves nodata polarity when tiny negative IDW controls round to signed zero', () => {
  const radius = getGPUInverseDistanceWeightingParameterValues(
    {searchRadius: -Number.MIN_VALUE},
    grid
  )[4];
  const power = getGPUInverseDistanceWeightingParameterValues(
    {searchRadius: 1, power: -Number.MIN_VALUE},
    grid
  )[5];
  // A rounded -0 compares >= 0 in WGSL; retain an invalid control rather than enabling a hit/mean.
  expect(Number.isNaN(radius) || radius < 0).toBe(true);
  expect(Number.isNaN(power) || power < 0).toBe(true);
  expect(() =>
    getGPUInverseDistanceWeightingParameterValues({searchRadius: Number.MIN_VALUE}, grid)
  ).toThrow(/float32/);
});
