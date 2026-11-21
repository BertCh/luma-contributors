// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUSpatialEmpiricalBayesRates} from '../../../src/gpu-spatial-analysis/rate-smoothing/gpu-spatial-empirical-bayes-rates';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createRandomWeights} from '../spatial-weights/spatial-weights-algebra-oracle';
import {ESDA_SPATIAL_EB_LOW_DISPERSION} from './eb-low-dispersion-fixture';
import {ESDA_SPATIAL_EB_LATTICE} from './esda-spatial-eb-fixture';
import {computeSpatialEmpiricalBayesOracle} from './spatial-empirical-bayes-oracle';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createRateAuditInput,
  RATE_AUDIT_EXPECTED
} from './rate-smoothing-mathematical-audit-fixture';

const GARBAGE = 0x7f7f7f7f;

async function run(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  input: {
    events: Float32Array;
    populations: Float32Array;
    offsets: number[];
    neighbors: number[];
    /** Weight per slot; every slot has weight 1 when omitted. */
    weights?: number[];
    mask?: Uint32Array;
    overflow?: number;
    candidateOverflow?: number;
  }
) {
  const rows = input.events.length;
  const graph = new GPUCommandGraph(device, {id: 'spatial-eb'});
  const slots = Math.max(input.neighbors.length, 1);
  const neighbors = Uint32Array.from(input.neighbors.length ? input.neighbors : [0]);
  const buffers = [
    createInputBuffer(device, input.events),
    createInputBuffer(device, input.populations),
    createInputBuffer(device, Uint32Array.from(input.offsets)),
    createInputBuffer(device, neighbors),
    createInputBuffer(
      device,
      input.weights ? Float32Array.from(input.weights) : new Float32Array(slots).fill(1)
    ),
    input.mask && createInputBuffer(device, input.mask),
    input.overflow === undefined
      ? undefined
      : createInputBuffer(device, Uint32Array.of(input.overflow)),
    input.candidateOverflow === undefined
      ? undefined
      : createInputBuffer(device, Uint32Array.of(input.candidateOverflow))
  ];
  const outputs = {
    spatialRate: createOutputBuffer(device, rows),
    smoothed: createOutputBuffer(device, rows),
    priorMean: createOutputBuffer(device, rows),
    priorVariance: createOutputBuffer(device, rows)
  };
  for (const buffer of Object.values(outputs)) {
    buffer.write(new Uint32Array(buffer.byteLength / 4).fill(GARBAGE));
  }
  graph.add(
    new GPUSpatialEmpiricalBayesRates({
      events: importGraphBuffer(graph, 'events', buffers[0]!, 'float32', rows),
      populations: importGraphBuffer(graph, 'populations', buffers[1]!, 'float32', rows),
      weights: {
        offsets: importGraphBuffer(graph, 'offsets', buffers[2]!, 'uint32', rows + 1),
        neighbors: importGraphBuffer(graph, 'neighbors', buffers[3]!, 'uint32', slots),
        weights: importGraphBuffer(graph, 'weights', buffers[4]!, 'float32', slots),
        status: {
          overflow: buffers[6] && importGraphBuffer(graph, 'overflow', buffers[6], 'uint32', 1),
          candidateOverflow:
            buffers[7] && importGraphBuffer(graph, 'candidate-overflow', buffers[7], 'uint32', 1)
        }
      },
      mask: buffers[5] && importGraphBuffer(graph, 'mask', buffers[5], 'uint32', rows),
      output: {
        spatialRates: importGraphBuffer(graph, 'sr', outputs.spatialRate, 'float32', rows),
        smoothedRates: importGraphBuffer(graph, 'sm', outputs.smoothed, 'float32', rows),
        priorMeans: importGraphBuffer(graph, 'pm', outputs.priorMean, 'float32', rows),
        priorVariances: importGraphBuffer(graph, 'pv', outputs.priorVariance, 'float32', rows)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    spatialRate: await readFloat32(outputs.spatialRate, rows),
    smoothed: await readFloat32(outputs.smoothed, rows),
    priorMean: await readFloat32(outputs.priorMean, rows),
    priorVariance: await readFloat32(outputs.priorVariance, rows)
  };
  compiled.destroy();
  for (const buffer of [...buffers, ...Object.values(outputs)]) buffer?.destroy();
  return result;
}

function expectClose(actual: number[], expected: number[], label: string, relative = 2e-3): void {
  expect(actual).toHaveLength(expected.length);
  for (const [row, value] of expected.entries()) {
    if (Number.isNaN(value)) {
      expect(actual[row], `${label} row ${row}`).toBeNaN();
    } else {
      expect(
        Math.abs(actual[row] - value),
        `${label} row ${row}: ${actual[row]} vs ${value}`
      ).toBeLessThanOrEqual(1e-6 + relative * Math.abs(value));
    }
  }
}

it('GPUSpatialEmpiricalBayesRates matches the oracle on a hand-checked path graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Path 0-1-2-3 plus isolate 4.
  const events = Float32Array.from([10, 30, 20, 50, 7]);
  const populations = Float32Array.from([100, 200, 100, 400, 50]);
  const offsets = [0, 1, 3, 5, 6, 6];
  const neighbors = [1, 0, 2, 1, 3, 2];
  const result = await run(device, {events, populations, offsets, neighbors});
  // Row 1: members {1,0,2}: E = 60, B = 400, rate 0.15.
  expect(result.spatialRate[1]).toBeCloseTo(0.15, 6);
  // Isolate: own rate, variance clamps to zero, smoothed equals own rate.
  expect(result.spatialRate[4]).toBeCloseTo(0.14, 6);
  expect(result.smoothed[4]).toBeCloseTo(0.14, 6);
  expect(result.priorVariance[4]).toBe(0);
  const oracle = computeSpatialEmpiricalBayesOracle({events, populations, offsets, neighbors});
  expectClose(result.spatialRate, oracle.spatialRate, 'spatialRate');
  expectClose(result.smoothed, oracle.smoothed, 'smoothed');
  expectClose(result.priorMean, oracle.priorMean, 'priorMean');
  expectClose(result.priorVariance, oracle.priorVariance, 'priorVariance', 1e-2);
  expect(result.smoothed.every(value => Number.isFinite(value))).toBe(true);
});

it('GPUSpatialEmpiricalBayesRates matches NumPy moments on every port with masked, invalid and island rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scale of [1, 2]) {
    const actual = await run(device, createRateAuditInput(scale));
    for (const name of ['spatialRate', 'smoothed', 'priorMean', 'priorVariance'] as const) {
      const expected = RATE_AUDIT_EXPECTED.spatial[name].map(value =>
        value === null ? NaN : value
      );
      expectClose(
        actual[name].map(value => value * (name === 'priorVariance' ? scale * scale : scale)),
        expected,
        name,
        2e-5
      );
    }
  }
});

it('GPUSpatialEmpiricalBayesRates matches the oracle with random weights, masks and bad rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 600;
  let state = 7;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  const populations = Float32Array.from({length: rows}, () =>
    Math.floor(50 + random() ** 3 * 20000)
  );
  const events = Float32Array.from({length: rows}, (_, row) =>
    Math.floor(populations[row] * (0.01 + 0.03 * random()) * (row % 7 === 0 ? 3 : 1))
  );
  populations[5] = 0;
  events[9] = NaN;
  const mask = Uint32Array.from({length: rows}, (_, row) => (row % 11 === 3 ? 0 : 1));
  const csr = createRandomWeights(rows, 0.02, 21, true);
  const input = {events, populations, offsets: csr.offsets, neighbors: csr.neighbors, mask};
  const result = await run(device, input);
  const oracle = computeSpatialEmpiricalBayesOracle(input);
  expectClose(result.spatialRate, oracle.spatialRate, 'spatialRate');
  expectClose(result.smoothed, oracle.smoothed, 'smoothed');
  expectClose(result.priorMean, oracle.priorMean, 'priorMean');
  expectClose(result.priorVariance, oracle.priorVariance, 'priorVariance', 1e-2);
  expect(result.spatialRate[5]).toBeNaN();
  expect(result.smoothed[3]).toBeNaN();
  expect(result.smoothed.filter(value => Number.isFinite(value)).length).toBeGreaterThan(400);
});

it('GPUSpatialEmpiricalBayesRates reproduces esda 2.10 Spatial_Rate and Spatial_Empirical_Bayes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = ESDA_SPATIAL_EB_LATTICE;
  const result = await run(device, {
    events: Float32Array.from(fixture.events),
    populations: Float32Array.from(fixture.populations),
    offsets: [...fixture.offsets],
    neighbors: [...fixture.neighbors]
  });
  expectClose(result.spatialRate, [...fixture.spatialRate], 'esda spatialRate', 1e-5);
  expectClose(result.smoothed, [...fixture.smoothed], 'esda smoothed', 2e-4);
  expect(result.priorVariance.some(value => value > 0)).toBe(true);
});

it('GPUSpatialEmpiricalBayesRates clamps local prior variance like esda on low-dispersion data', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = ESDA_SPATIAL_EB_LOW_DISPERSION;
  const result = await run(device, {
    events: Float32Array.from(fixture.events),
    populations: Float32Array.from(fixture.populations),
    offsets: [...fixture.offsets],
    neighbors: [...fixture.neighbors]
  });
  // esda 2.10.0 Spatial_Empirical_Bayes sets r_var[r_var < 0] = 0; common-rate counts make that happen.
  expect(Math.min(...result.priorVariance)).toBe(0);
  for (const [row, expected] of fixture.smoothed.entries()) {
    expect(Math.abs(result.smoothed[row] - expected), `row ${row}`).toBeLessThan(
      1e-6 + 2e-4 * expected
    );
  }
});

// spatial-statistics-05: neighbors are the active slots of the shared rule (positive weight,
// another row). A slot stored with weight 0 used to be pooled into the neighborhood.
it('GPUSpatialEmpiricalBayesRates does not pool a zero-weight slot', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Row 0 lists row 1 (weight 1) and row 2 (weight 0); rows 1 and 2 list nothing.
  // Row 0: members {0, 1}: E = 1 + 3 = 4, B = 10 + 10 = 20, rate 0.2. Pooling row 2 as well gave
  // (1 + 3 + 100) / 30 = 3.4667.
  const result = await run(device, {
    events: Float32Array.from([1, 3, 100]),
    populations: Float32Array.from([10, 10, 10]),
    offsets: [0, 2, 2, 2],
    neighbors: [1, 2],
    weights: [1, 0]
  });
  expect(result.spatialRate[0]).toBeCloseTo(0.2, 6);
  expect(result.priorMean[0]).toBeCloseTo(0.2, 6);
  // s2 = (10 (0.1 - 0.2)^2 + 10 (0.3 - 0.2)^2) / 20 = 0.01; a = 0.01 - 0.2 / (20 / 2) = -0.01,
  // clamped to 0, so the smoothed rate is the prior mean.
  expect(result.priorVariance[0]).toBe(0);
  expect(result.smoothed[0]).toBeCloseTo(0.2, 6);
  expect(result.spatialRate[2]).toBeCloseTo(10, 5);
});

it('GPUSpatialEmpiricalBayesRates suppresses all four outputs when either weights loss flag is nonzero', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Eight rows, one linked pair and six islands. Complete neighborhoods pool
  // to [0.2,0.2,0.4,0.5,0.6,0.7,0.8,0.9]; their Poisson-corrected variances clamp to zero.
  const input = {
    events: Float32Array.from([1, 3, 4, 5, 6, 7, 8, 9]),
    populations: new Float32Array(8).fill(10),
    offsets: [0, 1, 2, 2, 2, 2, 2, 2, 2],
    neighbors: [1, 0]
  };
  const complete = await run(device, {...input, overflow: 0, candidateOverflow: 0});
  const expected = [0.2, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
  for (const name of ['spatialRate', 'smoothed', 'priorMean'] as const)
    expectClose(complete[name], expected, name, 1e-6);
  expect(complete.priorVariance).toEqual(new Array(8).fill(0));
  for (const flags of [
    {overflow: 1, candidateOverflow: 0},
    {overflow: 0, candidateOverflow: 1}
  ]) {
    const actual = await run(device, {...input, ...flags});
    for (const output of Object.values(actual)) expect(output.every(Number.isNaN)).toBe(true);
  }
});

it('UST-03 rejects a negative focus and excludes negative donors', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await run(device, {
    events: Float32Array.of(-10, 2, 2, 2),
    populations: Float32Array.of(10, 10, 10, 10),
    offsets: [0, 1, 3, 5, 6],
    neighbors: [1, 0, 2, 1, 3, 2]
  });
  expect(result.smoothed).toHaveLength(4);
  expect(result.smoothed[0]).toBeNaN();
  for (const value of result.smoothed.slice(1)) {
    expect(Number.isFinite(value)).toBe(true);
    expect(value).toBeCloseTo(0.2, 6);
  }
});
