// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getPhilox4x32,
  getPhiloxOpenUnitFloat,
  PHILOX_PURPOSES
} from '../../../src/analysis-core/kernels/philox-random';
import {GPUConditionalGaussianSimulation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// Math correctness review, interpolation-density (open-interval uniform). The Box-Muller radius
// is `sqrt(-2 log(u))` with `u` uniform in the OPEN interval (0, 1). The kernels formed
// `u = (f32(word >> 8) + 0.5) / 2^24`, which rounds to exactly 1.0 when the top 24 bits of the
// word are all set, so that draw was exactly 0: the simulated value was the conditional mean,
// bit for bit. Seed 632507 is the first seed whose Philox word for (replicate 0, component 0)
// has those bits set (found by a scan of the CPU mirror `getPhilox4x32`).

const SEED = 632507;
const SILL = 1e4;

/** Standard normal draw of (replicate 0, component 0) from the Box-Muller definition. */
function getExpectedNormal(): number {
  const words = getPhilox4x32(
    [0, 0, 0, 0],
    [SEED, PHILOX_PURPOSES.conditionalGaussianSimulation.id]
  );
  // The premise of this test: the 24-bit uniform of this word is exactly 1 in float32.
  expect(words[0] >>> 8).toBe(0xffffff);
  expect(Math.fround((words[0] >>> 8) + 0.5) / 16777216).toBe(1);
  const radius = Math.sqrt(-2 * Math.log(getPhiloxOpenUnitFloat(words[0])));
  const angle = (2 * Math.PI * ((words[1] >>> 8) + 0.5)) / 16777216;
  return radius * Math.cos(angle);
}

async function runSimulation(queryCount: number) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // One observation of value 0 far beyond the range: the conditional moments of every query are
  // the prior ones (mean 0, variance SILL) to float32, and the queries are far from each other.
  const positionsBuffer = createInputBuffer(device, Float32Array.from([-1e6, -1e6]));
  const observationsBuffer = createInputBuffer(device, Float32Array.from([0, 0, 0, 1]));
  const queriesBuffer = createInputBuffer(
    device,
    Float32Array.from({length: queryCount * 2}, (_, component) =>
      component % 2 === 0 ? Math.floor(component / 2) * 1000 : 0
    )
  );
  const valuesBuffer = createOutputBuffer(device, queryCount);
  const momentsBuffer = createOutputBuffer(device, queryCount * 2);
  const statusBuffer = createOutputBuffer(device, 1);
  const graph = new GPUCommandGraph(device, {id: `simulation-open-uniform-${queryCount}`});
  const contributor = new GPUConditionalGaussianSimulation({
    positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', 1),
    observations: importGraphBuffer(graph, 'observations', observationsBuffer, 'float32x4', 1),
    queries: importGraphBuffer(graph, 'queries', queriesBuffer, 'float32x2', queryCount),
    covariance: {model: 'exponential', range: 10, sill: SILL},
    mean: 0,
    seed: SEED,
    replicateCount: 1,
    output: {
      values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', queryCount),
      moments: importGraphBuffer(graph, 'moments', momentsBuffer, 'float32x2', queryCount),
      status: importGraphBuffer(graph, 'status', statusBuffer, 'uint32', 1)
    }
  });
  graph.add(contributor);
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    values: await readFloat32(valuesBuffer, queryCount),
    moments: await readFloat32(momentsBuffer, queryCount * 2),
    status: await readUint32(statusBuffer, 1)
  };
  compiled.destroy();
  for (const buffer of [
    positionsBuffer,
    observationsBuffer,
    queriesBuffer,
    valuesBuffer,
    momentsBuffer,
    statusBuffer
  ]) {
    buffer.destroy();
  }
  return result;
}

for (const {path, queryCount} of [
  {path: 'cooperative', queryCount: 1},
  {path: 'blocked', queryCount: 129}
]) {
  it(`conditional simulation (${path}) draws from the open unit interval for the largest Philox word`, async () => {
    const expectedNormal = getExpectedNormal();
    // sqrt(-2 log(1 - 2^-24)) = 3.45e-4, times the cosine of the angle draw.
    expect(Math.abs(expectedNormal)).toBeGreaterThan(1e-4);
    expect(Math.abs(expectedNormal)).toBeLessThan(3.5e-4);
    const {values, moments, status} = await runSimulation(queryCount);
    expect(status).toEqual([0]);
    expect(moments[0]).toBeCloseTo(0, 6);
    expect(moments[1]).toBeCloseTo(SILL, 0);
    // Component 0 of the conditional square root is its own standard deviation.
    const draw = (values[0] - moments[0]) / Math.sqrt(moments[1]);
    expect(Math.abs(draw - expectedNormal)).toBeLessThan(1e-3 * Math.abs(expectedNormal));
  });
}
