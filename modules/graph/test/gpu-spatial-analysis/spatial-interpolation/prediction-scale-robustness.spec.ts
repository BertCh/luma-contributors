// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import type {GPUVectorFormat} from '@luma-contributors/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPULocalKriging} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {
  GPULocalRBF,
  type GPULocalRBFProps
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-rbf';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

/** Deterministic integer jitter so every coordinate stays exactly representable in float32. */
function getLattice(
  size: number,
  spacing: number,
  jitter: number
): {positions: number[][]; values: number[]} {
  const positions: number[][] = [];
  const values: number[] = [];
  let state = 12345;
  const next = () => {
    state = (state * 1103515245 + 12345) % 2147483648;
    return state / 2147483648;
  };
  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) {
      const x = column * spacing + Math.floor(next() * jitter);
      const y = row * spacing + Math.floor(next() * jitter);
      positions.push([x, y]);
      values.push(Math.sin(x / (3 * spacing)) + 0.5 * Math.cos(y / (2 * spacing)) + 2);
    }
  }
  return {positions, values};
}

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

const LATTICE = getLattice(7, 100, 40);
const QUERIES = [
  [210, 190],
  [330, 410],
  [455, 120],
  [150, 520]
];

/** Runs local kriging with every coordinate translated by origin and the covariance sill set. */
async function runLocalKriging(
  mode: 'ordinary' | 'universal',
  origin: number,
  sill: number,
  coordinateScale: number
): Promise<{prediction: number[]; variance: number[]; status: number[]}> {
  const harness = await createHarness();
  const values = harness.output(QUERIES.length, 'float32x4', 4);
  const status = harness.output(QUERIES.length, 'uint32');
  const shift = (point: number[]) => point.map(coordinate => origin + coordinate * coordinateScale);
  harness.graph.add(
    new GPULocalKriging({
      positions: harness.input(LATTICE.positions.flatMap(shift), 'float32x2', 2),
      observations: harness.input(
        LATTICE.values.flatMap(value => [value, 0, 0, 1]),
        'float32x4',
        4
      ),
      supports: harness.input(QUERIES.flatMap(shift), 'float32x2', 2),
      covariance: {model: 'exponential', range: 250 * coordinateScale, sill},
      mode,
      neighborCount: 12,
      output: {values: values.view, status: status.view}
    })
  );
  const cleanup = await harness.run();
  try {
    const read = await values.read();
    return {
      prediction: QUERIES.map((_, query) => read[query * 4]),
      variance: QUERIES.map((_, query) => read[query * 4 + 1]),
      status: Array.from(await status.read())
    };
  } finally {
    cleanup();
  }
}

for (const mode of ['ordinary', 'universal'] as const) {
  it(`local ${mode} kriging is invariant to sill and projected-coordinate origin`, async () => {
    const baseline = await runLocalKriging(mode, 0, 1, 1);
    expect(baseline.status).toEqual([0, 0, 0, 0]);
    // Sills from 1e-4 to 1e6 at coordinates around 1e6 m (exact integers in float32).
    for (const sill of [1e-4, 1e4, 1e6]) {
      const scaled = await runLocalKriging(mode, 900000, sill, 1);
      expect(scaled.status, `sill ${sill}`).toEqual([0, 0, 0, 0]);
      for (let query = 0; query < QUERIES.length; query++) {
        expect(scaled.prediction[query]).toBeCloseTo(baseline.prediction[query], 2);
        // Latent variance scales linearly with the sill.
        const expected = baseline.variance[query] * sill;
        expect(Math.abs(scaled.variance[query] - expected)).toBeLessThan(
          0.05 * Math.max(expected, 1e-3 * sill)
        );
      }
    }
  });
}

async function runLocalRBF(
  kernel: NonNullable<GPULocalRBFProps['kernel']>,
  degree: 1 | 2,
  origin: number,
  coordinateScale: number
): Promise<{prediction: number[]; status: number[]}> {
  const harness = await createHarness();
  const values = harness.output(QUERIES.length, 'float32x4', 4);
  const status = harness.output(QUERIES.length, 'uint32');
  const shift = (point: number[]) => point.map(coordinate => origin + coordinate * coordinateScale);
  harness.graph.add(
    new GPULocalRBF({
      positions: harness.input(LATTICE.positions.flatMap(shift), 'float32x2', 2),
      values: harness.input(LATTICE.values, 'float32'),
      queries: harness.input(QUERIES.flatMap(shift), 'float32x2', 2),
      neighborCount: 16,
      kernel,
      degree,
      output: {values: values.view, status: status.view}
    })
  );
  const cleanup = await harness.run();
  try {
    const read = await values.read();
    return {
      prediction: QUERIES.map((_, query) => read[query * 4]),
      status: Array.from(await status.read())
    };
  } finally {
    cleanup();
  }
}

for (const [kernel, degree] of [
  ['cubic', 1],
  ['thin-plate-spline', 1],
  ['quintic', 2]
] as const) {
  it(`local ${kernel} RBF is invariant to coordinate magnitude (1e6 m origin, 1e3 scale)`, async () => {
    const baseline = await runLocalRBF(kernel, degree, 0, 1);
    expect(baseline.status).toEqual([0, 0, 0, 0]);
    const scaled = await runLocalRBF(kernel, degree, 1000000, 1000);
    expect(scaled.status).toEqual([0, 0, 0, 0]);
    for (let query = 0; query < QUERIES.length; query++) {
      expect(scaled.prediction[query]).toBeCloseTo(baseline.prediction[query], 2);
    }
  });
}

it('local kriging with supplied candidate neighborhoods matches the full sample scan', async () => {
  const sampleCount = LATTICE.positions.length;
  async function run(candidates: 'scan' | 'fixed' | 'csr') {
    const harness = await createHarness();
    const values = harness.output(QUERIES.length, 'float32x4', 4);
    const status = harness.output(QUERIES.length, 'uint32');
    const everySample = Array.from({length: sampleCount}, (_, sample) => sample);
    // Pad each fixed row with 0xffffffff; CSR rows list the samples in reverse order.
    const fixed = QUERIES.flatMap(() => [...everySample, 0xffffffff, 0xffffffff]);
    const csr = QUERIES.flatMap(() => [...everySample].reverse());
    harness.graph.add(
      new GPULocalKriging({
        positions: harness.input(LATTICE.positions.flat(), 'float32x2', 2),
        observations: harness.input(
          LATTICE.values.flatMap(value => [value, 0, 0, 1]),
          'float32x4',
          4
        ),
        supports: harness.input(QUERIES.flat(), 'float32x2', 2),
        covariance: {model: 'exponential', range: 250, sill: 1},
        mode: 'ordinary',
        neighborCount: 12,
        ...(candidates === 'fixed'
          ? {
              neighborhoods: harness.input(fixed, 'uint32'),
              neighborhoodCapacity: sampleCount + 2
            }
          : candidates === 'csr'
            ? {
                neighborhoods: harness.input(csr, 'uint32'),
                neighborhoodOffsets: harness.input(
                  QUERIES.map((_, query) => query * sampleCount).concat(
                    QUERIES.length * sampleCount
                  ),
                  'uint32'
                )
              }
            : {}),
        output: {values: values.view, status: status.view}
      })
    );
    const cleanup = await harness.run();
    try {
      return {values: Array.from(await values.read()), status: Array.from(await status.read())};
    } finally {
      cleanup();
    }
  }
  const scan = await run('scan');
  expect(scan.status).toEqual([0, 0, 0, 0]);
  for (const candidates of ['fixed', 'csr'] as const) {
    const indexed = await run(candidates);
    expect(indexed.status).toEqual(scan.status);
    indexed.values.forEach((value, index) => expect(value).toBeCloseTo(scan.values[index], 4));
  }
});
