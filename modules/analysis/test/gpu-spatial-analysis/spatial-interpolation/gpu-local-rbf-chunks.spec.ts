// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPULocalRBF} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-rbf';
import {createVectorView} from '../../utils/gpu-contributor-test-utils';
import {createPaddedQueryFixture} from '../linear-referencing/gpu-linear-referencing-chunks.test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

async function runPrediction(
  device: Device,
  chunked: boolean,
  candidates: 'none' | 'fixed' | 'csr',
  sampleSmoothing = false,
  chunkLengths = [1, 3, 2]
) {
  const fixture = createPaddedQueryFixture(device);
  const {graph, input, output} = fixture;
  const positions = input('positions', 'float32x2', new Float32Array([0, 0, 1, 0, 0, 1]));
  const values = input('values', 'float32', new Float32Array([1, 2, 3]));
  const queryPoints = [
    [0, 0],
    [1, 0],
    [0, 1],
    [0.25, 0.25],
    [0.5, 0],
    [0, 0.5]
  ];
  const total = chunkLengths.reduce((sum, length) => sum + length, 0);
  const lengths = chunked ? chunkLengths : [total];
  const queries: GraphDataView<'float32x2'>[] = [];
  const results: GraphDataView<'float32x4'>[] = [];
  const statuses: GraphDataView<'uint32'>[] = [];
  let rowBase = 0;
  for (const [chunkIndex, length] of lengths.entries()) {
    queries.push(
      input(
        `queries-${chunkIndex}`,
        'float32x2',
        new Float32Array(Array.from({length}, (_, row) => queryPoints[(rowBase + row) % 6]).flat())
      )
    );
    results.push(output(`results-${chunkIndex}`, 'float32x4', length));
    statuses.push(output(`statuses-${chunkIndex}`, 'uint32', length));
    rowBase += length;
  }
  const candidateRows = new Uint32Array(
    Array.from({length: total}, (_, row) => (row % 6 < 3 ? row % 6 : 0xffffffff))
  );
  graph.add(
    new GPULocalRBF({
      positions,
      values,
      queries: chunked ? createVectorView('queries', 'float32x2', queries) : queries[0],
      output: {
        values: chunked ? createVectorView('results', 'float32x4', results) : results[0],
        status: chunked ? createVectorView('statuses', 'uint32', statuses) : statuses[0]
      },
      neighborCount: 3,
      neighborhoods:
        candidates === 'none' ? undefined : input('neighborhoods', 'uint32', candidateRows),
      neighborhoodOffsets:
        candidates === 'csr'
          ? input(
              'offsets',
              'uint32',
              new Uint32Array(Array.from({length: total + 1}, (_, row) => row))
            )
          : undefined,
      sampleSmoothing: sampleSmoothing
        ? input('smoothing', 'float32', new Float32Array(3))
        : undefined,
      neighborhoodCapacity: candidates === 'fixed' ? 1 : undefined,
      kernel: candidates === 'none' ? 'thin-plate-spline' : 'gaussian',
      degree: candidates === 'none' ? 1 : -1
    })
  );
  const compiled = graph.compile();
  try {
    return await fixture.submitTwice(compiled, async () => ({
      values: await fixture.readRows(results),
      status: await fixture.readRows(statuses)
    }));
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
}

it('matches packed results exactly for three uneven unaligned chunks and reproduces an affine plane', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const packed = await runPrediction(device, false, 'none');
  const chunked = await runPrediction(device, true, 'none');
  expect(chunked).toEqual(packed);
  expect(chunked.status).toEqual([0, 0, 0, 0, 0, 0]);
  // The independent plane is f(x,y) = 1 + x + 2y.
  for (const [index, expected] of [1, 2, 3, 1.75, 1.5, 2].entries())
    expect(chunked.values[index * 4]).toBeCloseTo(expected, 5);
});

for (const candidates of ['fixed', 'csr'] as const) {
  it(`keeps globally query-major ${candidates} candidate rows resident across chunks`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const packed = await runPrediction(device, false, candidates);
    const chunked = await runPrediction(device, true, candidates);
    expect(chunked).toEqual(packed);
    expect(chunked.status).toEqual([0, 0, 0, 1, 1, 1]);
    expect(chunked.values.filter((_, index) => index % 4 === 0).slice(0, 3)).toEqual([1, 2, 3]);
  });
}

it('keeps packed parity with CSR candidates and per-sample smoothing on an eight-binding device', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const packed = await runPrediction(device, false, 'csr', true);
  const chunked = await runPrediction(device, true, 'csr', true);
  expect(chunked).toEqual(packed);
  expect(chunked.status).toEqual([0, 0, 0, 1, 1, 1]);
  expect(chunked.values.filter((_, index) => index % 4 === 0).slice(0, 3)).toEqual([1, 2, 3]);
});

it('preserves distinct port offsets and padding across 31/32/33-row dispatches', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lengths = [31, 32, 33];
  const packed = await runPrediction(device, false, 'none', false, lengths);
  const chunked = await runPrediction(device, true, 'none', false, lengths);
  expect(chunked).toEqual(packed);
  expect(chunked.status).toEqual(Array(96).fill(0));
  for (let row = 0; row < 96; row++)
    expect(chunked.values[row * 4]).toBeCloseTo([1, 2, 3, 1.75, 1.5, 2][row % 6], 5);
});
