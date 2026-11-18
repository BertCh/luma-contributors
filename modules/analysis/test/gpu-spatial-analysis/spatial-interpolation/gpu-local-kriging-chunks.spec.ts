// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPULocalKriging,
  createLocalKrigingForTesting
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createPaddedQueryFixture} from '../linear-referencing/gpu-linear-referencing-chunks.test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const QUERY_POINTS = [
  [0, 0],
  [0.2, 0.1],
  [0.5, 0.2],
  [0.1, 0.6],
  [0.25, 0.25],
  [0.6, 0.1]
];

for (const candidates of ['fixed', 'csr'] as const) {
  for (const chunked of [false, true]) {
    it(`breaks reversed ${candidates} candidate ties by source row (${chunked ? 'chunked' : 'packed'})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = createPaddedQueryFixture(device);
      const {graph, input, output} = fixture;
      const lengths = chunked ? [1, 3, 2] : [6];
      const supports = lengths.map((length, index) =>
        input(`supports-${index}`, 'float32x2', new Float32Array(length * 2))
      );
      const values = lengths.map((length, index) => output(`values-${index}`, 'float32x4', length));
      const statuses = lengths.map((length, index) =>
        output(`statuses-${index}`, 'uint32', length)
      );
      graph.add(
        new GPULocalKriging({
          positions: input('positions', 'float32x2', new Float32Array([-1, 0, 1, 0])),
          observations: input(
            'observations',
            'float32x4',
            new Float32Array([2, 0, 0, 1, 10, 0, 0, 1])
          ),
          supports: chunked ? createVectorView('supports', 'float32x2', supports) : supports[0],
          mode: 'ordinary',
          neighborCount: 1,
          neighborhoodCapacity: 2,
          neighborhoods: input(
            'neighbors',
            'uint32',
            new Uint32Array(Array.from({length: 6}, () => [1, 0]).flat())
          ),
          neighborhoodOffsets:
            candidates === 'csr'
              ? input('offsets', 'uint32', new Uint32Array([0, 2, 4, 6, 8, 10, 12]))
              : undefined,
          covariance: {model: 'exponential', sill: 1, range: 2},
          output: {
            values: chunked ? createVectorView('values', 'float32x4', values) : values[0],
            status: chunked ? createVectorView('statuses', 'uint32', statuses) : statuses[0]
          }
        })
      );
      const compiled = graph.compile();
      try {
        const actual = await fixture.submitTwice(compiled, async () => ({
          values: await fixture.readRows(values),
          statuses: await fixture.readRows(statuses)
        }));
        expect(actual.statuses).toEqual(Array(6).fill(0));
        for (let row = 0; row < 6; row++) expect(actual.values[row * 4]).toBeCloseTo(2, 5);
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    });
  }
}

async function runKriging(
  device: Device,
  chunked: boolean,
  sampleIndex: 'grid' | 'brute-force',
  chunkLengths = [1, 3, 2]
) {
  const fixture = createPaddedQueryFixture(device);
  const {graph, input, output} = fixture;
  const positions = input('positions', 'float32x2', new Float32Array([0, 0, 1, 0, 0, 1]));
  const observations = input(
    'observations',
    'float32x4',
    new Float32Array([1, 0, 0, 1, 2, 0, 0, 1, 3, 0, 0, 1])
  );
  const sampleDrift = input(
    'sample-drift',
    'float32',
    new Float32Array([1, 0, 0, 1, 1, 0, 1, 0, 1])
  );
  const total = chunkLengths.reduce((sum, length) => sum + length, 0);
  const lengths = chunked ? chunkLengths : [total];
  const supportChunks: GraphDataView<'float32x4'>[] = [];
  const driftChunks: GraphDataView<'float32'>[] = [];
  const valueChunks: GraphDataView<'float32x4'>[] = [];
  const statusChunks: GraphDataView<'uint32'>[] = [];
  const weightChunks: GraphDataView<'float32'>[] = [];
  let rowBase = 0;
  for (const [chunkIndex, length] of lengths.entries()) {
    const supports = new Float32Array(length * 8);
    const drift = new Float32Array(length * 3);
    for (let row = 0; row < length; row++) {
      const [x, y] = QUERY_POINTS[(rowBase + row) % 6];
      supports.set([x - 0.05, y, 1, 0, x + 0.05, y, 3, 0], row * 8);
      drift.set([1, x + 0.025, y], row * 3);
    }
    supportChunks.push(input(`supports-${chunkIndex}`, 'float32x4', supports));
    driftChunks.push(input(`drift-${chunkIndex}`, 'float32', drift));
    valueChunks.push(output(`values-${chunkIndex}`, 'float32x4', length));
    statusChunks.push(output(`statuses-${chunkIndex}`, 'uint32', length));
    weightChunks.push(output(`weights-${chunkIndex}`, 'float32', length * 3));
    rowBase += length;
  }
  graph.add(
    createLocalKrigingForTesting({
      positions,
      observations,
      sampleDrift,
      supports: chunked
        ? createVectorView('supports', 'float32x4', supportChunks)
        : supportChunks[0],
      queryDrift: chunked
        ? createVectorView('query-drift', 'float32', driftChunks)
        : driftChunks[0],
      supportCount: 2,
      neighborCount: 3,
      mode: 'external-drift',
      driftCount: 3,
      sampleIndex,
      sampleIndexChunkQueryCount: 32,
      covariance: {model: 'exponential', sill: 1, range: 2},
      output: {
        values: chunked ? createVectorView('values', 'float32x4', valueChunks) : valueChunks[0],
        status: chunked ? createVectorView('statuses', 'uint32', statusChunks) : statusChunks[0],
        weights: chunked ? createVectorView('weights', 'float32', weightChunks) : weightChunks[0]
      }
    })
  );
  const compiled = graph.compile();
  try {
    return await fixture.submitTwice(compiled, async () => ({
      values: await fixture.readRows(valueChunks),
      statuses: await fixture.readRows(statusChunks),
      weights: await fixture.readRows(weightChunks)
    }));
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
}

for (const sampleIndex of ['grid', 'brute-force'] as const)
  it(`matches packed grouped queries and the affine-drift oracle (${sampleIndex})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const packed = await runKriging(device, false, sampleIndex);
    const chunked = await runKriging(device, true, sampleIndex);
    expect(chunked).toEqual(packed);
    expect(chunked.statuses).toEqual(Array(6).fill(0));
    for (const [queryIndex, [x, y]] of QUERY_POINTS.entries()) {
      // Affine drift constraints reproduce f(x,y)=1+x+2y, averaged with quadrature weights 1:3.
      expect(chunked.values[queryIndex * 4]).toBeCloseTo(1 + x + 0.025 + 2 * y, 4);
      const orderedSamples = [
        [0, 0],
        [1, 0],
        [0, 1]
      ]
        .map(([sampleX, sampleY], sampleIndex) => ({
          sampleIndex,
          distance: Math.hypot(sampleX - x - 0.025, sampleY - y)
        }))
        .sort(
          (first, second) =>
            first.distance - second.distance || first.sampleIndex - second.sampleIndex
        );
      const barycentric = [1 - x - 0.025 - y, x + 0.025, y];
      for (const [slot, sample] of orderedSamples.entries())
        expect(chunked.weights[queryIndex * 3 + slot]).toBeCloseTo(
          barycentric[sample.sampleIndex],
          4
        );
    }
  });

it('retains the same exact small-input prediction through grid and brute-force routes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  expect(await runKriging(device, true, 'grid')).toEqual(
    await runKriging(device, true, 'brute-force')
  );
});

it('writes 1.1 million prior predictions beyond the old default 32-neighbor weight binding limit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32' | 'float32x2' | 'float32x4'>(
    name: string,
    format: Format,
    data: Float32Array,
    length: number
  ) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, length);
  };
  const rowCount = 1_100_000;
  const neighborCount = 32;
  const supports = input('supports', 'float32x2', new Float32Array(rowCount * 2), rowCount);
  const predictionBuffer = createOutputBuffer(device, rowCount * 4);
  const statusBuffer = createOutputBuffer(device, rowCount);
  const weightBuffer = createOutputBuffer(device, rowCount * neighborCount);
  buffers.push(predictionBuffer, statusBuffer, weightBuffer);
  const predictions = importGraphBuffer(
    graph,
    'predictions',
    predictionBuffer,
    'float32x4',
    rowCount
  );
  const statuses = importGraphBuffer(graph, 'statuses', statusBuffer, 'uint32', rowCount);
  const weights = importGraphBuffer(
    graph,
    'weights',
    weightBuffer,
    'float32',
    rowCount * neighborCount
  );
  const lengths = [400_000, 400_000, 300_000];
  const split = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    name: string,
    view: GraphDataView<Format>,
    width = 1
  ) => {
    let rowBase = 0;
    return createVectorView(
      name,
      view.format,
      lengths.map(length => {
        const chunk = graph.createDataView(view.buffer, {
          format: view.format,
          length: length * width,
          byteOffset: rowBase * width * view.rowByteLength
        });
        rowBase += length;
        return chunk;
      })
    );
  };
  graph.add(
    new GPULocalKriging({
      positions: input('positions', 'float32x2', new Float32Array([0, 0]), 1),
      observations: input('observations', 'float32x4', new Float32Array([0, 0, 0, 0]), 1),
      supports: split('supports', supports),
      mode: 'simple',
      means: [7],
      emptyNeighborhood: 'prior',
      neighborCount,
      sampleIndex: 'brute-force',
      covariance: {model: 'exponential', sill: 1, range: 1},
      output: {
        values: split('predictions', predictions),
        status: split('statuses', statuses),
        weights: split('weights', weights, neighborCount)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    for (const row of [0, 399_999, 400_000, 800_000, rowCount - 1]) {
      const predictionBytes = await predictionBuffer.readAsync(row * 16, 16);
      expect(
        Array.from(new Float32Array(predictionBytes.buffer, predictionBytes.byteOffset, 4))
      ).toEqual([7, 1, 1, 0]);
      const weightBytes = await weightBuffer.readAsync(row * neighborCount * 4, neighborCount * 4);
      expect(
        Array.from(new Float32Array(weightBytes.buffer, weightBytes.byteOffset, neighborCount))
      ).toEqual(Array(neighborCount).fill(0));
      const statusBytes = await statusBuffer.readAsync(row * 4, 4);
      expect(new Uint32Array(statusBytes.buffer, statusBytes.byteOffset, 1)[0]).toBe(0);
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

it('uses global CSR query rows with local chunked supports and outputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    name: string,
    format: Format,
    data: Float32Array | Uint32Array,
    length: number
  ) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, length);
  };
  const chosenSamples = [0, 1, 2, 0, 2, 1];
  const samplePositions = [
    [0, 0],
    [1, 0],
    [0, 1]
  ];
  const supports = input(
    'supports',
    'float32x2',
    new Float32Array(chosenSamples.flatMap(index => samplePositions[index])),
    6
  );
  const split = <Format extends 'float32x2' | 'float32x4' | 'uint32'>(
    name: string,
    view: GraphDataView<Format>
  ) => {
    let rowBase = 0;
    return createVectorView(
      name,
      view.format,
      [1, 3, 2].map(length => {
        const chunk = graph.createDataView(view.buffer, {
          format: view.format,
          length,
          byteOffset: rowBase * view.rowByteLength
        });
        rowBase += length;
        return chunk;
      })
    );
  };
  // Independent output buffers per chunk keep aligned writable binding ranges disjoint.
  const predictionChunks = [1, 3, 2].map((length, index) => {
    const buffer = createOutputBuffer(device, length * 4);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `chunk-prediction-${index}`, buffer, 'float32x4', length),
      length
    };
  });
  const statusChunks = [1, 3, 2].map((length, index) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `chunk-status-${index}`, buffer, 'uint32', length),
      length
    };
  });
  graph.add(
    new GPULocalKriging({
      positions: input('positions', 'float32x2', new Float32Array(samplePositions.flat()), 3),
      observations: input(
        'observations',
        'float32x4',
        new Float32Array([11, 0, 0, 1, 22, 0, 0, 1, 33, 0, 0, 1]),
        3
      ),
      supports: split('supports', supports),
      mode: 'simple',
      neighborCount: 1,
      neighborhoods: input('neighbors', 'uint32', new Uint32Array(chosenSamples), 6),
      neighborhoodOffsets: input('offsets', 'uint32', new Uint32Array([0, 1, 2, 3, 4, 5, 6]), 7),
      covariance: {model: 'exponential', sill: 1, range: 1},
      output: {
        values: createVectorView(
          'predictions',
          'float32x4',
          predictionChunks.map(chunk => chunk.view)
        ),
        status: createVectorView(
          'statuses',
          'uint32',
          statusChunks.map(chunk => chunk.view)
        )
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const values: number[] = [];
    const status: number[] = [];
    for (const chunk of predictionChunks)
      values.push(...(await readFloat32(chunk.buffer, chunk.length * 4)));
    for (const chunk of statusChunks)
      status.push(...(await readUint32(chunk.buffer, chunk.length)));
    expect(status).toEqual(Array(6).fill(0));
    expect(values.filter((_, index) => index % 4 === 0)).toEqual([11, 22, 33, 11, 33, 22]);
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

for (const sampleIndex of ['grid', 'brute-force'] as const)
  it(`preserves grouped offsets and padding across 31/32/33-row dispatches (${sampleIndex})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const lengths = [31, 32, 33];
    const packed = await runKriging(device, false, sampleIndex, lengths);
    const chunked = await runKriging(device, true, sampleIndex, lengths);
    expect(chunked).toEqual(packed);
    expect(chunked.statuses).toEqual(Array(96).fill(0));
    for (let row = 0; row < 96; row++) {
      const [x, y] = QUERY_POINTS[row % 6];
      expect(chunked.values[row * 4]).toBeCloseTo(1 + x + 0.025 + 2 * y, 4);
      expect(
        chunked.weights.slice(row * 3, row * 3 + 3).reduce((sum, weight) => sum + weight, 0)
      ).toBeCloseTo(1, 5);
    }
  });

it('retains a single sample on a constant axis at a large float32 origin', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const krigingDevice: Device = device;
  async function run(origin: number, chunked: boolean, sampleIndex: 'grid' | 'brute-force') {
    const fixture = createPaddedQueryFixture(krigingDevice);
    const {graph, input, output} = fixture;
    const lengths = chunked ? [1, 3, 2] : [6];
    const supports = lengths.map((length, index) =>
      input(
        `supports-${index}`,
        'float32x2',
        new Float32Array(Array.from({length}, () => [origin + 2, 1]).flat())
      )
    );
    const predictions = lengths.map((length, index) =>
      output(`values-${index}`, 'float32x4', length)
    );
    const statuses = lengths.map((length, index) => output(`statuses-${index}`, 'uint32', length));
    const weights = lengths.map((length, index) => output(`weights-${index}`, 'float32', length));
    graph.add(
      new GPULocalKriging({
        positions: input('positions', 'float32x2', new Float32Array([origin, 0])),
        observations: input('observations', 'float32x4', new Float32Array([7, 0, 0, 1])),
        supports: chunked ? createVectorView('supports', 'float32x2', supports) : supports[0],
        neighborCount: 1,
        mode: 'ordinary',
        sampleIndex,
        covariance: {model: 'exponential', sill: 1, range: 2},
        output: {
          values: chunked ? createVectorView('values', 'float32x4', predictions) : predictions[0],
          status: chunked ? createVectorView('statuses', 'uint32', statuses) : statuses[0],
          weights: chunked ? createVectorView('weights', 'float32', weights) : weights[0]
        }
      })
    );
    const compiled = graph.compile();
    try {
      return await fixture.submitTwice(compiled, async () => ({
        values: await fixture.readRows(predictions),
        statuses: await fixture.readRows(statuses),
        weights: await fixture.readRows(weights)
      }));
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
  const baseline = await run(0, false, 'brute-force');
  expect(baseline.statuses).toEqual(Array(6).fill(0));
  expect(baseline.weights).toEqual(Array(6).fill(1));
  for (let row = 0; row < 6; row++) expect(baseline.values[row * 4]).toBeCloseTo(7, 6);
  for (const origin of [0, 20_000_000])
    for (const chunked of [false, true])
      for (const sampleIndex of ['grid', 'brute-force'] as const)
        expect(await run(origin, chunked, sampleIndex)).toEqual(baseline);
});
