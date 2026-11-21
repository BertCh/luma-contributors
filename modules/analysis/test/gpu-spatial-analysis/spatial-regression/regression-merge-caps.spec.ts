// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialErrorGM} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm';
import {GPUSpatialTwoStageLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';
import {GPUSpatialWeightsTraces} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import type {Buffer} from '@luma.gl/core';

it('three-row chain suppresses a lost submission and recovers its norm on the same graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const input = <Format extends 'uint32' | 'float32'>(
    format: Format,
    values: Uint32Array | Float32Array
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `chain-${serial++}`, buffer, format, values.length)
    };
  };
  const offsets = input('uint32', Uint32Array.of(0, 1, 3, 4));
  const neighbors = input('uint32', Uint32Array.of(1, 0, 2, 1));
  const loss = input('uint32', Uint32Array.of(0));
  const weights = input('float32', Float32Array.of(1, 1, 1, 1));
  const norm = createOutputBuffer(device, 1);
  buffers.push(norm);
  try {
    graph.add(
      new GPUSpatialWeightsTraces({
        weights: {
          offsets: offsets.view,
          neighbors: neighbors.view,
          weights: weights.view,
          status: {candidateOverflow: loss.view}
        },
        maximumPower: 2,
        mode: 'exact',
        output: {
          powerTraces: createTransientView(graph, 'chain-traces', 'float32', 2),
          powerSums: createTransientView(graph, 'chain-sums', 'float32', 2),
          infinityNorm: importGraphBuffer(graph, 'chain-norm', norm, 'float32', 1)
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect((await readFloat32(norm, 1))[0]).toBe(2);
      offsets.buffer.write(Uint32Array.of(0, 1, 2, 3));
      neighbors.buffer.write(Uint32Array.of(1, 0, 1, 1));
      loss.buffer.write(Uint32Array.of(1));
      submitGraph(device, compiled, undefined);
      expect(Number.isNaN((await readFloat32(norm, 1))[0])).toBe(true);
      offsets.buffer.write(Uint32Array.of(0, 1, 3, 4));
      neighbors.buffer.write(Uint32Array.of(1, 0, 2, 1));
      loss.buffer.write(Uint32Array.of(0));
      submitGraph(device, compiled, undefined);
      expect((await readFloat32(norm, 1))[0]).toBe(2);
    } finally {
      compiled.destroy();
    }
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
});

it('retains the validated diagonal chunks when the caller changes the split before declaration', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const input = <Format extends 'float32' | 'uint32'>(
    format: Format,
    values: Float32Array | Uint32Array
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, `snapshot-${serial++}`, buffer, format, values.length);
  };
  const createChunk = (rowStart: number, values: number[]) => ({
    rowStart,
    rowCount: values.length,
    weights: {
      offsets: input(
        'uint32',
        Uint32Array.from({length: values.length + 1}, (_, row) => row)
      ),
      neighbors: input(
        'uint32',
        Uint32Array.from(values, (_, row) => rowStart + row)
      ),
      weights: input('float32', Float32Array.from(values))
    }
  });
  const chunks = [createChunk(0, [1, 1]), createChunk(2, [1, 7])];
  const normBuffer = createOutputBuffer(device, 1);
  buffers.push(normBuffer);
  try {
    const contributor = new GPUSpatialWeightsTraces({
      weights: {rowCount: 4, chunks},
      maximumPower: 1,
      tileRowCount: 2,
      mode: 'stochastic',
      probeCount: 1,
      output: {
        powerTraces: createTransientView(graph, 'snapshot-traces', 'float32', 1),
        powerSums: createTransientView(graph, 'snapshot-sums', 'float32', 1),
        infinityNorm: importGraphBuffer(graph, 'snapshot-norm', normBuffer, 'float32', 1)
      }
    });
    chunks.splice(0, chunks.length, createChunk(0, [1]), createChunk(1, [1, 1, 7]));
    graph.add(contributor);
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect((await readFloat32(normBuffer, 1))[0]).toBe(7);
    } finally {
      compiled.destroy();
    }
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
});

it('accepts 4097-tile hierarchical layouts before adding GPU commands', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `merge-cap-${serial++}`, format, length);
  // Fixed input: each of the 4097 rows would produce one norm partial.
  const rows = 4097;
  const common = {
    id: 'bounded-merge',
    tileRowCount: 1,
    weights: {
      offsets: view('uint32', rows + 1),
      neighbors: view('uint32', rows),
      weights: view('float32', rows)
    },
    predictorCount: 1,
    predictors: view('float32', rows),
    response: view('float32', rows),
    output: {
      table: view('float32', 12),
      summary: view('float32', 32),
      status: {code: view('uint32', 1)}
    }
  };
  for (const create of [
    () => new GPUSpatialErrorGM(common),
    () => new GPUSpatialTwoStageLeastSquares(common),
    () =>
      new GPUSpatialWeightsTraces({
        ...common,
        maximumPower: 1,
        mode: 'stochastic',
        output: {powerTraces: view('float32', 1), powerSums: view('float32', 1)}
      })
  ])
    expect(create().tileCount).toBe(4097);
});
