// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {expectPredictionReadback} from './prediction-readback-expectations';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPULocalRBF} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-rbf';
import {GPULocalKriging} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {GPUInverseDistanceWeighting} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-inverse-distance-weighting';
import {getGPUInverseDistanceWeightingParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/spatial-interpolation-parameters';
import {GPUGaussianRandomField} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-gaussian-random-field';
import {getGPUGaussianRandomFieldParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gaussian-random-field-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

for (const kind of ['kriging', 'rbf'] as const) {
  it(`CPL-08 ${kind} distinguishes complete, truncated, intentional local and recovered CSR frames`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const input = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
      name: string,
      data: Float32Array | Uint32Array,
      format: Format
    ) => {
      const buffer = createInputBuffer(device, data);
      buffers.push(buffer);
      return {
        buffer,
        view: importGraphBuffer(
          graph,
          name,
          buffer,
          format,
          data.length / (format === 'float32x4' ? 4 : format === 'float32x2' ? 2 : 1)
        )
      };
    };
    const positions = input('positions', Float32Array.of(-1, 0, 1, 0), 'float32x2').view;
    const queries = input('queries', Float32Array.of(0, 0), 'float32x2').view;
    const offsets = input('offsets', Uint32Array.of(0, 2), 'uint32');
    const neighbors = input('neighbors', Uint32Array.of(0, 1), 'uint32');
    const loss = input('loss', Uint32Array.of(0), 'uint32');
    const valuesBuffer = createOutputBuffer(device, 4);
    const statusBuffer = createOutputBuffer(device, 1);
    buffers.push(valuesBuffer, statusBuffer);
    const output = {
      values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32x4', 1),
      status: importGraphBuffer(graph, 'status', statusBuffer, 'uint32', 1)
    };
    const neighborhoodWeights = {
      offsets: offsets.view,
      neighbors: neighbors.view,
      status: {overflow: loss.view}
    };
    graph.add(
      kind === 'kriging'
        ? new GPULocalKriging({
            positions,
            supports: queries,
            observations: input('data', Float32Array.of(0, 0, 0, 1, 10, 0, 0, 1), 'float32x4').view,
            covariance: {model: 'spherical', range: 4, sill: 1},
            mode: 'ordinary',
            neighborCount: 2,
            neighborhoodWeights,
            output
          })
        : new GPULocalRBF({
            positions,
            queries,
            values: input('data', Float32Array.of(0, 10), 'float32').view,
            kernel: 'gaussian',
            degree: 0,
            neighborCount: 2,
            neighborhoodWeights,
            output
          })
    );
    const compiled = graph.compile();
    try {
      // Reflection symmetry gives 5 for full support; intentionally retaining sample 0 gives 0.
      for (const [count, overflow, expectedStatus, expected] of [
        [2, 0, 0, 5],
        [1, 1, 3, NaN],
        [1, 0, 0, 0],
        [2, 0, 0, 5]
      ]) {
        offsets.buffer.write(Uint32Array.of(0, count));
        loss.buffer.write(Uint32Array.of(overflow));
        submitGraph(device, compiled);
        await device.handle.queue.onSubmittedWorkDone();
        expect(await readUint32(statusBuffer)).toEqual([expectedStatus]);
        expectPredictionReadback(await readFloat32(valuesBuffer), expected, 4);
      }
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}

it('UPT-06 nearest IDW returns 10 while the radius-only fixture returns 28', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Query is 0; values 10 and 100 at distances 1 and 2 give (10 + 100/4)/(1 + 1/4) = 28.
  for (const capacity of [0, 1]) {
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const input = (data: Float32Array) => {
      const buffer = createInputBuffer(device, data);
      buffers.push(buffer);
      return buffer;
    };
    const positions = input(Float32Array.of(1, 0, 2, 0));
    const data = input(Float32Array.of(10, 100));
    const parameters = input(new Float32Array(9));
    const values = createOutputBuffer(device, 1);
    const counts = createOutputBuffer(device, 1);
    buffers.push(values, counts);
    const grid = makeGPURasterMetadata({width: 1, height: 1, origin: [-0.5, 0.5]});
    const contributor = new GPUInverseDistanceWeighting({
      grid,
      positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', 2),
      values: importGraphBuffer(graph, 'data', data, 'float32', 2),
      parameters: importGraphBuffer(graph, 'params', parameters, 'float32', 9),
      indexGridSize: [1, 1],
      indexBounds: [-1, -1, 3, 1],
      maximumNeighborCount: capacity,
      output: {
        values: importGraphBuffer(graph, 'values', values, 'float32', 1),
        counts: importGraphBuffer(graph, 'counts', counts, 'uint32', 1)
      }
    });
    parameters.write(
      contributor.getParameterValues({searchRadius: 3, neighborCount: capacity}, grid)
    );
    graph.add(contributor);
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled);
      await device.handle.queue.onSubmittedWorkDone();
      expectPredictionReadback(await readFloat32(values), capacity ? 10 : 28);
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }
});

it('UPT-03 a tile-local one-cell IDW non-hit yields the independent mean 50', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = (data: Float32Array) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return buffer;
  };
  const grid = makeGPURasterMetadata({width: 1, height: 1, cellSize: [0.125, 1], origin: [0, 0.5]});
  const positions = input(Float32Array.of(0, 0, 0.125, 0));
  const data = input(Float32Array.of(0, 100));
  const parameters = input(getGPUInverseDistanceWeightingParameterValues({searchRadius: 1}, grid));
  const values = createOutputBuffer(device, 1);
  buffers.push(values);
  graph.add(
    new GPUInverseDistanceWeighting({
      grid,
      positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', 2),
      values: importGraphBuffer(graph, 'data', data, 'float32', 2),
      parameters: importGraphBuffer(graph, 'params', parameters, 'float32', 9),
      indexGridSize: [1, 1],
      indexBounds: [-1, -1, 1, 1],
      output: {values: importGraphBuffer(graph, 'values', values, 'float32', 1)}
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    await device.handle.queue.onSubmittedWorkDone();
    expectPredictionReadback(await readFloat32(values), 50);
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

it('CPL-08 conditioned fields expose producer loss separately from intentional approximation and recover', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = (data: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const positions = input(Float32Array.of(0, 0, 1, 0, 2, 0));
  const parameters = input(
    getGPUGaussianRandomFieldParameterValues({
      variogram: {model: 'exponential', sill: 1, nugget: 0, range: 3},
      seed: 1
    })
  );
  const indices = input(Uint32Array.of(0, 2));
  const data = input(Float32Array.of(7, 11));
  // Unequal CSR row lengths 2, 1, 3. Fixed widths would give the middle query both samples.
  const offsets = input(Uint32Array.of(0, 2, 3, 6));
  const neighborhoods = input(Uint32Array.of(0, 1, 0, 1, 0, 0xffffffff));
  const loss = input(Uint32Array.of(0));
  const values = output(3);
  const validity = output(3);
  const approximation = output(1);
  const candidateOverflow = output(1);
  const mass = output(1);
  const rawValues = output(3);
  graph.add(
    new GPUGaussianRandomField({
      id: 'raw',
      positions: importGraphBuffer(graph, 'raw-positions', positions, 'float32x2', 3),
      parameters: importGraphBuffer(graph, 'raw-parameters', parameters, 'float32', 16),
      modeCount: 1,
      output: {
        values: importGraphBuffer(graph, 'raw-values', rawValues, 'float32', 3),
        validity: importGraphBuffer(graph, 'raw-validity', output(3), 'uint32', 3),
        status: {
          approximation: importGraphBuffer(graph, 'raw-approximation', output(1), 'uint32', 1)
        },
        clippedEigenvalueMass: importGraphBuffer(graph, 'raw-mass', output(1), 'float32', 1)
      }
    })
  );
  graph.add(
    new GPUGaussianRandomField({
      positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', 3),
      parameters: importGraphBuffer(graph, 'parameters', parameters, 'float32', 16),
      modeCount: 1,
      conditioning: {
        indices: importGraphBuffer(graph, 'indices', indices, 'uint32', 2),
        values: importGraphBuffer(graph, 'data', data, 'float32', 2),
        neighborhoods: importGraphBuffer(graph, 'neighbors', neighborhoods, 'uint32', 6),
        neighborhoodOffsets: importGraphBuffer(graph, 'offsets', offsets, 'uint32', 4),
        neighborCount: 2,
        status: {candidateOverflow: importGraphBuffer(graph, 'loss', loss, 'uint32', 1)}
      },
      output: {
        values: importGraphBuffer(graph, 'values', values, 'float32', 3),
        validity: importGraphBuffer(graph, 'validity', validity, 'uint32', 3),
        status: {
          approximation: importGraphBuffer(graph, 'approximation', approximation, 'uint32', 1),
          candidateOverflow: importGraphBuffer(
            graph,
            'candidate-overflow',
            candidateOverflow,
            'uint32',
            1
          )
        },
        clippedEigenvalueMass: importGraphBuffer(graph, 'mass', mass, 'float32', 1)
      }
    })
  );
  const compiled = graph.compile();
  try {
    for (const overflow of [0, 1, 0]) {
      loss.write(Uint32Array.of(overflow));
      submitGraph(device, compiled);
      await device.handle.queue.onSubmittedWorkDone();
      expect(await readUint32(approximation)).toEqual([1]);
      expect(await readUint32(candidateOverflow)).toEqual([overflow]);
      const result = await readFloat32(values);
      expect(result.length).toBe(3);
      if (overflow) {
        expect(Array.from(result).every(Number.isNaN)).toBe(true);
        expect(Array.from(await readUint32(validity))).toEqual([0, 0, 0]);
      } else {
        expect(Number.isFinite(result[0])).toBe(true);
        expect(result[0]).toBe(7);
        expect(result[2]).toBe(11);
        const raw = await readFloat32(rawValues);
        expect(raw.length).toBe(3);
        expect(Array.from(raw).every(Number.isFinite)).toBe(true);
        // One noise-free conditioning row at distance 1 with exponential effective range 3:
        // correction = C(1)/C(0) * (7 - raw[0]) = exp(-1) * (7 - raw[0]).
        const expectedMiddle = raw[1] + 0.36787944117144233 * (7 - raw[0]);
        expect(Number.isFinite(result[1])).toBe(true);
        expect(Number.isFinite(expectedMiddle)).toBe(true);
        expect(result[1]).toBeCloseTo(expectedMiddle, 4);
        expect(Array.from(await readUint32(validity))).toEqual([1, 1, 1]);
      }
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

it('UPT-03 translated two-by-two IDW grid keeps nonzero cell steps', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = (data: Float32Array) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return buffer;
  };
  const grid = makeGPURasterMetadata({width: 2, height: 2, cellSize: 0.1, origin: [0, 0.1]});
  const positions = input(Float32Array.of(0, 0, 0.2, 0));
  const data = input(Float32Array.of(0, 100));
  const parameters = input(getGPUInverseDistanceWeightingParameterValues({searchRadius: 1}, grid));
  const values = createOutputBuffer(device, 4);
  buffers.push(values);
  graph.add(
    new GPUInverseDistanceWeighting({
      grid,
      positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', 2),
      values: importGraphBuffer(graph, 'data', data, 'float32', 2),
      parameters: importGraphBuffer(graph, 'params', parameters, 'float32', 9),
      indexGridSize: [1, 1],
      indexBounds: [-1, -1, 1, 1],
      output: {values: importGraphBuffer(graph, 'values', values, 'float32', 4)}
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    await device.handle.queue.onSubmittedWorkDone();
    const actual = await readFloat32(values);
    // Cell centers are (0.05/0.15, +/-0.05): squared distances are 0.005 and 0.025.
    // Inverse-square normalized weights are 5/6 and 1/6, reversed on the right.
    const expected = [16.666666666666668, 83.33333333333333, 16.666666666666668, 83.33333333333333];
    expect(actual.length).toBe(4);
    expect(Array.from(actual).every(Number.isFinite)).toBe(true);
    expected.forEach((value, index) => expect(actual[index]).toBeCloseTo(value, 4));
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
