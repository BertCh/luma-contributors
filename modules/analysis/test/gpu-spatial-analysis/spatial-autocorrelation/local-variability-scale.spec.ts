// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPULocalSpatialHeteroscedasticity,
  GPUMultivariateLocalGeary
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-local-variability';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {GraphRig} from '../areal-interpolation/areal-interpolation-harness';
import {requireTestDevice} from '../../utils/require-test-device';

for (const kind of ['losh', 'multivariate'] as const) {
  it(`${kind} uses an exact odd population above 2^24 and matches a closed-form local statistic`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rows = 2 ** 24 + 1;
    const graph = new GPUCommandGraph(device);
    const values = createTransientView(graph, 'values', 'float32', rows);
    const offsets = createTransientView(graph, 'offsets', 'uint32', rows + 1);
    const neighbors = createTransientView(graph, 'neighbors', 'uint32', rows);
    const weights = createTransientView(graph, 'weights', 'float32', rows);
    const statistic = createTransientView(graph, 'statistic', 'float32', rows);
    const status = createTransientView(graph, 'status', 'uint32', rows);
    const countBuffer = createOutputBuffer(device, 1);
    const samplesBuffer = createOutputBuffer(device, 7);
    const count = importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1);
    const samples = importGraphBuffer(graph, 'samples', samplesBuffer, 'float32', 7);
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'cycle-initialize',
        operation: 'test',
        invocationCount: rows + 1,
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read_write'},
          {name: 'offsets', view: offsets, type: 'u32', access: 'read_write'},
          {name: 'neighbors', view: neighbors, type: 'u32', access: 'read_write'},
          {name: 'weights', view: weights, type: 'f32', access: 'read_write'}
        ],
        body: `offsets[offsetsOffset + index] = index; if (index < ${rows}u) { values[valuesOffset + index] = f32(index & 1u); neighbors[neighborsOffset + index] = (index + 1u) % ${rows}u; weights[weightsOffset + index] = 1.0; }`
      })
    );
    const input = {weights: {offsets, neighbors, weights}, values};
    graph.add(
      kind === 'losh'
        ? new GPULocalSpatialHeteroscedasticity({
            ...input,
            output: {statistic, status, residualCount: count}
          })
        : new GPUMultivariateLocalGeary({
            ...input,
            variableCount: 1,
            output: {statistic, status, includedCount: count}
          })
    );
    const sampleRows = [0, 1, 2 ** 24 - 2, 2 ** 24 - 1, 2 ** 24, rows - 2, rows - 1];
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'gather-samples',
        operation: 'test',
        declarations: GET_NAN_WGSL,
        invocationCount: sampleRows.length,
        bindings: [
          {name: 'statistic', view: statistic, type: 'f32', access: 'read'},
          {name: 'status', view: status, type: 'u32', access: 'read'},
          {name: 'samples', view: samples, type: 'f32', access: 'read_write'}
        ],
        body: `let sampleRows = array<u32, 7>(${sampleRows.map(row => `${row}u`).join(', ')}); let row = sampleRows[index]; samples[samplesOffset + index] = select(getNaN(), statistic[statisticOffset + row], status[statusOffset + row] == 0u);`
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(countBuffer)).toEqual([rows]);
      const actual = await readFloat32(samplesBuffer);
      const probability = Math.floor(rows / 2) / rows;
      for (const [sampleIndex, row] of sampleRows.entries()) {
        const expected =
          kind === 'losh'
            ? row === rows - 2
              ? 0
              : rows / (rows - 1)
            : row === rows - 1
              ? 0
              : 1 / (probability * (1 - probability));
        expect(actual[sampleIndex]).toBeCloseTo(expected, 5);
      }
    } finally {
      compiled.destroy();
      countBuffer.destroy();
      samplesBuffer.destroy();
    }
  }, 300_000);
}

it('tiled two-variable Geary and LOSH match a four-row closed form', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device);
  try {
    const weights = {
      offsets: rig.input(Uint32Array.of(0, 1, 2, 3, 4), 'uint32'),
      neighbors: rig.input(Uint32Array.of(1, 2, 3, 0), 'uint32'),
      weights: rig.input(Float32Array.of(1, 1, 1, 1), 'float32')
    };
    const geary = rig.output('float32', 4);
    const losh = rig.output('float32', 4);
    const population = rig.output('uint32', 1);
    rig.run(
      new GPUMultivariateLocalGeary({
        weights,
        variableCount: 2,
        values: rig.input(Float32Array.of(0, 4, 1, 2, 0, 4, 1, 2), 'float32'),
        output: {
          statistic: geary.view,
          status: rig.output('uint32', 4).view,
          count: population.view
        }
      }),
      new GPULocalSpatialHeteroscedasticity({
        weights,
        values: rig.input(Float32Array.of(0, 1, 0, 1), 'float32'),
        output: {statistic: losh.view, status: rig.output('uint32', 4).view}
      })
    );
    expect(await population.readUint32()).toEqual([4]);
    for (const value of await geary.readFloat32()) expect(value).toBeCloseTo(4, 6);
    for (const value of await losh.readFloat32()) expect(value).toBeCloseTo(1, 6);
  } finally {
    rig.destroy();
  }
});

for (const values of [
  [2 ** 67, 2 ** 67 + 2 ** 48],
  [0, 2 ** 64]
])
  it(`Geary preserves finite variance and dimensionless distances for [${values}]`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new GraphRig(device);
    const statistic = rig.output('float32', 2);
    const status = rig.output('uint32', 2);
    try {
      rig.run(
        new GPUMultivariateLocalGeary({
          variableCount: 1,
          weights: {
            offsets: rig.input(Uint32Array.of(0, 1, 2), 'uint32'),
            neighbors: rig.input(Uint32Array.of(1, 0), 'uint32'),
            weights: rig.input(Float32Array.of(1, 1), 'float32')
          },
          values: rig.input(Float32Array.from(values), 'float32'),
          output: {statistic: statistic.view, status: status.view}
        })
      );
      expect(await status.readUint32()).toEqual([0, 0]);
      // For any two distinct rows, squared difference divided by population variance is four.
      for (const value of await statistic.readFloat32()) expect(value).toBeCloseTo(4, 6);
    } finally {
      rig.destroy();
    }
  });
