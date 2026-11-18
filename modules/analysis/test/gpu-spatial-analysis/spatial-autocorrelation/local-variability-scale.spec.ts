// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPULocalSpatialHeteroscedasticity,
  GPU_LOCAL_VARIABILITY_STATUS,
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
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

for (const kind of ['losh', 'geary'] as const)
  it(`final2 ${kind} getter-backed CSR and inherited mask match ordinary objects`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    try {
      const original = rig.uploadWeights({
        offsets: [0, 1, 3, 5, 6],
        neighbors: [1, 0, 2, 1, 3, 2],
        weights: [1, 1, 1, 1, 1, 1],
        distances: []
      });
      const mask = rig.input(Uint32Array.of(1, 1, 1, 0), 'uint32', 4);
      const values = rig.input(Float32Array.of(1, 2, 4, 8), 'float32', 4);
      const results = [];
      for (const getterBacked of [false, true]) {
        let backing = original;
        let backingMask = mask;
        const weights = getterBacked
          ? Object.create({
              get offsets() {
                return backing.offsets;
              },
              get neighbors() {
                return backing.neighbors;
              },
              get weights() {
                return backing.weights;
              }
            })
          : original;
        const statistic = rig.output('float32', 4);
        const status = rig.output('uint32', 4);
        const props = Object.assign(
          Object.create(
            getterBacked
              ? {
                  get mask() {
                    return backingMask;
                  }
                }
              : {}
          ),
          {
            id: `${kind}-${getterBacked}`,
            weights,
            values,
            variableCount: 1,
            output: {statistic: statistic.view, status: status.view}
          },
          getterBacked ? {} : {mask}
        );
        const producer =
          kind === 'losh'
            ? new GPULocalSpatialHeteroscedasticity(props)
            : new GPUMultivariateLocalGeary(props);
        if (getterBacked) {
          backing = {...original, offsets: undefined as never};
          backingMask = undefined as never;
        }
        rig.graph.add(producer);
        results.push({statistic, status});
      }
      const compiled = rig.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const [ordinary, getters] = await Promise.all(
          results.map(result => readFloat32(result.statistic.buffer, 4))
        );
        expect(getters).toEqual(ordinary);
        for (const result of results)
          expect(await readUint32(result.status.buffer, 4)).toEqual([0, 0, 0, 1]);
        const expected = kind === 'losh' ? [1 / 7, 10 / 7, 1 / 7] : [9 / 14, 45 / 28, 18 / 7];
        expected.forEach((value, row) => {
          expect(Number.isFinite(getters[row])).toBe(true);
          expect(getters[row]).toBeCloseTo(value, 5);
        });
        expect(Number.isNaN(getters[3])).toBe(true);
      } finally {
        compiled.destroy();
      }
    } finally {
      rig.destroy();
    }
  });

it('final1 candidate-only loss invalidates LOSH and its residuals, then recovers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const candidateOverflow = rig.input(new Uint32Array([0]), 'uint32', 1);
    const weights = {
      ...rig.uploadWeights({
        offsets: [0, 1, 3, 5, 6],
        neighbors: [1, 0, 2, 1, 3, 2],
        weights: [1, 1, 1, 1, 1, 1],
        distances: []
      }),
      status: {candidateOverflow}
    };
    const statistic = rig.output('float32', 4);
    const status = rig.output('uint32', 4);
    const residuals = rig.output('float32', 4);
    const count = rig.output('uint32', 1);
    rig.graph.add(
      new GPULocalSpatialHeteroscedasticity({
        weights,
        values: rig.input(new Float32Array([1, 2, 4, 8]), 'float32', 4),
        output: {
          statistic: statistic.view,
          status: status.view,
          normalizedResiduals: residuals.view,
          residualCount: count.view
        }
      })
    );
    const compiled = rig.graph.compile();
    try {
      let complete: number[] | undefined;
      for (const flag of [0, 1, 0]) {
        rig.bufferOf(candidateOverflow).write(new Uint32Array([flag]));
        submitGraph(device, compiled, undefined);
        const result = await readFloat32(statistic.buffer, 4);
        const residualValues = await readFloat32(residuals.buffer, 4);
        expect(await readUint32(status.buffer, 4)).toEqual(
          new Array(4).fill(
            flag ? GPU_LOCAL_VARIABILITY_STATUS.INCOMPLETE : GPU_LOCAL_VARIABILITY_STATUS.OK
          )
        );
        if (flag) {
          expect(result.every(Number.isNaN)).toBe(true);
          expect(residualValues.every(Number.isNaN)).toBe(true);
          expect(await readUint32(count.buffer, 1)).toEqual([0]);
        } else {
          expect(result.every(Number.isFinite)).toBe(true);
          expect(residualValues.every(Number.isFinite)).toBe(true);
          // Squared residuals [1, 1/4, 1, 16] have mean 73/16.
          for (const [row, expected] of [4 / 73, 16 / 73, 130 / 73, 16 / 73].entries())
            expect(result[row]).toBeCloseTo(expected, 5);
          for (const [row, expected] of [16 / 73, 4 / 73, 16 / 73, 256 / 73].entries())
            expect(residualValues[row]).toBeCloseTo(expected, 5);
          expect(await readUint32(count.buffer, 1)).toEqual([4]);
          if (complete) expect(result).toEqual(complete);
          complete = result;
        }
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});

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
