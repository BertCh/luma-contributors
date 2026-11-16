// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createStatisticsPaddingProbe} from './statistics-padding-probe';
import {createSpatialStatisticsCollectiveNode} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-statistics-collective-node';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUGlobalSpatialStatistics} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/gpu-global-spatial-statistics';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {GraphRig} from '../areal-interpolation/areal-interpolation-harness';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  computeGlobalSpatialStatisticsOracle,
  getGearyMoments,
  getMoranMoments
} from './global-spatial-statistics-oracle';

for (const rows of [2 ** 24 + 1, 30_000_000]) {
  it(`global statistics and analytic inference match a closed form on ${rows} alternating cycle rows`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const offsets = createTransientView(graph, 'cycle-offsets', 'uint32', rows + 1);
    const neighbors = createTransientView(graph, 'cycle-neighbors', 'uint32', rows);
    const weights = createTransientView(graph, 'cycle-weights', 'float32', rows);
    const values = createTransientView(graph, 'cycle-values', 'float32', rows);
    const summaryBuffer = createOutputBuffer(device, 52);
    const countsBuffer = createOutputBuffer(device, 3);
    const joinsBuffer = createOutputBuffer(device, 3);
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'closed-form-cycle',
        operation: 'test',
        invocationCount: rows + 1,
        bindings: [
          {name: 'offsets', view: offsets, type: 'u32', access: 'read_write'},
          {name: 'neighbors', view: neighbors, type: 'u32', access: 'read_write'},
          {name: 'weights', view: weights, type: 'f32', access: 'read_write'},
          {name: 'values', view: values, type: 'f32', access: 'read_write'}
        ],
        body: `offsets[offsetsOffset + index] = index;
if (index < ${rows}u) { neighbors[neighborsOffset + index] = (index + 1u) % ${rows}u; weights[weightsOffset + index] = 1.0; values[valuesOffset + index] = f32(index & 1u); }`
      })
    );
    graph.add(
      new GPUGlobalSpatialStatistics({
        weights: {offsets, neighbors, weights},
        values,
        statistics: ['moran', 'geary', 'getisOrdG', 'joinCount'],
        output: {
          summary: importGraphBuffer(graph, 'summary', summaryBuffer, 'float32', 52),
          counts: importGraphBuffer(graph, 'counts', countsBuffer, 'uint32', 3),
          joinCounts: importGraphBuffer(graph, 'joins', joinsBuffer, 'uint32', 3)
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const summary = await readFloat32(summaryBuffer);
      const black = Math.floor(rows / 2);
      const white = rows - black;
      const probability = black / rows;
      const complement = white / rows;
      const squared = rows * probability * complement;
      const kurtosis = (probability ** 3 + complement ** 3) / (probability * complement);
      const moran = getMoranMoments(rows, rows, rows, 4 * rows, kurtosis);
      const geary = getGearyMoments(rows, rows, rows, 4 * rows, kurtosis);
      expect(Array.from(await readUint32(countsBuffer))).toEqual([rows, 0, black]);
      expect(Array.from(await readUint32(joinsBuffer))).toEqual([0, 2 * black, rows % 2]);
      expect(summary[4]).toBeCloseTo(probability, 6);
      expect(summary[5]).toBeCloseTo(probability * complement, 6);
      expect(summary[8]).toBeCloseTo(-black / white, 5);
      expect(summary[9]).toBe(Math.fround(-1 / (rows - 1)));
      expect(summary[16]).toBeCloseTo(((rows - 1) * black) / (rows * squared), 5);
      expect(summary[24]).toBe(0);
      for (const [position, expected] of [
        [10, moran[1]],
        [13, moran[2]],
        [18, geary[0]],
        [21, geary[1]],
        [26, (moran[2] * squared ** 2) / (black * (black - 1)) ** 2]
      ] as const) {
        expect(Math.abs(summary[position] - expected)).toBeLessThan(
          1e-4 * Math.abs(expected) + 1e-30
        );
      }
    } finally {
      compiled.destroy();
      summaryBuffer.destroy();
      countsBuffer.destroy();
      joinsBuffer.destroy();
    }
  }, 300_000);
}

it('unequal chunks, masks, islands, and row normalization match an independent global oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 769;
  const values = Float32Array.from({length: rows}, (_, row) =>
    row % 5 === 0 ? 0 : (row % 11) + 1
  );
  const secondValues = Float32Array.from({length: rows}, (_, row) => (row % 7) - 2);
  const mask = Uint32Array.from({length: rows}, (_, row) => (row % 17 === 0 ? 0 : 1));
  const offsets = Uint32Array.from({length: rows + 1}, (_, row) => row);
  const neighbors = Uint32Array.from({length: rows}, (_, row) => (row + 1) % rows);
  const weights = Float32Array.from({length: rows}, (_, row) => (row % 23 === 0 ? 0 : 2));
  for (const islands of ['keep', 'exclude'] as const)
    for (const chunkSizes of [[rows], [1, 255, 256, 257]]) {
      const graph = new GPUCommandGraph(device);
      const buffers: ReturnType<typeof createInputBuffer>[] = [];
      let serial = 0;
      const input = <Format extends 'float32' | 'uint32'>(
        data: Float32Array | Uint32Array,
        format: Format
      ) => {
        // A four-byte prefix exercises the actual aligned storage range and element offset.
        const padded =
          data instanceof Float32Array
            ? new Float32Array(data.length + 1)
            : new Uint32Array(data.length + 1);
        padded.set(data, 1);
        const buffer = createInputBuffer(device, padded);
        buffers.push(buffer);
        const whole = importGraphBuffer(
          graph,
          `input-${serial++}`,
          buffer,
          format,
          data.length + 1
        );
        return graph.createDataView(whole.buffer, {format, length: data.length, byteOffset: 4});
      };
      let rowStart = 0;
      const chunks = chunkSizes.map(rowCount => {
        const start = rowStart;
        rowStart += rowCount;
        return {
          rowStart: start,
          rowCount,
          weights: {
            offsets: input(
              Uint32Array.from({length: rowCount + 1}, (_, row) => row),
              'uint32'
            ),
            neighbors: input(neighbors.slice(start, rowStart), 'uint32'),
            weights: input(weights.slice(start, rowStart), 'float32')
          }
        };
      });
      const summaryBuffer = createOutputBuffer(device, 52);
      const countsBuffer = createOutputBuffer(device, 3);
      const contributor = new GPUGlobalSpatialStatistics({
        weights: {
          rowCount: rows,
          chunks,
          information: {rowCount: rows, transform: 'row', symmetric: false}
        },
        values: input(values, 'float32'),
        secondValues: input(secondValues, 'float32'),
        mask: input(mask, 'uint32'),
        islands,
        statistics: ['moran', 'geary', 'getisOrdG', 'bivariateMoran', 'joinCount'],
        output: {
          summary: importGraphBuffer(graph, 'summary', summaryBuffer, 'float32', 52),
          counts: importGraphBuffer(graph, 'counts', countsBuffer, 'uint32', 3)
        }
      });
      graph.add(contributor);
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const actual = await readFloat32(summaryBuffer);
        const expected = computeGlobalSpatialStatisticsOracle({
          weights: {offsets, neighbors, weights},
          values,
          secondValues,
          mask,
          islands,
          standardized: true
        });
        expect(Array.from(await readUint32(countsBuffer))).toEqual([
          expected.count,
          expected.islandCount,
          expected.blackCount
        ]);
        for (const [name, position] of [
          ['moran', 8],
          ['geary', 16],
          ['getisOrdG', 24],
          ['bivariateMoran', 32]
        ] as const) {
          const fields = [
            'statistic',
            'expected',
            'varianceNormality',
            'zNormality',
            'pNormality',
            'varianceRandomization',
            'zRandomization',
            'pRandomization'
          ] as const;
          for (const [field, key] of fields.entries()) {
            const value = expected[name][key];
            if (Number.isNaN(value)) expect(actual[position + field]).toBeNaN();
            else
              expect(Math.abs(actual[position + field] - value)).toBeLessThan(
                1e-4 * Math.max(Math.abs(value), 1e-4)
              );
          }
        }
      } finally {
        compiled.destroy();
        for (const buffer of buffers) buffer.destroy();
        summaryBuffer.destroy();
        countsBuffer.destroy();
      }
    }
});

for (const mode of ['tiled', 'reference'] as const)
  it(`${mode} global statistics compile on WebGPU and match a four-row cycle`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new GraphRig(device);
    // Import one physical output allocation once. Counts are written separately; summary/joins
    // are co-bound and use disjoint aligned bindings.
    const packed = rig.output('uint32', 128);
    const summary = rig.graph.createDataView(packed.view.buffer, {
      format: 'float32',
      length: 52,
      byteOffset: 12
    });
    const joins = rig.graph.createDataView(packed.view.buffer, {
      format: 'uint32',
      length: 3,
      byteOffset: 256
    });
    const counts = rig.graph.createDataView(packed.view.buffer, {format: 'uint32', length: 3});
    try {
      rig.run(
        new GPUGlobalSpatialStatistics({
          mode,
          weights: {
            offsets: rig.input(Uint32Array.of(0, 1, 2, 3, 4), 'uint32'),
            neighbors: rig.input(Uint32Array.of(1, 2, 3, 0), 'uint32'),
            weights: rig.input(Float32Array.of(1, 1, 1, 1), 'float32')
          },
          values: rig.input(Float32Array.of(0, 1, 0, 1), 'float32'),
          statistics: ['moran', 'geary', 'getisOrdG', 'joinCount'],
          output: {summary, joinCounts: joins, counts}
        })
      );
      const actual = (await packed.readFloat32()).slice(3, 55);
      expect(actual[8]).toBeCloseTo(-1, 6);
      expect(actual[16]).toBeCloseTo(1.5, 6);
      expect(actual[24]).toBe(0);
      expect((await packed.readUint32()).slice(64, 67)).toEqual([0, 4, 0]);
      expect((await packed.readUint32()).slice(0, 3)).toEqual([4, 0, 2]);
    } finally {
      rig.destroy();
    }
  });

for (const rows of [1, 257, 769])
  for (const readControl of [false, true])
    it(`cooperative tails compile with ${rows} rows and ${readControl ? 'read' : 'unread'} control values`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new GraphRig(device);
      const controlValues = new Uint32Array(64);
      controlValues[0] = 7;
      const controlBuffer = device.createBuffer({
        data: controlValues,
        usage: Buffer.STORAGE | Buffer.UNIFORM | Buffer.COPY_DST
      });
      const control = importGraphBuffer(
        rig.graph,
        'cooperative-control',
        controlBuffer,
        'uint32',
        64
      );
      const output = rig.output('uint32', Math.ceil(rows / 256));
      try {
        rig.graph.add(
          createSpatialStatisticsCollectiveNode(rig.graph, {
            id: 'cooperative-tail',
            operation: 'statistics-webgpu-regression',
            invocationCount: rows,
            control,
            bindings: [{name: 'result', view: output.view, type: 'u32', access: 'read_write'}],
            declarations: 'var<workgroup> partialCounts: array<u32, 256>;',
            body: `partialCounts[localInvocationIndex] = select(0u, ${readControl ? 'getStatisticsControl(0u)' : '1u'}, index < INVOCATION_COUNT);
workgroupBarrier();
for (var stride = 128u; stride > 0u; stride /= 2u) {
 if (localInvocationIndex < stride) { partialCounts[localInvocationIndex] += partialCounts[localInvocationIndex + stride]; }
 workgroupBarrier();
}
if (localInvocationIndex == 0u) { result[resultOffset + index / 256u] = partialCounts[0]; }`
          })
        );
        rig.run();
        const expected = (factor: number) =>
          Array.from(
            {length: Math.ceil(rows / 256)},
            (_, groupIndex) => Math.min(256, rows - groupIndex * 256) * factor
          );
        expect(await output.readUint32()).toEqual(expected(readControl ? 7 : 1));
        controlValues[0] = 11;
        controlBuffer.write(controlValues);
        rig.resubmit();
        expect(await output.readUint32()).toEqual(expected(readControl ? 11 : 1));
      } finally {
        rig.destroy();
        controlBuffer.destroy();
      }
    });

it('cooperative statistics count logical groups across a padded 2D dispatch on WebGPU', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const output = createOutputBuffer(device, 1);
  output.write(Uint32Array.of(0));
  const {node, groupCount} = createStatisticsPaddingProbe(
    graph,
    importGraphBuffer(graph, 'total', output, 'uint32', 1)
  );
  graph.add(node);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    expect(await readUint32(output)).toEqual([groupCount]);
    output.write(Uint32Array.of(0));
    submitGraph(device, compiled);
    expect(await readUint32(output)).toEqual([groupCount]);
  } finally {
    compiled.destroy();
    output.destroy();
  }
});
