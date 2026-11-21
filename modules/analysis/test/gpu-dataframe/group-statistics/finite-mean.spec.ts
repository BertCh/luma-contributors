// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUGroupStatistics} from '../../../src/dataframe-analysis/group-statistics/gpu-group-statistics';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {ValueCubeRig} from '../value-cube/value-cube-rig';
import {getColumnReductionNodes} from '../../../src/dataframe-analysis/group-statistics/group-statistics-moments';

it.each([
  undefined,
  4
])('DPC-109 seven signed maxima keep their group means (tile=%s)', async scratchTileSize => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new ValueCubeRig(device);
  const means = rig.output('float32', 3);
  const sumValues = rig.output('float32', 3);
  const variances = rig.output('float32', 3);
  try {
    rig.run(
      new GPUGroupStatistics({
        keys: rig.input(
          Uint32Array.from([...new Array(7).fill(0), ...new Array(7).fill(1), 2]),
          'uint32'
        ),
        scratchTileSize,
        variance: 'population',
        columns: [
          {
            values: rig.input(
              Float32Array.from([
                ...new Array(7).fill(3.4028234663852886e38),
                ...new Array(7).fill(-3.4028234663852886e38),
                NaN
              ]),
              'float32'
            ),
            statistics: ['sum', 'mean', 'variance'],
            output: {means, sumValues, variances}
          }
        ],
        output: {
          keys: rig.output('uint32', 3),
          counts: rig.output('uint32', 3),
          status: {count: rig.output('uint32', 1), overflow: rig.output('uint32', 1)}
        }
      })
    );
    const result = await rig.readFloat(means);
    console.log('DPC-109 group means adapter/readback', device.info, scratchTileSize, result);
    // Closed form mean([x]*7)=x and population variance=0; each signed sum exceeds f32.
    expect(result).toEqual([3.4028234663852886e38, -3.4028234663852886e38, NaN]);
    expect(await rig.readFloat(sumValues)).toEqual([Infinity, -Infinity, 0]);
    expect(await rig.readFloat(variances)).toEqual([0, 0, NaN]);
  } finally {
    rig.destroy();
  }
});

it('DPC-109 injected group totals retain count 16777217 and empty means', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new ValueCubeRig(device);
  const means = rig.output('float32', 5);
  const sumValues = rig.output('float32', 5);
  const counts = [7, 16777217, 7, 16777217, 0];
  // maxFloat32=(2^24-1)*2^104. For rowCount=16777217, headroom=25 and E=254
  // give unitExponent=58: encode each exact total in those units, without allocating rows.
  const words = counts.flatMap((count, index) => {
    const total = BigInt.asUintN(
      96,
      BigInt(count) * 0xffffffn * (1n << 46n) * (index < 2 ? 1n : -1n)
    );
    return [
      Number(total & 0xffffffffn),
      Number((total >> 32n) & 0xffffffffn),
      Number(total >> 64n)
    ];
  });
  const reduction = getColumnReductionNodes(rig.graph, {
    id: 'injected-groups',
    operation: 'GPUGroupStatistics',
    structure: {capacity: 5, rowCount: 16777217} as Parameters<
      typeof getColumnReductionNodes
    >[1]['structure'],
    valueChunks: [],
    needs: {sums: true, minimum: false, maximum: false, moments: false, zScore: false},
    variance: 'population',
    sumScale: 65536,
    output: {means, sumValues}
  });
  // Replace only the clear and accumulation inputs; exercise the actual group publication node.
  const finish = reduction.nodes.find(node => node.id.endsWith('-finish-sums'))!;
  // The kernel's bindings are graph views; bind uploaded totals to the two read descriptors.
  // Use the graph copy primitive through a tiny upload kernel to retain production offsets.
  const {createWGSLKernelNode} = await import(
    '../../../src/analysis-core/kernels/wgsl-kernel-nodes'
  );
  const sources = rig.input(Uint32Array.from([...words, 254]), 'uint32');
  const countSource = rig.input(Uint32Array.from(counts), 'uint32');
  const sumResource = finish.resources![1];
  if (!('buffer' in sumResource)) throw new Error('group mean sum words must be a buffer');
  const sumWords = sumResource.buffer as GraphDataView<'uint32'>;
  try {
    rig.graph.add(
      createWGSLKernelNode(rig.graph, {
        id: 'inject-group-totals',
        operation: 'mean-regression',
        variant: 'inject',
        invocationCount: 16,
        bindings: [
          {name: 'source', view: sources, type: 'u32', access: 'read'},
          {name: 'destination', view: sumWords, type: 'u32', access: 'read_write'}
        ],
        body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
      })
    );
    rig.graph.add(
      createWGSLKernelNode(rig.graph, {
        id: 'inject-group-counts',
        operation: 'mean-regression',
        variant: 'inject',
        invocationCount: 5,
        bindings: [
          {name: 'source', view: countSource, type: 'u32', access: 'read'},
          {name: 'destination', view: reduction.finiteCounts, type: 'u32', access: 'read_write'}
        ],
        body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
      })
    );
    rig.graph.add(finish);
    rig.run();
    const result = await rig.readFloat(means);
    console.log('DPC-109 injected group means adapter/readback', device.info, result);
    expect(result).toEqual([
      3.4028234663852886e38,
      3.4028234663852886e38,
      -3.4028234663852886e38,
      -3.4028234663852886e38,
      NaN
    ]);
    expect(await rig.readFloat(sumValues)).toEqual([Infinity, Infinity, -Infinity, -Infinity, 0]);
  } finally {
    rig.destroy();
  }
});
