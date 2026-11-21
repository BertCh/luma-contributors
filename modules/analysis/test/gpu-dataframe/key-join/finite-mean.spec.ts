// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUKeyLookup} from '../../../src/dataframe-analysis/key-join/gpu-key-join';
import {FINITE_MEAN_WGSL} from '../../../src/dataframe-analysis/key-join/finite-mean-wgsl';
import {FIXED_POINT_WGSL} from '../../../src/analysis-core/kernels/fixed-point-accumulator';
import {GET_NAN_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {ValueCubeRig} from '../value-cube/value-cube-rig';

it('earlier-fix key lookup keeps seven maximum finite values finite on the adapter', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  console.log('key-join extreme finite mean adapter', device.info);
  const rig = new ValueCubeRig(device);
  const means = rig.output('float32', 3);
  const sums = rig.output('float32', 3);
  try {
    rig.run(
      new GPUKeyLookup({
        leftKeys: rig.input(new Uint32Array([7, 9, 7]), 'uint32'),
        rightKeys: rig.input(new Uint32Array(7).fill(7), 'uint32'),
        aggregates: [
          {
            operation: 'mean',
            column: rig.input(new Float32Array(7).fill(3.4028234663852886e38), 'float32'),
            output: means
          },
          {
            operation: 'sum',
            column: rig.input(new Float32Array(7).fill(3.4028234663852886e38), 'float32'),
            output: sums
          }
        ],
        output: {}
      })
    );
    const result = await rig.readFloat(means);
    console.log('key-join seven-row extreme means', result);
    // Closed form mean([x]*7)=x; the separately published sum necessarily overflows.
    expect(result).toEqual([3.4028234663852886e38, NaN, 3.4028234663852886e38]);
    expect(await rig.readFloat(sums)).toEqual([Infinity, NaN, Infinity]);
  } finally {
    rig.destroy();
  }
});

it('earlier-fix mean rounding retains counts above binary32 integer precision', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new ValueCubeRig(device);
  // Exact independently constructed totals: maxFloat32 = (2^24-1)*2^104.
  // Inject totals into the production mean helper to exercise n=16777217 without 16M rows.
  const counts = [7, 16777217, 7, 16777217];
  const words = counts.flatMap((count, index) => {
    const signed = BigInt(count) * 0xffffffn * (index < 2 ? 1n : -1n);
    const total = BigInt.asUintN(96, signed);
    return [
      Number(total & 0xffffffffn),
      Number((total >> 32n) & 0xffffffffn),
      Number(total >> 64n),
      count
    ];
  });
  const totals = rig.input(Uint32Array.from(words), 'uint32');
  const output = rig.output('float32', 4);
  try {
    const node = createWGSLKernelNode(rig.graph, {
      id: 'extreme-mean-counts',
      operation: 'mean-regression',
      variant: 'exact-counts',
      invocationCount: 4,
      bindings: [
        {name: 'totals', view: totals, type: 'u32', access: 'read'},
        {name: 'means', view: output, type: 'f32', access: 'read_write'}
      ],
      declarations: `${GET_NAN_WGSL}\n${FIXED_POINT_WGSL}\n${FINITE_MEAN_WGSL}`,
      body: `let base = totalsOffset + 4u * index;
  let total = vec3<u32>(totals[base],totals[base+1u],totals[base+2u]);
  means[meansOffset+index] = getFiniteFixedPointMean(total,104i,totals[base+3u]);`
    });
    rig.graph.add(node);
    rig.run();
    const result = await rig.readFloat(output);
    console.log('key-join large-count mean adapter/readback', device.info, result);
    expect(result).toEqual([
      3.4028234663852886e38, 3.4028234663852886e38, -3.4028234663852886e38, -3.4028234663852886e38
    ]);
  } finally {
    rig.destroy();
  }
});

it('B1 ordinary sum and mean lookups compile every generated module without optional loss output', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new ValueCubeRig(device);
  const sums = rig.output('float32', 2);
  const means = rig.output('float32', 2);
  try {
    const column = rig.input(Float32Array.of(1, 2, 3), 'float32');
    rig.run(
      new GPUKeyLookup({
        leftKeys: rig.input(Uint32Array.of(4, 9), 'uint32'),
        rightKeys: rig.input(Uint32Array.of(4, 9, 4), 'uint32'),
        aggregates: [
          {operation: 'sum', column, output: sums},
          {operation: 'mean', column, output: means}
        ],
        output: {}
      })
    );
    // Key 4 has [1,3], key 9 has [2].
    expect(await rig.readFloat(sums)).toEqual([4, 2]);
    expect(await rig.readFloat(means)).toEqual([2, 2]);
  } finally {
    rig.destroy();
  }
});
