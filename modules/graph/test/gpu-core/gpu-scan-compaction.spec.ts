// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, GPUCompaction, GPUScan} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  makePseudoRandomUint32,
  withDeviceCapabilities,
  type CapabilityOverrides
} from './gpu-primitive-test-utils';

const STORAGE_USAGE = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;

it('GPUScan matches a CPU reference across sizes, modes and strategies', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  for (const length of [1, 2, 255, 256, 257, 511, 513, 65536, 65537, 70001]) {
    const values = makePseudoRandomUint32(length, length).map(value => value >>> 20);
    for (const mode of ['exclusive', 'inclusive'] as const) {
      const expected = getCpuScan(values, mode);
      for (const [name, overrides] of [
        ['subgroup', {}],
        ['portable', {disableSubgroups: true}]
      ] as const) {
        expect(
          await runScan(device, values, mode, overrides),
          `${length} rows ${mode} (${name})`
        ).toEqual(expected);
      }
    }
  }
});

it('GPUScan plans bounded multi-dimensional dispatches', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // 70001 rows need 274 workgroups; a limit of 8 per dimension forces a 3D layout.
  const values = makePseudoRandomUint32(70001, 3).map(value => value >>> 20);
  expect(await runScan(device, values, 'exclusive', {maxComputeWorkgroupsPerDimension: 8})).toEqual(
    getCpuScan(values, 'exclusive')
  );
});

it('GPUScan handles more rows than 65535 workgroups of 256', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const length = 65535 * 256 + 1000;
  if (length * 4 > device.limits.maxStorageBufferBindingSize) return;
  const values = new Uint32Array(length).fill(1);
  const result = await runScan(device, values, 'inclusive');
  // Prefix sums of ones are the 1-based row numbers; spot check every boundary region.
  for (const row of [0, 255, 256, 65535, 65536, 65535 * 256 - 1, 65535 * 256, length - 1]) {
    expect(result[row], `row ${row}`).toBe(row + 1);
  }
}, 120000);

it('GPUCompaction is stable for non-power-of-two sizes and both subgroup strategies', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  for (const length of [1, 3, 255, 256, 257, 1000, 65537]) {
    const values = makePseudoRandomUint32(length, length + 11);
    // Flags use arbitrary nonzero values to confirm any nonzero entry selects its row.
    const flags = makePseudoRandomUint32(length, length + 29).map(value =>
      value % 3 === 0 ? 0 : value
    );
    const expected = values.filter((_, index) => flags[index] !== 0);
    for (const [name, overrides] of [
      ['subgroup', {}],
      ['portable', {disableSubgroups: true}]
    ] as const) {
      for (const strategy of ['scan', 'block'] as const) {
        const result = await runCompaction(device, values, flags, overrides, strategy);
        expect(result.count, `${length} rows count (${name}, ${strategy})`).toBe(expected.length);
        expect(
          result.values.slice(0, expected.length),
          `${length} rows (${name}, ${strategy})`
        ).toEqual(Array.from(expected));
      }
    }
  }
});

it('GPUCompaction block path covers block boundaries, empty and full selections', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // 1,024 rows per block: one dispatch up to 1,024 rows, a block-count scan beyond, and a
  // multi-level block-count scan past 4,096 blocks.
  for (const length of [1023, 1024, 1025, 2048, 5000, 4096 * 1024 + 3]) {
    const values = makePseudoRandomUint32(length, length + 1);
    for (const [name, flags] of [
      ['none', new Uint32Array(length)],
      ['all', new Uint32Array(length).fill(7)],
      ['sparse', Uint32Array.from({length}, (_, index) => (index % 997 === 3 ? 1 : 0))],
      ['dense', Uint32Array.from({length}, (_, index) => (index % 5 === 0 ? 0 : 0x80000000))]
    ] as const) {
      const expected = values.filter((_, index) => flags[index] !== 0);
      for (const disableSubgroups of [false, true]) {
        const result = await runCompaction(device, values, flags, {disableSubgroups}, 'block');
        const label = `${length} rows ${name} subgroups off: ${disableSubgroups}`;
        expect(result.count, `${label} count`).toBe(expected.length);
        expect(result.values.slice(0, expected.length), label).toEqual(Array.from(expected));
      }
    }
  }
}, 120000);

it('GPUCompaction defaults to the block strategy and honours the scan algorithm option', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const probe = new GPUCommandGraph(device);
  const view = importUint32View(
    probe,
    'probe',
    device.createBuffer({byteLength: 16, usage: STORAGE_USAGE}),
    4
  );
  const defaultCompaction = new GPUCompaction({
    input: view,
    flags: view,
    output: view,
    count: view
  });
  expect(defaultCompaction.strategy).toBe('block');
  for (const length of [1, 1500, 100001]) {
    const values = makePseudoRandomUint32(length, length + 11);
    const flags = makePseudoRandomUint32(length, length + 12).map(value => value % 3);
    const expected = values.filter((_, index) => flags[index] !== 0);
    for (const scanAlgorithm of ['multi-pass', 'decoupled-lookback'] as const) {
      const result = await runCompaction(device, values, flags, {}, 'scan', scanAlgorithm);
      expect(result.count, `${length} rows ${scanAlgorithm} count`).toBe(expected.length);
      expect(result.values.slice(0, expected.length), `${length} rows ${scanAlgorithm}`).toEqual(
        Array.from(expected)
      );
    }
  }
});

it('GPUCompaction handles inputs above a bounded dispatch limit', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const values = makePseudoRandomUint32(70001, 5);
  const flags = makePseudoRandomUint32(70001, 6).map(value => value & 1);
  const expected = values.filter((_, index) => flags[index] !== 0);
  for (const strategy of ['scan', 'block'] as const) {
    const result = await runCompaction(
      device,
      values,
      flags,
      {maxComputeWorkgroupsPerDimension: 8},
      strategy
    );
    expect(result.count, strategy).toBe(expected.length);
    expect(result.values.slice(0, expected.length), strategy).toEqual(Array.from(expected));
  }
});

function getCpuScan(values: Uint32Array, mode: 'exclusive' | 'inclusive'): number[] {
  const result: number[] = [];
  let sum = 0;
  for (const value of values) {
    result.push(mode === 'inclusive' ? (sum + value) >>> 0 : sum);
    sum = (sum + value) >>> 0;
  }
  return result;
}

async function runScan(
  device: Device,
  values: Uint32Array,
  mode: 'exclusive' | 'inclusive',
  overrides: CapabilityOverrides = {}
): Promise<number[]> {
  return withDeviceCapabilities(device, overrides, async () => {
    const inputBuffer = device.createBuffer({data: values, usage: STORAGE_USAGE});
    const outputBuffer = device.createBuffer({byteLength: values.byteLength, usage: STORAGE_USAGE});
    const graph = new GPUCommandGraph(device);
    graph.add(
      new GPUScan({
        input: importUint32View(graph, 'input', inputBuffer, values.length),
        output: importUint32View(graph, 'output', outputBuffer, values.length),
        mode
      })
    );
    try {
      return Array.from(await executeGraph(device, graph, outputBuffer, values.length));
    } finally {
      inputBuffer.destroy();
      outputBuffer.destroy();
    }
  });
}

async function runCompaction(
  device: Device,
  values: Uint32Array,
  flags: Uint32Array,
  overrides: CapabilityOverrides = {},
  strategy: 'scan' | 'block' = 'scan',
  scanAlgorithm?: 'multi-pass' | 'decoupled-lookback'
): Promise<{count: number; values: number[]}> {
  return withDeviceCapabilities(device, overrides, async () => {
    const inputBuffer = device.createBuffer({data: values, usage: STORAGE_USAGE});
    const flagBuffer = device.createBuffer({data: flags, usage: STORAGE_USAGE});
    const outputBuffer = device.createBuffer({byteLength: values.byteLength, usage: STORAGE_USAGE});
    const countBuffer = device.createBuffer({byteLength: 4, usage: STORAGE_USAGE});
    const graph = new GPUCommandGraph(device);
    graph.add(
      new GPUCompaction({
        input: importUint32View(graph, 'input', inputBuffer, values.length),
        flags: importUint32View(graph, 'flags', flagBuffer, flags.length),
        output: importUint32View(graph, 'output', outputBuffer, values.length),
        count: importUint32View(graph, 'count', countBuffer, 1),
        strategy,
        scanAlgorithm
      })
    );
    try {
      const count = (await executeGraph(device, graph, countBuffer, 1))[0];
      // The compacted values land in the caller-owned output buffer by the time the count reads.
      const bytes = await outputBuffer.readAsync();
      return {
        count,
        values: Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, values.length))
      };
    } finally {
      inputBuffer.destroy();
      flagBuffer.destroy();
      outputBuffer.destroy();
      countBuffer.destroy();
    }
  });
}

function importUint32View(graph: GPUCommandGraph, id: string, buffer: Buffer, length: number) {
  return graph.createDataView(
    graph.importBuffer({id, byteLength: buffer.byteLength, usage: buffer.usage}, buffer),
    {format: 'uint32', length}
  );
}

async function executeGraph(
  device: Device,
  graph: GPUCommandGraph,
  readBuffer: Buffer,
  readLength: number
): Promise<Uint32Array> {
  const compiled = graph.compile();
  try {
    const commandEncoder = device.createCommandEncoder({id: 'scan-compaction-test'});
    compiled.encode(commandEncoder, {parameters: undefined});
    device.submit(commandEncoder.finish());
    const bytes = await readBuffer.readAsync();
    return new Uint32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + readLength * 4));
  } finally {
    compiled.destroy();
  }
}
