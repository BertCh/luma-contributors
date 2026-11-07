// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, GPUSort, type GPUSortProps} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {makePseudoRandomUint32, withDeviceCapabilities} from './gpu-primitive-test-utils';

const STORAGE_USAGE = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;

type SortResult = {keys: number[]; values: number[]};

describe('GPUSort onesweep', () => {
  it('matches a stable CPU sort across sizes, digit widths and directions', async () => {
    const device = await getWebGPUTestDevice();
    if (!device) return;
    for (const length of [1, 2, 255, 256, 257, 2048, 2049, 10_000, 100_001, 1_048_577]) {
      const keys = makePseudoRandomUint32(length, length + 5);
      for (const direction of ['ascending', 'descending'] as const) {
        for (const digitBits of [4, 8] as const) {
          if (length > 200_000 && digitBits === 4) continue;
          const result = await runSort(device, keys, {direction, digitBits});
          const expected = getStableSort(keys, direction);
          expect(result.keys, `${length} ${direction} ${digitBits}-bit keys`).toEqual(
            expected.keys
          );
          expect(result.values, `${length} ${direction} ${digitBits}-bit stability`).toEqual(
            expected.values
          );
        }
      }
    }
  }, 180_000);

  it('sorts adversarial key distributions stably', async () => {
    const device = await getWebGPUTestDevice();
    if (!device) return;
    const length = 30_000;
    const cases: Record<string, Uint32Array> = {
      allEqual: new Uint32Array(length).fill(0x1234),
      maximum: new Uint32Array(length).fill(0xffffffff),
      ascending: Uint32Array.from({length}, (_, index) => index),
      descending: Uint32Array.from({length}, (_, index) => length - index),
      twoDigits: Uint32Array.from({length}, (_, index) => (index % 2 ? 0x80000000 : 3)),
      // One digit bucket holds almost everything, so a single look-back chain does all the work.
      skewed: Uint32Array.from({length}, (_, index) => (index % 1000 === 0 ? index : 0x5500)),
      highBytesOnly: Uint32Array.from({length}, (_, index) => (index % 251) << 24)
    };
    for (const [name, keys] of Object.entries(cases)) {
      for (const direction of ['ascending', 'descending'] as const) {
        const result = await runSort(device, keys, {direction});
        const expected = getStableSort(keys, direction);
        expect(result.keys, `${name} ${direction} keys`).toEqual(expected.keys);
        expect(result.values, `${name} ${direction} stability`).toEqual(expected.values);
      }
    }
  }, 120_000);

  it('processes only the requested key bits and honors smaller tiles', async () => {
    const device = await getWebGPUTestDevice();
    if (!device) return;
    const keys = makePseudoRandomUint32(20_000, 99);
    for (const [keyBits, digitBits, elementsPerThread] of [
      [20, 8, 8],
      [13, 8, 4],
      [10, 4, 1],
      [32, 8, 1]
    ] as const) {
      const masked = keys.map(key => (keyBits === 32 ? key : key & (2 ** keyBits - 1)));
      const result = await runSort(device, masked, {keyBits, digitBits, elementsPerThread});
      const expected = getStableSort(masked, 'ascending');
      expect(result.keys, `${keyBits} bits, ${digitBits}-bit digits`).toEqual(expected.keys);
      expect(result.values, `${keyBits} bits stability`).toEqual(expected.values);
    }
  });

  it('stays correct when every look-back falls back to recounting predecessors', async () => {
    const device = await getWebGPUTestDevice();
    if (!device) return;
    for (const length of [5000, 40_000]) {
      const keys = makePseudoRandomUint32(length, 17).map(key => key & 0xffff);
      const result = await runSort(device, keys, {onesweepMaximumSpins: 0});
      const expected = getStableSort(keys, 'ascending');
      expect(result.keys, `${length} keys`).toEqual(expected.keys);
      expect(result.values, `${length} stability`).toEqual(expected.values);
    }
  });

  it('clears look-back state when a compiled graph executes repeatedly', async () => {
    const device = await getWebGPUTestDevice();
    if (!device) return;
    const keys = makePseudoRandomUint32(50_000, 21);
    const result = await runSort(device, keys, {}, {executionCount: 4});
    const expected = getStableSort(keys, 'ascending');
    expect(result.keys).toEqual(expected.keys);
    expect(result.values).toEqual(expected.values);
  });

  it('plans bounded multi-dimensional dispatches and agrees with the radix sort', async () => {
    const device = await getWebGPUTestDevice();
    if (!device) return;
    const keys = makePseudoRandomUint32(30_000, 3);
    // 15 tiles of 2,048 keys need a 3D layout under a limit of 3 workgroups per dimension.
    const result = await runSort(device, keys, {}, {maxComputeWorkgroupsPerDimension: 3});
    const radix = await runSort(device, keys, {algorithm: 'radix'});
    expect(result).toEqual(radix);
  });
});

function getStableSort(keys: Uint32Array, direction: 'ascending' | 'descending'): SortResult {
  const indices = Array.from(keys, (_, index) => index);
  indices.sort((left, right) => {
    if (keys[left] === keys[right]) return left - right;
    return direction === 'ascending' ? keys[left] - keys[right] : keys[right] - keys[left];
  });
  return {keys: indices.map(index => keys[index]), values: indices};
}

async function runSort(
  device: Device,
  keys: Uint32Array,
  props: Partial<GPUSortProps> = {},
  options: {maxComputeWorkgroupsPerDimension?: number; executionCount?: number} = {}
): Promise<SortResult> {
  return withDeviceCapabilities(device, options, async () => {
    const length = keys.length;
    const values = Uint32Array.from(keys, (_, index) => index);
    const buffers = {
      keys: device.createBuffer({data: keys, usage: STORAGE_USAGE}),
      values: device.createBuffer({data: values, usage: STORAGE_USAGE}),
      outputKeys: device.createBuffer({byteLength: length * 4, usage: STORAGE_USAGE}),
      outputValues: device.createBuffer({byteLength: length * 4, usage: STORAGE_USAGE})
    };
    const graph = new GPUCommandGraph(device);
    const importView = (id: string, buffer: Buffer) =>
      graph.createDataView(
        graph.importBuffer({id, byteLength: buffer.byteLength, usage: buffer.usage}, buffer),
        {format: 'uint32', length}
      );
    graph.add(
      new GPUSort({
        algorithm: 'onesweep',
        ...props,
        keys: importView('keys', buffers.keys),
        values: importView('values', buffers.values),
        outputKeys: importView('outputKeys', buffers.outputKeys),
        outputValues: importView('outputValues', buffers.outputValues)
      })
    );
    const compiled = graph.compile();
    try {
      // Re-executing one compiled graph must clear the published look-back state each time.
      for (let execution = 0; execution < (options.executionCount ?? 1); execution++) {
        const commandEncoder = device.createCommandEncoder({id: 'onesweep-test'});
        compiled.encode(commandEncoder, {parameters: undefined});
        device.submit(commandEncoder.finish());
      }
      const [keyBytes, valueBytes] = await Promise.all([
        buffers.outputKeys.readAsync(),
        buffers.outputValues.readAsync()
      ]);
      return {
        keys: Array.from(new Uint32Array(keyBytes.buffer, keyBytes.byteOffset, length)),
        values: Array.from(new Uint32Array(valueBytes.buffer, valueBytes.byteOffset, length))
      };
    } finally {
      compiled.destroy();
      for (const buffer of Object.values(buffers)) buffer.destroy();
    }
  });
}
