// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, GPUSort, type GPUSortProps} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  makePseudoRandomUint32,
  withDeviceCapabilities,
  type CapabilityOverrides
} from './gpu-primitive-test-utils';

type SortOptions = Pick<GPUSortProps, 'algorithm' | 'direction' | 'keyBits' | 'digitBits'> & {
  elementsPerThread?: number;
};

it('GPUSort rejects non-power-of-two elementsPerThread at construction', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const graph = new GPUCommandGraph(device);
  const makeView = (id: string) =>
    graph.createDataView(graph.createTransientBuffer({id, byteLength: 64, usage: Buffer.STORAGE}), {
      format: 'uint32',
      length: 16
    });
  const props = (elementsPerThread: number) => ({
    keys: makeView(`keys-${elementsPerThread}`),
    values: makeView(`values-${elementsPerThread}`),
    outputKeys: makeView(`output-keys-${elementsPerThread}`),
    outputValues: makeView(`output-values-${elementsPerThread}`),
    elementsPerThread
  });
  for (const invalid of [0, 3, 5, 6, 7, 9, 33, 64, 1.5]) {
    expect(() => new GPUSort(props(invalid)), `elementsPerThread ${invalid}`).toThrow(
      /elementsPerThread must be a power of two/
    );
  }
  for (const valid of [1, 2, 4, 8, 16, 32]) {
    expect(() => new GPUSort(props(valid)), `elementsPerThread ${valid}`).not.toThrow();
  }
});

it('GPUSort radix and bitonic agree with a stable CPU sort across sizes and strategies', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // Non-powers of two and tile boundaries (256 * 8 = 2048 keys per radix workgroup).
  for (const length of [2, 255, 257, 1000, 2047, 2048, 2049, 5000]) {
    // Few distinct keys force many ties, which exposes any instability.
    const keys = makePseudoRandomUint32(length, length).map(value => value % 97);
    const values = Uint32Array.from({length}, (_, index) => index);
    for (const direction of ['ascending', 'descending'] as const) {
      const expected = getStableSortedPairs(keys, values, direction);
      for (const [name, overrides] of [
        ['subgroup', {}],
        ['portable', {disableSubgroups: true}]
      ] as const) {
        for (const algorithm of length <= 2048
          ? (['bitonic', 'radix'] as const)
          : (['radix'] as const)) {
          const result = await runSort(device, keys, values, {algorithm, direction}, overrides);
          expect(result, `${length} rows ${algorithm} ${direction} (${name})`).toEqual(expected);
        }
      }
    }
  }
});

it('GPUSort supports eight-bit digits, narrow keys and bounded dispatch limits', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const length = 9000;
  const keys = makePseudoRandomUint32(length, 41);
  const values = Uint32Array.from({length}, (_, index) => index);
  const expected = getStableSortedPairs(keys, values, 'ascending');
  expect(
    await runSort(device, keys, values, {algorithm: 'radix', digitBits: 8}),
    'eight-bit digits'
  ).toEqual(expected);
  expect(
    await runSort(
      device,
      keys,
      values,
      {algorithm: 'radix', elementsPerThread: 1},
      {maxComputeWorkgroupsPerDimension: 8}
    ),
    'one key per thread under a three-dimensional dispatch (36 workgroups)'
  ).toEqual(expected);

  const narrowKeys = keys.map(value => value & 0xffff);
  expect(
    await runSort(device, narrowKeys, values, {algorithm: 'radix', keyBits: 16}),
    '16-bit keys need only the low 16 bits'
  ).toEqual(getStableSortedPairs(narrowKeys, values, 'ascending'));
});

function getStableSortedPairs(
  keys: Uint32Array,
  values: Uint32Array,
  direction: 'ascending' | 'descending'
): {keys: number[]; values: number[]} {
  const order = Array.from(keys.keys()).sort((left, right) =>
    keys[left] === keys[right]
      ? left - right
      : direction === 'ascending'
        ? keys[left] - keys[right]
        : keys[right] - keys[left]
  );
  return {keys: order.map(index => keys[index]), values: order.map(index => values[index])};
}

async function runSort(
  device: Device,
  keys: Uint32Array,
  values: Uint32Array,
  options: SortOptions,
  overrides: CapabilityOverrides = {}
): Promise<{keys: number[]; values: number[]}> {
  return withDeviceCapabilities(device, overrides, async () => {
    const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
    const buffers = {
      keys: device.createBuffer({data: keys, usage}),
      values: device.createBuffer({data: values, usage}),
      outputKeys: device.createBuffer({byteLength: keys.byteLength, usage}),
      outputValues: device.createBuffer({byteLength: values.byteLength, usage})
    };
    const graph = new GPUCommandGraph(device);
    const importView = (name: keyof typeof buffers) =>
      graph.createDataView(
        graph.importBuffer(
          {id: name, byteLength: buffers[name].byteLength, usage: buffers[name].usage},
          buffers[name]
        ),
        {format: 'uint32', length: keys.length}
      );
    graph.add(
      new GPUSort({
        keys: importView('keys'),
        values: importView('values'),
        outputKeys: importView('outputKeys'),
        outputValues: importView('outputValues'),
        ...options
      })
    );
    const compiled = graph.compile();
    try {
      const commandEncoder = device.createCommandEncoder({id: 'sort-parity-test'});
      compiled.encode(commandEncoder, {parameters: undefined});
      device.submit(commandEncoder.finish());
      const [keyBytes, valueBytes] = await Promise.all([
        buffers.outputKeys.readAsync(),
        buffers.outputValues.readAsync()
      ]);
      return {
        keys: Array.from(new Uint32Array(keyBytes.buffer, keyBytes.byteOffset, keys.length)),
        values: Array.from(new Uint32Array(valueBytes.buffer, valueBytes.byteOffset, keys.length))
      };
    } finally {
      compiled.destroy();
      for (const buffer of Object.values(buffers)) buffer.destroy();
    }
  });
}
