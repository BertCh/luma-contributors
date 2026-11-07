// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GPUReduction,
  type GPUReductionOperation
} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  withDeviceCapabilities,
  makePseudoRandomUint32,
  type CapabilityOverrides
} from './gpu-primitive-test-utils';

type Format = 'uint32' | 'sint32' | 'float32';

it('GPUReduction matches a CPU reference across sizes and strategies', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const portableOverrides = {disableSubgroups: true};
  // Includes non-powers of two, level boundaries (256, 65536) and the multi-row-per-thread regime.
  for (const length of [1, 2, 255, 256, 257, 1000, 65536, 65537, 262147]) {
    const values = makePseudoRandomUint32(length, length).map(value => value >>> 8);
    const expectedSum = values.reduce((sum, value) => (sum + value) >>> 0, 0);
    const expectedMinimum = values.reduce((a, b) => Math.min(a, b), 0xffffffff);
    const expectedMaximum = values.reduce((a, b) => Math.max(a, b), 0);
    for (const [name, target] of [
      ['subgroup', undefined],
      ['portable', portableOverrides]
    ] as const) {
      const label = `${length} rows (${name})`;
      expect(await runReduction(device, values, 'uint32', 'sum', target), `${label} sum`).toEqual([
        expectedSum
      ]);
      expect(await runReduction(device, values, 'uint32', 'min', target), `${label} min`).toEqual([
        expectedMinimum
      ]);
      expect(await runReduction(device, values, 'uint32', 'max', target), `${label} max`).toEqual([
        expectedMaximum
      ]);
      expect(
        await runReduction(device, values, 'uint32', 'extent', target),
        `${label} extent`
      ).toEqual([expectedMinimum, expectedMaximum]);
    }
  }
});

it('GPUReduction handles inputs above the per-dimension dispatch limit', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // A limit of 5 workgroups per dimension forces a 3D layout for 70000 rows (69 workgroups).
  const limited = {maxComputeWorkgroupsPerDimension: 5};
  const values = makePseudoRandomUint32(70001, 7).map(value => value >>> 8);
  const expectedSum = values.reduce((sum, value) => (sum + value) >>> 0, 0);
  expect(await runReduction(device, values, 'uint32', 'sum', limited)).toEqual([expectedSum]);
  expect(await runReduction(device, values, 'uint32', 'max', limited)).toEqual([
    values.reduce((a, b) => Math.max(a, b), 0)
  ]);
});

it('GPUReduction compensated sums stay accurate where float32 summation drifts', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // One large value followed by many small ones: naive float32 accumulation drops every small term.
  const length = 100000;
  const values = new Float32Array(length);
  values[0] = 16777216;
  values.fill(1, 1);
  const exact = 16777216 + (length - 1);
  const plain = (await runReduction(device, values, 'float32', 'sum'))[0];
  const compensated = [
    (await runReduction(device, values, 'float32', 'sum', undefined, true))[0],
    (await runReduction(device, values, 'float32', 'sum', {disableSubgroups: true}, true))[0]
  ];
  for (const result of compensated) {
    // float32 can only represent exact + 0 or +/-2 here, so allow one unit in the last place.
    expect(Math.abs(result - exact), 'compensated error is at most one ulp').toBeLessThanOrEqual(2);
    expect(Math.abs(result - exact)).toBeLessThanOrEqual(Math.abs(plain - exact));
  }
});

it('GPUReduction compensated sums preserve infinities and reject non-float32 formats', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  expect(
    await runReduction(
      device,
      Float32Array.from([1, Infinity, 2]),
      'float32',
      'sum',
      undefined,
      true
    )
  ).toEqual([Infinity]);
  expect(
    await runReduction(device, Float32Array.from([1.5, 2.5, -1]), 'float32', 'sum', undefined, true)
  ).toEqual([3]);
  const graph = new GPUCommandGraph(device);
  const handle = graph.createTransientBuffer({id: 'a', byteLength: 16, usage: Buffer.STORAGE});
  const out = graph.createTransientBuffer({id: 'b', byteLength: 4, usage: Buffer.STORAGE});
  expect(
    () =>
      new GPUReduction({
        input: graph.createDataView(handle, {format: 'uint32', length: 4}),
        output: graph.createDataView(out, {format: 'uint32', length: 1}),
        operation: 'sum',
        compensated: true
      })
  ).toThrow(/float32 sum/);
});

async function runReduction(
  device: Device,
  values: Uint32Array | Int32Array | Float32Array,
  format: Format,
  operation: GPUReductionOperation,
  overrides: CapabilityOverrides = {},
  compensated: boolean = false
): Promise<number[]> {
  return withDeviceCapabilities(device, overrides, () =>
    executeReduction(device, values, format, operation, compensated)
  );
}

async function executeReduction(
  device: Device,
  values: Uint32Array | Int32Array | Float32Array,
  format: Format,
  operation: GPUReductionOperation,
  compensated: boolean
): Promise<number[]> {
  const outputLength = operation === 'extent' ? 2 : 1;
  const inputBuffer = device.createBuffer({data: values, usage: Buffer.STORAGE | Buffer.COPY_DST});
  const outputBuffer = device.createBuffer({
    byteLength: outputLength * 4,
    usage: Buffer.STORAGE | Buffer.COPY_SRC
  });
  const graph = new GPUCommandGraph(device);
  const importView = (id: string, buffer: Buffer, length: number) =>
    graph.createDataView(
      graph.importBuffer({id, byteLength: buffer.byteLength, usage: buffer.usage}, buffer),
      {format, length}
    );
  graph.add(
    new GPUReduction({
      input: importView('input', inputBuffer, values.length),
      output: importView('output', outputBuffer, outputLength),
      operation,
      compensated
    })
  );
  const compiled = graph.compile();
  try {
    const commandEncoder = device.createCommandEncoder({id: 'reduction-test'});
    compiled.encode(commandEncoder, {parameters: undefined});
    device.submit(commandEncoder.finish());
    const bytes = await outputBuffer.readAsync();
    const typed =
      format === 'uint32'
        ? new Uint32Array(bytes.buffer, bytes.byteOffset, outputLength)
        : format === 'sint32'
          ? new Int32Array(bytes.buffer, bytes.byteOffset, outputLength)
          : new Float32Array(bytes.buffer, bytes.byteOffset, outputLength);
    return Array.from(typed);
  } finally {
    compiled.destroy();
    inputBuffer.destroy();
    outputBuffer.destroy();
  }
}
