// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * PLAT-01. Runs each published primitive (`GPUScan`, `GPUSort`, `GPUCompaction`, `GPUReduction`)
 * above 4,194,240 rows, the one-dimensional dispatch product (65,535 workgroups of 64), and checks
 * every output row against a CPU computation. This settles the "bounded dispatch above about 4.2M
 * rows" entry in `dev-docs/upstream-workarounds.md`. A row count the device cannot hold is skipped
 * with a logged reason, so a skip is never a pass for that size.
 */

import {Buffer, type Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GPUCompaction,
  GPUReduction,
  GPUScan,
  GPUSort,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {requireTestDevice} from '../utils/require-test-device';
import {makePseudoRandomUint32} from './gpu-primitive-test-utils';

const STORAGE_USAGE = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
const ONE_DIMENSIONAL_DISPATCH_ROWS = 65535 * 64;
const ROW_COUNTS = [ONE_DIMENSIONAL_DISPATCH_ROWS + 1, 1 << 24, 1 << 25];

function view(graph: GPUCommandGraph, buffer: Buffer, length = buffer.byteLength / 4) {
  return graph.createDataView(
    graph.importBuffer(
      {id: `buffer-${buffer.id}`, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    ),
    {format: 'uint32', length}
  );
}

function getSkipReason(device: Device, rowCount: number): string | undefined {
  const byteLength = rowCount * 4;
  const {maxStorageBufferBindingSize, maxBufferSize} = device.limits;
  if (byteLength > maxStorageBufferBindingSize || byteLength > maxBufferSize) {
    return `${rowCount} rows need ${byteLength} bytes; maxStorageBufferBindingSize=${maxStorageBufferBindingSize}, maxBufferSize=${maxBufferSize}`;
  }
  return undefined;
}

async function runGraph(
  device: Device,
  buffers: Buffer[],
  makeProducer: (graph: GPUCommandGraph) => GPUCommandNodeProducer
): Promise<void> {
  const graph = new GPUCommandGraph(device);
  graph.add(makeProducer(graph));
  const compiled = graph.compile();
  try {
    const commandEncoder = device.createCommandEncoder({id: 'plat01'});
    compiled.encode(commandEncoder, {parameters: undefined});
    device.submit(commandEncoder.finish());
    await buffers[buffers.length - 1].readAsync(0, 4);
  } finally {
    compiled.destroy();
  }
}

async function readUint32(buffer: Buffer, rowCount: number): Promise<Uint32Array> {
  const bytes = await buffer.readAsync(0, rowCount * 4);
  return new Uint32Array(bytes.buffer, bytes.byteOffset, rowCount);
}

describe('published primitives above the one-dimensional dispatch product (PLAT-01)', () => {
  it('records the device', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    console.log(
      `PLAT01_DEVICE ${JSON.stringify({
        vendor: device.info.vendor,
        renderer: device.info.renderer,
        type: device.info.type,
        maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
        maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
        maxBufferSize: device.limits.maxBufferSize,
        maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
        maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
        subgroups: device.features.has('subgroups')
      })}`
    );
  });

  for (const rowCount of ROW_COUNTS) {
    describe(`${rowCount} rows`, () => {
      it('GPUScan matches a CPU exclusive prefix sum', {timeout: 300000}, async context => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const skipReason = getSkipReason(device, rowCount);
        if (skipReason) {
          console.log(`PLAT01_SKIPPED GPUScan ${skipReason}`);
          return context.skip();
        }
        const values = Uint32Array.from({length: rowCount}, (_, row) => (row * 7) % 5);
        const buffers = [
          device.createBuffer({data: values, usage: STORAGE_USAGE}),
          device.createBuffer({byteLength: rowCount * 4, usage: STORAGE_USAGE})
        ];
        try {
          await runGraph(
            device,
            buffers,
            graph => new GPUScan({input: view(graph, buffers[0]), output: view(graph, buffers[1])})
          );
          const actual = await readUint32(buffers[1], rowCount);
          let running = 0;
          let firstMismatch = -1;
          for (let row = 0; row < rowCount && firstMismatch < 0; row++) {
            if (actual[row] !== running) firstMismatch = row;
            running = (running + values[row]) >>> 0;
          }
          expect(firstMismatch).toBe(-1);
        } finally {
          buffers.forEach(buffer => buffer.destroy());
        }
      });

      it('GPUCompaction keeps flagged rows in order', {timeout: 300000}, async context => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const skipReason = getSkipReason(device, rowCount);
        if (skipReason) {
          console.log(`PLAT01_SKIPPED GPUCompaction ${skipReason}`);
          return context.skip();
        }
        const values = makePseudoRandomUint32(rowCount, 3);
        const flags = Uint32Array.from({length: rowCount}, (_, row) => (row % 3 === 0 ? 1 : 0));
        const buffers = [
          device.createBuffer({data: values, usage: STORAGE_USAGE}),
          device.createBuffer({data: flags, usage: STORAGE_USAGE}),
          device.createBuffer({byteLength: rowCount * 4, usage: STORAGE_USAGE}),
          device.createBuffer({byteLength: 4, usage: STORAGE_USAGE})
        ];
        try {
          await runGraph(
            device,
            buffers,
            graph =>
              new GPUCompaction({
                input: view(graph, buffers[0]),
                flags: view(graph, buffers[1]),
                output: view(graph, buffers[2]),
                count: view(graph, buffers[3], 1)
              })
          );
          const expected = values.filter((_, row) => flags[row] !== 0);
          const count = (await readUint32(buffers[3], 1))[0];
          expect(count).toBe(expected.length);
          const actual = await readUint32(buffers[2], expected.length);
          let firstMismatch = -1;
          for (let row = 0; row < expected.length && firstMismatch < 0; row++) {
            if (actual[row] !== expected[row]) firstMismatch = row;
          }
          expect(firstMismatch).toBe(-1);
        } finally {
          buffers.forEach(buffer => buffer.destroy());
        }
      });

      it('GPUReduction matches CPU sum, min and max', {timeout: 300000}, async context => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const skipReason = getSkipReason(device, rowCount);
        if (skipReason) {
          console.log(`PLAT01_SKIPPED GPUReduction ${skipReason}`);
          return context.skip();
        }
        const values = Uint32Array.from(
          {length: rowCount},
          (_, row) => 1 + ((Math.imul(row, 2654435761) >>> 0) % 1000)
        );
        let expectedSum = 0;
        let expectedMin = Infinity;
        let expectedMax = -Infinity;
        for (const value of values) {
          expectedSum = (expectedSum + value) >>> 0;
          expectedMin = Math.min(expectedMin, value);
          expectedMax = Math.max(expectedMax, value);
        }
        const input = device.createBuffer({data: values, usage: STORAGE_USAGE});
        const outputs = {
          sum: device.createBuffer({byteLength: 4, usage: STORAGE_USAGE}),
          extent: device.createBuffer({byteLength: 8, usage: STORAGE_USAGE})
        };
        try {
          await runGraph(
            device,
            [input, outputs.sum],
            graph =>
              new GPUReduction({
                input: view(graph, input),
                output: view(graph, outputs.sum, 1),
                operation: 'sum'
              })
          );
          expect((await readUint32(outputs.sum, 1))[0]).toBe(expectedSum);
          await runGraph(
            device,
            [input, outputs.extent],
            graph =>
              new GPUReduction({
                input: view(graph, input),
                output: view(graph, outputs.extent, 2),
                operation: 'extent'
              })
          );
          expect(Array.from(await readUint32(outputs.extent, 2))).toEqual([
            expectedMin,
            expectedMax
          ]);
        } finally {
          input.destroy();
          outputs.sum.destroy();
          outputs.extent.destroy();
        }
      });

      it('GPUSort orders keys and carries values stably', {timeout: 600000}, async context => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const skipReason = getSkipReason(device, rowCount);
        if (skipReason) {
          console.log(`PLAT01_SKIPPED GPUSort ${skipReason}`);
          return context.skip();
        }
        // Few distinct keys force many ties, which exercises stability.
        const keys = makePseudoRandomUint32(rowCount, 5).map(value => value % 65536);
        const rowIndices = Uint32Array.from({length: rowCount}, (_, row) => row);
        const buffers = [
          device.createBuffer({data: keys, usage: STORAGE_USAGE}),
          device.createBuffer({data: rowIndices, usage: STORAGE_USAGE}),
          device.createBuffer({byteLength: rowCount * 4, usage: STORAGE_USAGE}),
          device.createBuffer({byteLength: rowCount * 4, usage: STORAGE_USAGE})
        ];
        try {
          await runGraph(
            device,
            buffers,
            graph =>
              new GPUSort({
                keys: view(graph, buffers[0]),
                values: view(graph, buffers[1]),
                outputKeys: view(graph, buffers[2]),
                outputValues: view(graph, buffers[3]),
                keyBits: 16
              })
          );
          const expectedKeys = Uint32Array.from(keys).sort();
          const actualKeys = await readUint32(buffers[2], rowCount);
          const actualValues = await readUint32(buffers[3], rowCount);
          let keyMismatch = -1;
          let pairingMismatch = -1;
          let stabilityMismatch = -1;
          const seen = new Uint8Array(rowCount);
          let duplicateValue = -1;
          for (let row = 0; row < rowCount; row++) {
            if (keyMismatch < 0 && actualKeys[row] !== expectedKeys[row]) keyMismatch = row;
            const source = actualValues[row];
            if (source >= rowCount || keys[source] !== actualKeys[row]) {
              if (pairingMismatch < 0) pairingMismatch = row;
            } else if (seen[source]++ && duplicateValue < 0) {
              duplicateValue = row;
            }
            if (
              stabilityMismatch < 0 &&
              row > 0 &&
              actualKeys[row] === actualKeys[row - 1] &&
              actualValues[row] <= actualValues[row - 1]
            ) {
              stabilityMismatch = row;
            }
          }
          expect({keyMismatch, pairingMismatch, duplicateValue, stabilityMismatch}).toEqual({
            keyMismatch: -1,
            pairingMismatch: -1,
            duplicateValue: -1,
            stabilityMismatch: -1
          });
        } finally {
          buffers.forEach(buffer => buffer.destroy());
        }
      });
    });
  }
});
