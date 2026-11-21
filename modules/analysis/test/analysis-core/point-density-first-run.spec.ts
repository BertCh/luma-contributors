// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Vitest scans its test entries before running them. Make the iframe's public dependencies
// visible to that scan so opening the iframe cannot trigger dependency optimization/reloads.
// These imports create no device. The example itself runs only in the independent iframe,
// which still verifies that its adapter registry starts empty.
import {Buffer, luma} from '@luma.gl/core';
import {webgpuAdapter} from '@luma.gl/core/webgpu';
import {GPUCommandGraph} from '@luma.gl/graph';
import {GPUPointDensity} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../src/testing/require-test-device';

type FirstRunMessage = {
  type: 'first-density-result';
  baselineAdapterMissing: boolean;
  missingAdapterRejected: boolean;
  displayed: [number[], number][];
  error?: string;
};

it('executes the displayed program in a fresh browser realm without a prepared device', async () => {
  requireTestDevice(navigator.gpu);
  const frame = document.createElement('iframe');
  frame.src = '/modules/analysis/test/analysis-core/point-density-first-run.html';
  let timeout: ReturnType<typeof setTimeout> | undefined;
  let receive: (event: MessageEvent) => void = () => {};
  try {
    const result = new Promise<FirstRunMessage>((resolve, reject) => {
      receive = event => {
        if (event.source === frame.contentWindow && event.data?.type === 'first-density-result') {
          resolve(event.data);
        }
      };
      window.addEventListener('message', receive);
      timeout = setTimeout(() => reject(new Error('Fresh first-run frame did not finish')), 30_000);
    });
    document.body.append(frame);
    const message = await result;
    expect(message.error).toBeUndefined();
    expect(message.baselineAdapterMissing).toBe(true);
    expect(message.missingAdapterRejected).toBe(true);
    expect(message.displayed).toEqual([[[2, 0, 1, 1], 0]]);
  } finally {
    clearTimeout(timeout);
    window.removeEventListener('message', receive);
    frame.remove();
  }
});

it('FST-11 reads named density outputs after an unrelated allocation', async () => {
  requireTestDevice(navigator.gpu);
  const device = await luma.createDevice({type: 'webgpu', adapters: [webgpuAdapter]});
  const buffers: Buffer[] = [];
  let compiled: Awaited<ReturnType<GPUCommandGraph['compileAsync']>> | undefined;
  try {
    const graph = new GPUCommandGraph(device, {id: 'density-allocation-control'});
    const allocate = (id: string, data: Float32Array | Uint32Array) => {
      const buffer = device.createBuffer({
        id,
        data,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      buffers.push(buffer);
      const handle = graph.importBuffer(
        {id, byteLength: buffer.byteLength, usage: buffer.usage},
        buffer
      );
      return {buffer, handle};
    };
    const positions = graph.createDataView(
      allocate('positions', Float32Array.of(1.25, 0.25, 1.75, 0.75, 0.25, 1.25, 1.25, 1.25, 3, 3))
        .handle,
      {format: 'float32x2', length: 5}
    );
    const bounds = graph.createDataView(allocate('bounds', Float32Array.of(0, 0, 2, 2)).handle, {
      format: 'float32',
      length: 4
    });
    allocate('unrelated-weights', Float32Array.of(1, 1, 1, 1));
    const valuesAllocation = allocate('values', new Float32Array(4).fill(-1));
    const overflowAllocation = allocate('overflow', new Uint32Array(1).fill(99));
    graph.add(
      new GPUPointDensity({
        positions,
        bounds,
        gridSize: [2, 2],
        statistic: 'count',
        output: {
          values: graph.createDataView(valuesAllocation.handle, {format: 'float32', length: 4}),
          status: {
            overflow: graph.createDataView(overflowAllocation.handle, {format: 'uint32', length: 1})
          }
        }
      })
    );
    compiled = await graph.compileAsync();
    const encoder = device.createCommandEncoder();
    compiled.encode(encoder, {parameters: undefined});
    device.submit(encoder.finish());
    const valueBytes = await valuesAllocation.buffer.readAsync();
    const overflowBytes = await overflowAllocation.buffer.readAsync();
    // Hand-binned, X fastest: no southwest points, two southeast, one in each north cell.
    expect(Array.from(new Float32Array(valueBytes.buffer, valueBytes.byteOffset, 4))).toEqual([
      0, 2, 1, 1
    ]);
    expect(new Uint32Array(overflowBytes.buffer, overflowBytes.byteOffset, 1)[0]).toBe(0);
  } finally {
    compiled?.destroy();
    for (const buffer of buffers) buffer.destroy();
    device.destroy();
  }
});
