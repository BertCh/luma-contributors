// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {Buffer, type Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {MeteredCommandGraph} from '../src/session/metered-command-graph';

/** Two copy passes through one 1 KiB transient buffer, so the compiled graph owns scratch. */
function addTransientRoundTrip(graph: MeteredCommandGraph, device: Device): Buffer[] {
  const source = device.createBuffer({
    byteLength: 1024,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const destination = device.createBuffer({
    byteLength: 1024,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const sourceHandle = graph.importBuffer({id: 'source', byteLength: 1024, usage}, source);
  const destinationHandle = graph.importBuffer(
    {id: 'destination', byteLength: 1024, usage},
    destination
  );
  const scratch = graph.createTransientBuffer({id: 'scratch', byteLength: 1024, usage});
  graph.addCopyPass({
    id: 'into-scratch',
    resources: [
      {buffer: sourceHandle, usage: 'copy-source'},
      {buffer: scratch, usage: 'copy-destination'}
    ],
    compile: () => ({
      encode: context =>
        context.commandEncoder.copyBufferToBuffer({
          sourceBuffer: context.getBuffer(sourceHandle),
          destinationBuffer: context.getBuffer(scratch),
          size: 1024
        })
    })
  });
  graph.addCopyPass({
    id: 'out-of-scratch',
    resources: [
      {buffer: scratch, usage: 'copy-source'},
      {buffer: destinationHandle, usage: 'copy-destination'}
    ],
    compile: () => ({
      encode: context =>
        context.commandEncoder.copyBufferToBuffer({
          sourceBuffer: context.getBuffer(scratch),
          destinationBuffer: context.getBuffer(destinationHandle),
          size: 1024
        })
    })
  });
  return [source, destination];
}

test('metered graph records the same transient bytes from compile and compileAsync', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const synchronousGraph = new MeteredCommandGraph(device, {id: 'metered-synchronous'});
  const asynchronousGraph = new MeteredCommandGraph(device, {id: 'metered-asynchronous'});
  const buffers = [
    ...addTransientRoundTrip(synchronousGraph, device),
    ...addTransientRoundTrip(asynchronousGraph, device)
  ];
  expect(asynchronousGraph.transientByteLength).toBe(0);
  const synchronous = synchronousGraph.compile();
  const asynchronous = await asynchronousGraph.compileAsync();
  try {
    expect(synchronousGraph.transientByteLength).toBeGreaterThanOrEqual(1024);
    expect(asynchronousGraph.transientByteLength).toBe(synchronousGraph.transientByteLength);
    expect(asynchronousGraph.transientByteLength).toBe(
      asynchronous.stats.physicalTransientResourceBytes
    );
  } finally {
    synchronous.destroy();
    asynchronous.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
