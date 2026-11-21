// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createGPUCopyCommandNode, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {createPreparedGeometrySnapshot} from '../../../src/gpu-spatial-analysis/spatial-join/prepared-geometry-snapshot';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {readFloat32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('ERG-02 reverse submission retains each encoded geometry generation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph<{frame: 'A' | 'B'}>(device);
  const buffers: Buffer[] = [];
  const events: string[] = [];
  const allocate = (name: string, values: Float32Array) => {
    const buffer = device.createBuffer({
      data: values,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'float32x2', 1)};
  };
  // Exactly representable distinct geometry, independently expected at each capture destination.
  const sourceA = allocate('source-A', Float32Array.of(10, 20));
  const sourceB = allocate('source-B', Float32Array.of(30, 40));
  const source = allocate('source', Float32Array.of(0, 0));
  const outputA = allocate('output-A', Float32Array.of(0, 0));
  const outputB = allocate('output-B', Float32Array.of(0, 0));
  const generation = Uint32Array.of(1, 0);
  const snapshot = createPreparedGeometrySnapshot(graph, {
    id: 'reverse',
    views: {positions: source.view},
    generation,
    ownedBuffers: buffers,
    recordEncoding: () => events.push(`request:${generation[0]}`)
  });
  const control = buffers.find(buffer => buffer.id === 'reverse-requested-generation')!;
  const writeSpy = vi.spyOn(control, 'write');
  graph.add(
    createGPUCopyCommandNode<{frame: 'A' | 'B'}>({
      id: 'select-geometry',
      resources: [
        {buffer: sourceA.view, usage: 'copy-source'},
        {buffer: sourceB.view, usage: 'copy-source'},
        {buffer: source.view, usage: 'copy-destination'}
      ],
      compile: () => ({
        encode: ({commandEncoder, parameters}) =>
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: parameters.frame === 'A' ? sourceA.buffer : sourceB.buffer,
            destinationBuffer: source.buffer,
            size: 8
          })
      })
    })
  );
  for (const node of snapshot.nodes) graph.add(node);
  graph.add(snapshot.commit(snapshot.nodes));
  graph.add(
    createGPUCopyCommandNode<{frame: 'A' | 'B'}>({
      id: 'capture-geometry',
      resources: [
        {buffer: snapshot.views.positions, usage: 'copy-source'},
        {buffer: outputA.view, usage: 'copy-destination'},
        {buffer: outputB.view, usage: 'copy-destination'}
      ],
      compile: () => ({
        encode: ({commandEncoder, getBuffer, parameters}) =>
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: getBuffer(snapshot.views.positions),
            destinationBuffer: parameters.frame === 'A' ? outputA.buffer : outputB.buffer,
            size: 8
          })
      })
    })
  );
  const compiled = graph.compile();
  const encode = (frame: 'A' | 'B') => {
    events.push(`encode:${frame}`);
    const encoder = device.createCommandEncoder();
    try {
      compiled.encode(encoder, {parameters: {frame}});
      return encoder.finish();
    } finally {
      encoder.destroy();
    }
  };
  const first = encode('A');
  generation[0] = 2;
  const second = encode('B');
  try {
    events.push('submit:B');
    device.submit(second);
    events.push('submit:A');
    device.submit(first);
    expect(events).toEqual([
      'encode:A',
      'request:1',
      'encode:B',
      'request:2',
      'submit:B',
      'submit:A'
    ]);
    expect(await readFloat32(outputB.buffer, 2)).toEqual([30, 40]);
    // The immutable byte table records generation-specific offsets in each command stream.
    expect(writeSpy).not.toHaveBeenCalled();
    expect(await readFloat32(outputA.buffer, 2)).toEqual([10, 20]);
  } finally {
    writeSpy.mockRestore();
    first.destroy();
    second.destroy();
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
