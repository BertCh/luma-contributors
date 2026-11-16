// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  createGPUCopyCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect} from 'vitest';
import {createVectorView, readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';

/** Distinct finite guards make skipped stores and prefix/tail overwrites observable. */
const FLOAT_SENTINEL = -987654.25;
const INTEGER_SENTINEL = 0xdeadbeef;
/** Extra backing words verify that writes preserve the storage surrounding each logical view. */
const GUARD_WORD_COUNT = 4;

/** Caller-owned guarded buffers, reusable across encodings and mutable between submissions. */
export function createEmbeddingScaleHarness(device: Device) {
  const graph = new GPUCommandGraph(device);
  const ownedBuffers: Buffer[] = [];
  function rows<Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    lengths: readonly number[],
    components = 1,
    offsets: readonly number[] = [4, 252, 256]
  ) {
    const sentinel = format === 'float32' ? FLOAT_SENTINEL : INTEGER_SENTINEL;
    const chunks = lengths.map((length, chunkIndex) => {
      const firstWord = offsets[chunkIndex % offsets.length] / 4;
      const count = length * components;
      const initial =
        format === 'float32'
          ? new Float32Array(firstWord + count + GUARD_WORD_COUNT).fill(sentinel)
          : new Uint32Array(firstWord + count + GUARD_WORD_COUNT).fill(sentinel);
      const buffer = device.createBuffer({
        data: initial,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      ownedBuffers.push(buffer);
      const handle = graph.importBuffer(
        {id: `${name}-${chunkIndex}`, byteLength: buffer.byteLength, usage: buffer.usage},
        buffer
      );
      const view = graph.createDataView(handle, {format, length: count, byteOffset: firstWord * 4});
      return {buffer, view, firstWord, count, initial};
    });
    return {
      view: createVectorView(
        name,
        format,
        chunks.map(chunk => chunk.view)
      ),
      reset() {
        for (const chunk of chunks) chunk.buffer.write(chunk.initial);
      },
      write(values: readonly number[]) {
        expect(values.length).toBe(chunks.reduce((total, chunk) => total + chunk.count, 0));
        let firstElement = 0;
        for (const chunk of chunks) {
          const data = values.slice(firstElement, firstElement + chunk.count);
          chunk.buffer.write(
            format === 'float32' ? Float32Array.from(data) : Uint32Array.from(data),
            chunk.firstWord * 4
          );
          firstElement += chunk.count;
        }
      },
      async read() {
        const result: number[] = [];
        for (const chunk of chunks) {
          const data = await (format === 'float32'
            ? readFloat32(chunk.buffer)
            : readUint32(chunk.buffer));
          expect(data.slice(0, chunk.firstWord)).toEqual(
            new Array<number>(chunk.firstWord).fill(sentinel)
          );
          expect(data.slice(chunk.firstWord + chunk.count)).toEqual(
            new Array<number>(GUARD_WORD_COUNT).fill(sentinel)
          );
          result.push(...data.slice(chunk.firstWord, chunk.firstWord + chunk.count));
        }
        return result;
      }
    };
  }
  return {
    graph,
    rows,
    packed<Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format,
      values: readonly number[]
    ) {
      const field = rows(name, format, [values.length]);
      field.write(values);
      return {...field, view: field.view.data[0]};
    },
    /**
     * Poison the dispatch metadata AFTER its consumers on every encoding. A second successful
     * encoding then proves the upload replays rather than relying on persistent transient data.
     */
    addWithMetadataPoison(contributor: GPUCommandNodeProducer & {readonly id: string}) {
      const nodes = contributor.getCommandNodes(graph);
      const upload = nodes.find(node => node.id.endsWith('-dispatch-upload'));
      const destination = upload?.resources?.find(
        resource => 'buffer' in resource && resource.usage === 'copy-destination'
      );
      if (!destination || !('buffer' in destination))
        throw new Error('embedding test needs the dispatch upload resource');
      const metadata = destination.buffer as GraphDataView<'uint32'>;
      const zeros = rows(`${contributor.id}-metadata-zeros`, 'uint32', [metadata.length], 1, [0]);
      zeros.write(new Array<number>(metadata.length).fill(0));
      const source = zeros.view.data[0];
      const poison = createGPUCopyCommandNode({
        id: `${contributor.id}-poison-metadata`,
        resources: [
          {buffer: source, usage: 'copy-source'},
          {buffer: metadata, usage: 'copy-destination'}
        ],
        compile: () => ({
          encode: ({commandEncoder, getBuffer}) =>
            commandEncoder.copyBufferToBuffer({
              sourceBuffer: getBuffer(source),
              sourceOffset: source.byteOffset,
              destinationBuffer: getBuffer(metadata),
              destinationOffset: metadata.byteOffset,
              size: metadata.length * 4
            })
        })
      });
      graph.add({getCommandNodes: () => [...nodes, poison]});
    },
    destroy() {
      for (const buffer of ownedBuffers) buffer.destroy();
    }
  };
}
