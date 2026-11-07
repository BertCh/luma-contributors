// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import type {
  CompiledGPUCommandGraph,
  GPUCommandGraph,
  GraphBufferHandle,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {getGPUVectorFormatInfo, type GPUVectorFormat} from '@luma.gl/graph/gpu-data';

/**
 * Imports an application-owned buffer into a command graph and returns a typed view of it, using
 * only the public `GPUCommandGraph.importBuffer()` and `createDataView()` methods.
 *
 * @param graph Graph that receives the buffer.
 * @param id Debug id of the imported resource.
 * @param buffer Buffer owned by the application; the graph never destroys it.
 * @param format Fixed-width element format.
 * @param length Row count. Defaults to the number of whole rows the buffer can hold.
 */
export function importGraphBuffer<Format extends GPUVectorFormat, Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  buffer: Buffer,
  format: Format,
  length?: number
): GraphDataView<Format> {
  const rowByteLength = getGPUVectorFormatInfo(format).byteLength;
  const handle = graph.importBuffer(
    {id, byteLength: buffer.byteLength, usage: buffer.usage},
    buffer
  );
  return graph.createDataView(handle, {
    format,
    length: length ?? Math.floor(buffer.byteLength / rowByteLength)
  });
}

/**
 * Imports each application-owned buffer once into this graph. Every request gets a precisely
 * typed, non-owning view over the same handle; view creation never reimports the resource.
 * Keep this importer with the graph whose handles it owns, including during graph rebuilds.
 */
export function createGraphBufferImporter<Parameters>(graph: GPUCommandGraph<Parameters>) {
  const handles = new Map<Buffer, GraphBufferHandle>();
  return <Format extends GPUVectorFormat>(
    buffer: Buffer,
    format: Format,
    length?: number
  ): GraphDataView<Format> => {
    let handle = handles.get(buffer);
    if (!handle) {
      handle = graph.importBuffer(
        {id: buffer.id, byteLength: buffer.byteLength, usage: buffer.usage},
        buffer
      );
      handles.set(buffer, handle);
    }
    return graph.createDataView(handle, {
      format,
      length: length ?? Math.floor(buffer.byteLength / getGPUVectorFormatInfo(format).byteLength)
    });
  };
}

/** Encodes a compiled graph that takes no per-frame parameters and submits it to the device queue. */
export function submitGraph(device: Device, compiled: CompiledGPUCommandGraph<undefined>): void {
  const commandEncoder = device.createCommandEncoder({id: `${compiled.id}-encoder`});
  compiled.encode(commandEncoder, {parameters: undefined});
  device.submit(commandEncoder.finish());
}

/**
 * Publishes a view into an application-owned sink with an ordinary copy node. The graph orders the
 * copy after earlier writers through the declared `copy-source` use, so the observed result is
 * the value at this point of the authored sequence. Uses only the public `addCopyPass()`.
 *
 * @param graph Graph that receives the copy node.
 * @param source Packed view to observe; its backing buffer needs `Buffer.COPY_SRC`.
 * @param sink Packed destination view of the same format; its buffer needs `Buffer.COPY_DST`.
 * @param options Node id and the node the copy follows.
 */
export function addGraphCopy<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  source: GraphDataView,
  sink: GraphDataView,
  options: {id: string; afterNode?: string}
): void {
  graph.addCopyPass({
    id: options.id,
    ...(options.afterNode !== undefined && {dependsOn: [options.afterNode]}),
    resources: [
      {buffer: source, usage: 'copy-source'},
      {buffer: sink, usage: 'copy-destination'}
    ],
    compile: () => ({
      encode: ({commandEncoder, getBuffer}) => {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: getBuffer(source),
          sourceOffset: source.byteOffset,
          destinationBuffer: getBuffer(sink),
          destinationOffset: sink.byteOffset,
          size: Math.min(source.length * source.rowByteLength, sink.length * sink.rowByteLength)
        });
      }
    })
  });
}
