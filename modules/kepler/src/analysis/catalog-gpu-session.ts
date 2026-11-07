// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {CompiledGPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {readBufferFloat32, readBufferUint32} from '../runtime/gpu-readback';
import {getGPUVectorFormatInfo, type GPUVectorFormat} from '@luma.gl/graph/gpu-data';

/** Thrown by GPU runners for inputs the contributor cannot take; the catalog falls back to the CPU engine. */
export class CatalogGpuUnsupportedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'CatalogGpuUnsupportedError';
  }
}

export type CatalogGpuOutput<Format extends GPUVectorFormat> = {
  buffer: Buffer;
  view: GraphDataView<Format>;
  length: number;
  components: number;
};

/**
 * Owns the graph, the caller-owned storage buffers and the compiled graphs of one GPU run.
 * Contributors never submit; `run` compiles, encodes and submits once. `destroy` releases everything.
 */
export class CatalogGpuSession {
  readonly device: Device;
  readonly graph: GPUCommandGraph;
  private readonly buffers: Buffer[] = [];
  private readonly compiledGraphs: CompiledGPUCommandGraph<undefined>[] = [];
  private readonly sessionId: string;
  private resourceCount = 0;

  constructor(device: Device, id: string) {
    this.device = device;
    this.sessionId = id;
    this.graph = new GPUCommandGraph(device, {id});
  }

  /** Uploads values as a storage input. An empty array is padded to one word. */
  input<Format extends GPUVectorFormat>(
    name: string,
    values: Float32Array | Uint32Array | Int32Array,
    format: Format
  ): GraphDataView<Format> {
    const components = getGPUVectorFormatInfo(format).byteLength / 4;
    const length = Math.floor(values.length / components);
    const data = values.length ? values : new Uint32Array(components);
    const buffer = this.device.createBuffer({
      id: `${this.sessionId}-${name}`,
      data,
      usage: Buffer.STORAGE | Buffer.COPY_DST
    });
    this.buffers.push(buffer);
    return this.graph.createDataView(
      this.graph.importBuffer(
        {id: `${name}-${this.resourceCount++}`, byteLength: buffer.byteLength, usage: buffer.usage},
        buffer
      ),
      {format, length}
    );
  }

  /** Creates a zero-filled storage output with `length` rows. */
  output<Format extends GPUVectorFormat>(
    name: string,
    format: Format,
    length: number
  ): CatalogGpuOutput<Format> {
    const components = getGPUVectorFormatInfo(format).byteLength / 4;
    const buffer = this.device.createBuffer({
      id: `${this.sessionId}-${name}`,
      byteLength: Math.max(length * components, 1) * Uint32Array.BYTES_PER_ELEMENT,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    this.buffers.push(buffer);
    const view = this.graph.createDataView(
      this.graph.importBuffer(
        {id: `${name}-${this.resourceCount++}`, byteLength: buffer.byteLength, usage: buffer.usage},
        buffer
      ),
      {format, length}
    );
    return {buffer, view, length, components};
  }

  /** Compiles, encodes and submits the graph once. */
  run(): void {
    const compiled = this.graph.compile() as CompiledGPUCommandGraph<undefined>;
    this.compiledGraphs.push(compiled);
    const commandEncoder = this.device.createCommandEncoder({id: `${this.sessionId}-encoder`});
    compiled.encode(commandEncoder, {parameters: undefined});
    this.device.submit(commandEncoder.finish());
  }

  async readFloat32(
    output: CatalogGpuOutput<GPUVectorFormat>,
    rows?: number
  ): Promise<Float32Array> {
    return readBufferFloat32(output.buffer, (rows ?? output.length) * output.components);
  }

  async readUint32(output: CatalogGpuOutput<GPUVectorFormat>, rows?: number): Promise<Uint32Array> {
    return readBufferUint32(output.buffer, (rows ?? output.length) * output.components);
  }

  /** Reads one uint32 scalar. */
  async readScalar(output: CatalogGpuOutput<'uint32'>): Promise<number> {
    return (await this.readUint32(output, 1))[0];
  }

  /** Hands an output buffer to the caller: `destroy` no longer releases it. */
  detach(output: CatalogGpuOutput<GPUVectorFormat>): Buffer {
    const index = this.buffers.indexOf(output.buffer);
    if (index >= 0) this.buffers.splice(index, 1);
    return output.buffer;
  }

  destroy(): void {
    for (const compiled of this.compiledGraphs) compiled.destroy();
    for (const buffer of this.buffers) buffer.destroy();
    this.compiledGraphs.length = 0;
    this.buffers.length = 0;
  }
}

/** Runs `action` with a fresh session and always releases its resources. */
export async function withCatalogGpuSession<Result>(
  device: Device,
  id: string,
  action: (session: CatalogGpuSession) => Promise<Result>
): Promise<Result> {
  const session = new CatalogGpuSession(device, id);
  try {
    return await action(session);
  } finally {
    session.destroy();
  }
}
