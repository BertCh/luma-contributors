// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {CompiledGPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {readBufferFloat32, readBufferUint32} from '../runtime/gpu-readback';
import {getGPUVectorFormatInfo, type GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {validateCapacity, validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {KeplerAllocationError} from '../runtime/dataset-registry';

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
 * Owns caller storage and one public compiled graph, reused by subsequent serialized runs.
 * Contributors never submit. Disposal waits for compilation and submitted GPU completion.
 */
export class CatalogGpuSession {
  readonly device: Device;
  readonly graph: GPUCommandGraph;
  private readonly buffers: Buffer[] = [];
  private readonly inputBuffers = new WeakMap<object, Buffer>();
  private compiled?: CompiledGPUCommandGraph<undefined>;
  private compilation?: Promise<void>;
  private readonly completions = new Set<Promise<void>>();
  private readonly fences = new Set<ReturnType<Device['createFence']>>();
  private disposal?: Promise<void>;
  private destroyed = false;
  private allocatedByteLength = 0;
  private readonly sessionId: string;
  private resourceCount = 0;

  constructor(
    device: Device,
    id: string,
    private readonly maximumBufferBytes = Infinity
  ) {
    if (maximumBufferBytes !== Infinity)
      validateCapacity('CatalogGpuSession', 'maximumBufferBytes', maximumBufferBytes, 0);
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
    this.checkAllocation(data.byteLength);
    const buffer = this.device.createBuffer({
      id: `${this.sessionId}-${name}`,
      data,
      usage: Buffer.STORAGE | Buffer.COPY_DST
    });
    this.buffers.push(buffer);
    this.allocatedByteLength += buffer.byteLength;
    const view = this.graph.createDataView(
      this.graph.importBuffer(
        {id: `${name}-${this.resourceCount++}`, byteLength: buffer.byteLength, usage: buffer.usage},
        buffer
      ),
      {format, length}
    );
    this.inputBuffers.set(view, buffer);
    return view;
  }

  /** Creates a zero-filled storage output with `length` rows. */
  output<Format extends GPUVectorFormat>(
    name: string,
    format: Format,
    length: number
  ): CatalogGpuOutput<Format> {
    const components = getGPUVectorFormatInfo(format).byteLength / 4;
    const byteLength = Math.max(length * components, 1) * Uint32Array.BYTES_PER_ELEMENT;
    this.checkAllocation(byteLength);
    const buffer = this.device.createBuffer({
      id: `${this.sessionId}-${name}`,
      byteLength,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    this.buffers.push(buffer);
    this.allocatedByteLength += buffer.byteLength;
    const view = this.graph.createDataView(
      this.graph.importBuffer(
        {id: `${name}-${this.resourceCount++}`, byteLength: buffer.byteLength, usage: buffer.usage},
        buffer
      ),
      {format, length}
    );
    return {buffer, view, length, components};
  }

  /** Compiles once, retaining the public compiled graph for subsequent submissions. */
  async compile(signal?: AbortSignal): Promise<void> {
    this.checkActive(signal);
    if (this.compiled) return;
    this.compilation ??= this.graph.compileAsync().then(compiled => {
      this.compiled = compiled as CompiledGPUCommandGraph<undefined>;
    });
    await this.compilation;
    this.checkActive(signal);
  }

  /** Encodes and submits the retained graph; cancellation never destroys work in flight. */
  run(signal?: AbortSignal): void {
    this.checkActive(signal);
    validateRequirement(
      'CatalogGpuSession',
      !this.compilation || Boolean(this.compiled),
      'compile must finish before run'
    );
    this.compiled ??= this.graph.compile() as CompiledGPUCommandGraph<undefined>;
    const commandEncoder = this.device.createCommandEncoder({id: `${this.sessionId}-encoder`});
    this.compiled.encode(commandEncoder, {parameters: undefined});
    signal?.throwIfAborted();
    this.device.submit(commandEncoder.finish());
    // Fence failure cannot prove completion: retain resources until device loss in that case.
    let completion: Promise<void>;
    try {
      const fence = this.device.createFence();
      this.fences.add(fence);
      completion = Promise.race([fence.signaled.catch(() => this.device.lost), this.device.lost])
        .then(() => undefined)
        .finally(() => {
          fence.destroy();
          this.fences.delete(fence);
        });
    } catch {
      completion = this.device.lost.then(() => undefined);
    }
    this.completions.add(completion);
    void completion.then(
      () => this.completions.delete(completion),
      () => this.completions.delete(completion)
    );
  }

  async waitForCompletion(): Promise<void> {
    await Promise.all(this.completions);
  }

  /** Updates a caller-owned input between serialized executions without rebuilding the graph. */
  writeInput(view: GraphDataView<GPUVectorFormat>, values: Float32Array): void {
    this.checkActive();
    validateRequirement(
      'CatalogGpuSession',
      this.completions.size === 0,
      'input updates must await GPU completion'
    );
    const buffer = this.inputBuffers.get(view);
    validateRequirement('CatalogGpuSession', Boolean(buffer), 'input must be session owned');
    validateRequirement(
      'CatalogGpuSession',
      values.byteLength === view.length * getGPUVectorFormatInfo(view.format).byteLength,
      'input values must match view byteLength'
    );
    buffer!.write(values);
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
    if (index >= 0) {
      this.buffers.splice(index, 1);
      this.allocatedByteLength -= output.buffer.byteLength;
    }
    return output.buffer;
  }

  destroy(): Promise<void> {
    return this.dispose();
  }

  /** Waits for completion, releases independent resources, and retains failed releases for retry. */
  dispose(): Promise<void> {
    if (this.disposal) return this.disposal;
    this.destroyed = true;
    this.disposal = (async () => {
      await Promise.allSettled([this.compilation, ...this.completions]);
      const failures: unknown[] = [];
      try {
        this.compiled?.destroy();
        this.compiled = undefined;
      } catch (error) {
        failures.push(error);
      }
      for (const buffer of [...this.buffers]) {
        try {
          buffer.destroy();
          this.buffers.splice(this.buffers.indexOf(buffer), 1);
          this.allocatedByteLength -= buffer.byteLength;
        } catch (error) {
          failures.push(error);
        }
      }
      for (const fence of this.fences) {
        try {
          fence.destroy();
          this.fences.delete(fence);
        } catch (error) {
          failures.push(error);
        }
      }
      if (failures.length)
        throw new AggregateError(failures, `${this.sessionId}: CatalogGpuSession cleanup failed`);
    })().catch(error => {
      this.disposal = undefined;
      throw error;
    });
    return this.disposal;
  }

  private checkActive(signal?: AbortSignal): void {
    signal?.throwIfAborted();
    validateRequirement('CatalogGpuSession', !this.destroyed, 'session must be active');
  }

  private checkAllocation(byteLength: number): void {
    this.checkActive();
    if (this.allocatedByteLength + byteLength > this.maximumBufferBytes)
      throw new KeplerAllocationError('reservation');
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
    await session.dispose();
  }
}
