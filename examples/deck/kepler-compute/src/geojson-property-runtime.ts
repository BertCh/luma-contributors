// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';

export type GeoJSONPropertySource<Format extends 'float32' | 'unorm8x4'> = {
  values: GPUVector<Format>;
  /** Native presence predicate, computed once at ingestion (truthiness or finite offset presence). */
  validity: GPUVector<'uint32'>;
};

/** Resolve immutable source properties against mutable native defaults without source readback. */
export class GeoJSONPropertyRuntime<Format extends 'float32' | 'unorm8x4'> {
  readonly output: GPUVector<Format>;
  private readonly buffers: Buffer[] = [];
  private readonly controls: Buffer;
  private readonly compiled: CompiledGPUCommandGraph;
  private fallback = new Uint32Array(1);
  private revision = 0;
  private encodedRevision = -1;
  private destroyed = false;

  constructor(
    private readonly device: Device,
    readonly source: GeoJSONPropertySource<Format>,
    fallback: number | readonly number[]
  ) {
    const graph = new GPUCommandGraph(device, {id: 'geojson-property-fallback'});
    // Property uploads retain matching source batch boundaries and packed four-byte values.
    assert(source.values.data.length === source.validity.data.length);
    assert(source.values.length === source.validity.length);
    try {
      this.controls = this.allocate(4);
      const controls = graph.importBuffer(
        {id: 'fallback', byteLength: 4, usage: this.controls.usage},
        this.controls
      );
      this.output = new GPUVector({
        type: 'data',
        name: 'geojson-property-fallback',
        format: source.values.format,
        ownsData: false,
        data: source.values.data.map((chunk, index) => {
          const validity = source.validity.data[index];
          // Borrowed inputs must be packed, aligned, on this device, and batch aligned.
          assert(chunk.length === validity.length);
          assert(chunk.byteStride === 4 && validity.byteStride === 4);
          assert(chunk.byteOffset % 4 === 0 && validity.byteOffset % 4 === 0);
          assert(chunk.buffer.device === device && validity.buffer.device === device);
          const buffer = this.allocate(Math.max(4, chunk.length * 4));
          const output = new GPUData({
            format: source.values.format,
            buffer,
            length: chunk.length,
            ownsBuffer: false
          });
          if (!chunk.length) return output;
          const valuesBuffer = graph.importBuffer(
            {id: `values-${index}`, byteLength: chunk.buffer.byteLength, usage: chunk.buffer.usage},
            chunk.buffer
          );
          const validityBuffer = graph.importBuffer(
            {
              id: `validity-${index}`,
              byteLength: validity.buffer.byteLength,
              usage: validity.buffer.usage
            },
            validity.buffer
          );
          const outputBuffer = graph.importBuffer(
            {id: `output-${index}`, byteLength: buffer.byteLength, usage: buffer.usage},
            buffer
          );
          const workgroupCount = Math.ceil(chunk.length / 256);
          const maximum = device.limits.maxComputeWorkgroupsPerDimension;
          const workgroupsX = Math.min(workgroupCount, maximum);
          const workgroupsY = Math.ceil(workgroupCount / maximum);
          // Source batches must fit the device's two-dimensional dispatch capacity.
          assert(workgroupsY <= maximum);
          graph.addComputePass({
            id: `property-${index}`,
            resources: [
              {buffer: valuesBuffer, usage: 'storage-read'},
              {buffer: validityBuffer, usage: 'storage-read'},
              {buffer: controls, usage: 'storage-read'},
              {buffer: outputBuffer, usage: 'storage-write'}
            ],
            compile: () => {
              const kernel = new Kernel(device, {
                id: 'geojson-property-fallback',
                source: /* wgsl */ `
@group(0) @binding(0) var<storage, read> values: array<u32>;
@group(0) @binding(1) var<storage, read> validity: array<u32>;
@group(0) @binding(2) var<storage, read> fallback: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) local: u32) {
  let index = (group.y * ${workgroupsX}u + group.x) * 256u + local;
  if (index >= ${chunk.length}u) { return; }
  output[index] = select(fallback[0], values[index + ${chunk.byteOffset / 4}u], validity[index + ${validity.byteOffset / 4}u] != 0u);
}`,
                shaderLayout: {
                  bindings: ['values', 'validity', 'fallback', 'output'].map((name, location) => ({
                    name,
                    group: 0,
                    location,
                    type: location === 3 ? 'storage' : 'read-only-storage'
                  }))
                }
              });
              return {
                encode: ({computePass, getBuffer}) =>
                  kernel.dispatch(computePass, {
                    bindings: {
                      values: getBuffer(valuesBuffer),
                      validity: getBuffer(validityBuffer),
                      fallback: getBuffer(controls),
                      output: getBuffer(outputBuffer)
                    },
                    x: workgroupsX,
                    y: workgroupsY
                  }),
                destroy: () => kernel.destroy()
              };
            }
          });
          return output;
        })
      });
      this.setFallback(fallback);
      this.compiled = graph.compile();
    } catch (error) {
      for (const buffer of this.buffers) buffer.destroy();
      throw error;
    }
  }

  get colors(): GPUVector<'unorm8x4'> | undefined {
    return this.output.format === 'unorm8x4' ? (this.output as GPUVector<'unorm8x4'>) : undefined;
  }
  get values(): GPUVector<'float32'> | undefined {
    return this.output.format === 'float32' ? (this.output as GPUVector<'float32'>) : undefined;
  }
  setFallback(value: number | readonly number[]): boolean {
    const format = this.source.values.format;
    assert(format === 'float32' || format === 'unorm8x4');
    const fallback = getGeoJSONPropertyFallbackWord(format, value);
    if (this.fallback[0] !== fallback) {
      this.fallback = new Uint32Array([fallback]);
      this.revision++;
      return true;
    }
    return false;
  }
  encode(encoder: CommandEncoder): void {
    assert(!this.destroyed && encoder.device === this.device);
    if (this.encodedRevision === this.revision) return;
    this.device.writeBufferViaCommandEncoder(encoder, this.controls, this.fallback);
    this.compiled.encode(encoder, {parameters: undefined});
    this.encodedRevision = this.revision;
  }
  discardEncoding(): void {
    this.encodedRevision = -1;
  }
  get accountedResidentByteLength(): number {
    return this.destroyed
      ? 0
      : this.buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0) +
          this.compiled.stats.physicalTransientResourceBytes;
  }
  /** Caller fences dependent submissions first. The source vectors remain borrowed. */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.compiled.destroy();
    for (const buffer of this.buffers) buffer.destroy();
  }
  private allocate(byteLength: number): Buffer {
    const buffer = this.device.createBuffer({
      byteLength,
      usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    this.buffers.push(buffer);
    return buffer;
  }
}

/** Match native packed byte colors; scalar controls use the renderer's Float32 precision. */
export function getGeoJSONPropertyFallbackWord(
  format: 'float32' | 'unorm8x4',
  value: number | readonly number[]
): number {
  if (format === 'float32') {
    // Native scalar defaults must be finite and representable by the render attribute.
    assert(typeof value === 'number' && Number.isFinite(Math.fround(value)));
    return new Uint32Array(new Float32Array([value]).buffer)[0];
  }
  assert(Array.isArray(value));
  const color = new Uint8Array([value[0] ?? 0, value[1] ?? 0, value[2] ?? 0, value[3] ?? 255]);
  return new Uint32Array(color.buffer)[0];
}
