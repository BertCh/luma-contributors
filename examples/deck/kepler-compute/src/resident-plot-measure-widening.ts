// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {
  GPUCommandGraph,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  type CompiledGPUCommandGraph
} from '@luma.gl/graph/gpu-core';
import {assertExactFloat32IntegerRange} from './integer-style-range';

/**
 * Synchronously built Float32 view of an Int32 or Uint32 y-axis column. A compute pass per source
 * chunk converts into owned Float32 buffers on every `encode`, so plot measures can be constructed
 * without awaiting. Exactness (every value within +/-2^24) is certified by one bounded extent
 * readback in `assertExactRange`, which measures await before their first summary.
 */
export class ResidentPlotMeasureWidening {
  readonly vector: GPUVector<'float32'>;
  /** Owned Float32 bytes, for working-memory accounting. */
  readonly byteLength: number;
  private readonly buffers: Buffer[] = [];
  private readonly compiled: CompiledGPUCommandGraph;
  private rangeCheck?: Promise<void>;

  constructor(
    private readonly device: Device,
    private readonly source: GPUVector<'sint32'> | GPUVector<'uint32'>
  ) {
    const graph = new GPUCommandGraph(device, {id: 'kepler-plot-measure-widening'});
    const input = graph.importGPUVector('source', source);
    const chunks: GPUData<'float32'>[] = [];
    let byteLength = 0;
    try {
      for (const [chunkIndex, chunk] of input.data.entries()) {
        const buffer = device.createBuffer({
          id: 'plot-measure-widened',
          byteLength: Math.max(4, chunk.length * 4),
          usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
        });
        this.buffers.push(buffer);
        byteLength += buffer.byteLength;
        chunks.push(
          new GPUData({buffer, format: 'float32', length: chunk.length, ownsBuffer: false})
        );
        if (!chunk.length) continue;
        const output = graph.createDataView(
          graph.importBuffer(
            {id: `widened-${chunkIndex}`, byteLength: buffer.byteLength, usage: buffer.usage},
            buffer
          ),
          {format: 'float32', length: chunk.length}
        );
        const dispatch = getBoundedDispatchLayout(
          'PlotMeasureWidening',
          chunk.length,
          256,
          device.limits.maxComputeWorkgroupsPerDimension
        );
        const conversion =
          source.format === 'sint32'
            ? `f32(bitcast<i32>(input[${chunk.byteOffset / 4}u + index]))`
            : `f32(input[${chunk.byteOffset / 4}u + index])`;
        graph.addComputePass({
          id: `widen-${chunkIndex}`,
          resources: [
            {buffer: chunk, usage: 'storage-read'},
            {buffer: output, usage: 'storage-write'}
          ],
          compile: ({device: compileDevice}) => {
            const kernel = new Kernel(compileDevice, {
              source: `
@group(0) @binding(0) var<storage,read> input: array<u32>;
@group(0) @binding(1) var<storage,read_write> output: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) localInvocationIndex: u32, @builtin(workgroup_id) workgroupId: vec3<u32>) {
  ${getBoundedInvocationIndexSource(dispatch, 256)}
  if (index >= ${chunk.length}u) {return;}
  output[index] = ${conversion};
}`,
              shaderLayout: {
                bindings: ['input', 'output'].map((name, location) => ({
                  name,
                  location,
                  group: 0,
                  type: 'storage' as const
                }))
              }
            });
            return {
              encode: ({computePass, getBuffer}) =>
                kernel.dispatch(computePass, {
                  bindings: {
                    input: getBuffer(chunk.buffer),
                    output: getBuffer(output.buffer)
                  },
                  ...dispatch
                }),
              destroy: () => kernel.destroy()
            };
          }
        });
      }
      this.compiled = graph.compile();
    } catch (error) {
      this.buffers.forEach(buffer => buffer.destroy());
      throw error;
    }
    this.byteLength = byteLength;
    this.vector = new GPUVector<'float32'>({
      name: 'plot-measure-widened',
      type: 'data',
      format: 'float32',
      data: chunks,
      ownsData: false
    });
  }

  /** Encode before any pass that reads `vector`. */
  encode(encoder: CommandEncoder): void {
    this.compiled.encode(encoder, {parameters: undefined});
  }

  /** Rejects when a stored value lies outside +/-2^24, where Float32 conversion would round. */
  assertExactRange(): Promise<void> {
    this.rangeCheck ??= assertExactFloat32IntegerRange(this.device, this.source);
    return this.rangeCheck;
  }

  destroy(): void {
    this.compiled.destroy();
    this.buffers.forEach(buffer => buffer.destroy());
  }
}

/** Dictionary codes are labels, not measurements. */
export function isWidenableResidentPlotMeasure(
  vector: GPUVector | undefined,
  hasDictionary: boolean
): vector is GPUVector<'sint32'> | GPUVector<'uint32'> {
  return !hasDictionary && (vector?.format === 'sint32' || vector?.format === 'uint32');
}
