// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  getWGSLKernelBinding
} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from './gpu-contributor-test-utils';

/** One linear test kernel over shared WGSL modules: read-only inputs and one `u32` output. */
export type WGSLModuleKernel = {
  /** Kernel ID. */
  id: string;
  /** Read-only inputs by WGSL binding name; the element type follows the array type. */
  inputs: Record<string, Float32Array | Uint32Array>;
  /** Number of `u32` words in the `output` binding. */
  outputLength: number;
  /** Module-scope declarations (the modules under test and helpers). */
  declarations: string;
  /** Statements run once per `index` in `[0, invocationCount)`. */
  body: string;
  /** Number of invocations. */
  invocationCount: number;
};

/**
 * Runs one kernel built with `createWGSLKernelNode` and returns the `output` words. Float results
 * are written with `bitcast<u32>` and read back through {@link getFloatWords}.
 */
export async function runWGSLModuleKernel(
  device: Device,
  kernel: WGSLModuleKernel
): Promise<Uint32Array> {
  const graph = new GPUCommandGraph(device, {id: kernel.id});
  const buffers = Object.entries(kernel.inputs).map(([name, values]) => ({
    name,
    values,
    buffer: createInputBuffer(device, values.length > 0 ? values : new Uint32Array(1))
  }));
  const outputBuffer = createOutputBuffer(device, kernel.outputLength);
  const bindings = buffers.map(({name, values, buffer}) =>
    getWGSLKernelBinding(
      name,
      importGraphBuffer(
        graph,
        name,
        buffer,
        values instanceof Float32Array ? 'float32' : 'uint32',
        Math.max(values.length, 1)
      )
    )
  );
  bindings.push(
    getWGSLKernelBinding(
      'output',
      importGraphBuffer(graph, 'output', outputBuffer, 'uint32', kernel.outputLength),
      'read_write'
    )
  );
  graph.add(
    createWGSLKernelNode(graph, {
      id: kernel.id,
      operation: 'WGSLModuleSpec',
      bindings,
      invocationCount: kernel.invocationCount,
      declarations: kernel.declarations,
      body: kernel.body
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const words = Uint32Array.from(await readUint32(outputBuffer, kernel.outputLength));
  compiled.destroy();
  for (const {buffer} of buffers) {
    buffer.destroy();
  }
  outputBuffer.destroy();
  return words;
}

/** Reinterprets output words as float32 values. */
export function getFloatWords(words: Uint32Array): Float32Array {
  return new Float32Array(words.buffer, words.byteOffset, words.length);
}

/** Deterministic uniform random numbers in `[0, 1)` (mulberry32). */
export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}
