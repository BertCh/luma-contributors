// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createOutputBuffer, submitGraph} from '../../utils/gpu-contributor-test-utils';

const WORDS_PER_ROW: Partial<Record<GPUVectorFormat, number>> = {
  uint32x2: 2,
  float32x2: 2,
  sint32: 1,
  uint32: 1,
  float32: 1
};

/**
 * Value-cube specs graph rig: imports typed inputs and outputs of any 32-bit format, runs the
 * graph once, and reads outputs back as raw words. Readbacks happen only after the whole graph has
 * been submitted, so a spec that uses it reads final outputs only.
 */
export class ValueCubeRig {
  readonly graph: GPUCommandGraph;
  private readonly buffers: Buffer[] = [];
  private readonly outputBuffers = new Map<GraphDataView, Buffer>();
  private serial = 0;
  private compiled?: {destroy(): void};
  private submitCount = 0;

  constructor(readonly device: Device) {
    this.graph = new GPUCommandGraph(device, {id: 'value-cube-spec'});
  }

  /** Imports an input array as a view of `format`. */
  input<Format extends GPUVectorFormat>(
    data: Float32Array | Uint32Array | Int32Array,
    format: Format
  ): GraphDataView<Format> {
    const buffer = this.device.createBuffer({
      data,
      usage: Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC
    });
    this.buffers.push(buffer);
    const words = WORDS_PER_ROW[format] ?? 1;
    return importGraphBuffer(
      this.graph,
      `input-${this.serial++}`,
      buffer,
      format,
      data.length / words
    );
  }

  /** Imports a zero-filled output of `length` rows. */
  output<Format extends GPUVectorFormat>(format: Format, length: number): GraphDataView<Format> {
    const words = WORDS_PER_ROW[format] ?? 1;
    const buffer = createOutputBuffer(this.device, length * words);
    // A recognisable fill, so a cell that a node forgot to write is noticed.
    buffer.write(new Uint32Array(buffer.byteLength / 4).fill(0x7f7f7f7f));
    this.buffers.push(buffer);
    const view = importGraphBuffer(this.graph, `output-${this.serial++}`, buffer, format, length);
    this.outputBuffers.set(view, buffer);
    return view;
  }

  /** Adds producers, compiles and submits once. */
  run(...producers: GPUCommandNodeProducer[]): void {
    if (this.submitCount > 0) {
      throw new Error(
        'ValueCubeRig runs one submission per rig: no CPU step may sit between stages'
      );
    }
    for (const producer of producers) {
      this.graph.add(producer);
    }
    const compiled = this.graph.compile();
    this.compiled = compiled;
    submitGraph(this.device, compiled, undefined);
    this.submitCount++;
  }

  /** Number of graph submissions so far. */
  get submissions(): number {
    return this.submitCount;
  }

  /** Reads an output view as raw words, after the one submission. */
  async readWords(view: GraphDataView): Promise<Uint32Array> {
    if (this.submitCount !== 1) {
      throw new Error('ValueCubeRig reads outputs only after the single graph submission');
    }
    const bytes = await this.outputBuffers.get(view)!.readAsync();
    const words = (WORDS_PER_ROW[view.format] ?? 1) * view.length;
    return new Uint32Array(bytes.buffer, bytes.byteOffset, words).slice();
  }

  async readUint(view: GraphDataView<'uint32'>): Promise<number[]> {
    return Array.from(await this.readWords(view));
  }

  async readFloat(view: GraphDataView<'float32'>): Promise<number[]> {
    const words = await this.readWords(view);
    return Array.from(new Float32Array(words.buffer, words.byteOffset, words.length));
  }

  /** Reads `uint32x2` rows as unsigned 64-bit keys. */
  async readKeys(view: GraphDataView<'uint32x2'>): Promise<bigint[]> {
    const words = await this.readWords(view);
    return Array.from(
      {length: view.length},
      (_, row) => (BigInt(words[2 * row + 1]) << 32n) | BigInt(words[2 * row])
    );
  }

  destroy(): void {
    this.compiled?.destroy();
    for (const buffer of this.buffers) {
      buffer.destroy();
    }
  }
}

/** mulberry32 generator with values in [0, 1). */
export function createValueCubeRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Splits unsigned 64-bit keys into little-endian `(low, high)` words. */
export function getKeyWords(keys: readonly bigint[]): Uint32Array {
  const words = new Uint32Array(keys.length * 2);
  keys.forEach((key, row) => {
    words[2 * row] = Number(key & 0xffffffffn);
    words[2 * row + 1] = Number(key >> 32n);
  });
  return words;
}
