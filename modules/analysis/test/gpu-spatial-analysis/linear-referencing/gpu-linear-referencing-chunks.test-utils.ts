// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

/** Small parity fixtures give every port a different component offset, all beyond 256 bytes. */
export function createPaddedQueryFixture(device: Device) {
  const graph = new GPUCommandGraph(device);
  const sentinel = 0xa5a5a5a5;
  const allocations: {buffer: Buffer; prefix: number; length: number; words: number}[] = [];
  const views = new Map<GraphDataView, (typeof allocations)[number]>();
  function allocate<Format extends GraphDataView['format']>(
    name: string,
    format: Format,
    length: number,
    data?: Float32Array | Uint32Array
  ) {
    const components = Number(format.split('x')[1] ?? 1);
    // Coprime steps give adjacent ports distinct prefixes even inside the aligned binding.
    const prefix = 67 + allocations.length * 19;
    const words = length * components;
    const values = new Uint32Array(prefix + words + 17).fill(sentinel);
    if (data) values.set(new Uint32Array(data.buffer, data.byteOffset, data.length), prefix);
    const buffer = createOutputBuffer(device, values.length);
    buffer.write(values);
    const backing = importGraphBuffer(graph, name, buffer, 'uint32', values.length);
    const view = graph.createDataView(backing.buffer, {format, length, byteOffset: prefix * 4});
    const allocation = {buffer, prefix, length: values.length, words};
    allocations.push(allocation);
    views.set(view, allocation);
    return view;
  }
  async function read(view: GraphDataView) {
    const {buffer, prefix, words} = views.get(view)!;
    const values = await (view.format === 'uint32' ? readUint32(buffer) : readFloat32(buffer));
    return values.slice(prefix, prefix + words);
  }
  async function readRows(chunks: readonly GraphDataView[]) {
    return (await Promise.all(chunks.map(read))).flat();
  }
  async function submitTwice<Result>(
    compiled: ReturnType<typeof graph.compile>,
    readResult: () => Promise<Result>
  ): Promise<Result> {
    let first: Result | undefined;
    for (let submission = 0; submission < 2; submission++) {
      submitGraph(device, compiled, undefined);
      const result = await readResult();
      for (const {buffer, prefix, words, length} of allocations) {
        const bits = await readUint32(buffer);
        expect(bits.slice(0, prefix)).toEqual(Array(prefix).fill(sentinel));
        expect(bits.slice(prefix + words)).toEqual(Array(length - prefix - words).fill(sentinel));
      }
      if (submission === 0) first = result;
      else expect(result).toEqual(first);
    }
    return first!;
  }
  return {
    graph,
    input: <Format extends GraphDataView['format']>(
      name: string,
      format: Format,
      data: Float32Array | Uint32Array
    ) => allocate(name, format, data.length / Number(format.split('x')[1] ?? 1), data),
    output: <Format extends GraphDataView['format']>(
      name: string,
      format: Format,
      length: number
    ) => allocate(name, format, length),
    read,
    readRows,
    submitTwice,
    destroy: () => {
      for (const {buffer} of allocations) buffer.destroy();
    }
  };
}

/** Checks generated numeric tokens, including values a syntax-only WGSL parser accepts. */
export function expectValidQueryShaderLiterals(source: string) {
  const code = source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '');
  expect(code).not.toMatch(/\b(?:NaN|Infinity|nan|inf)\b/);
  const literals = /\b(0[xX][0-9a-fA-F]+|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)([uifh]?)(?![\w.])/g;
  for (const match of code.matchAll(literals)) {
    const [, digits, suffix] = match;
    const floating = !digits.startsWith('0x') && /[.eE]/.test(digits);
    if (floating || suffix === 'f' || suffix === 'h') {
      expect(Number.isFinite(Math.fround(Number(digits))), match[0]).toBe(true);
      if (suffix === 'h') expect(Math.abs(Number(digits)), match[0]).toBeLessThanOrEqual(65504);
    } else {
      const negative = code.slice(0, match.index).trimEnd().endsWith('-');
      const maximum =
        suffix === 'u'
          ? 0xffffffffn
          : suffix === 'i'
            ? negative
              ? 0x80000000n
              : 0x7fffffffn
            : negative
              ? 0x8000000000000000n
              : 0x7fffffffffffffffn;
      expect(BigInt(digits) <= maximum, match[0]).toBe(true);
    }
  }
}
