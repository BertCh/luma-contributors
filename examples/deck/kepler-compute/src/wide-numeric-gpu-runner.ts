// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import {GPUCommandGraph, getBoundedDispatchLayout, getBoundedInvocationIndexSource} from '@luma.gl/graph/gpu-core';
import {WIDE_NUMERIC_WGSL_HELPERS} from './wide-numeric-gpu-wgsl';

export type WideNumericKernelProps = {
  device: Device;
  id: string;
  /** Resident 64-bit words (low, high) per row; every chunk is scanned, chunk boundaries are preserved. */
  words: GPUVector<'uint32x2'> | GPUVector<'uint32x4'>;
  /** Words per row in `words`: 2 for 64-bit values (default), 4 for Decimal128. */
  wordsPerRow?: number;
  /** WGSL boolean over `pair` giving `selected` (present and the expression); without it `selected` is true. */
  selectionExpression?: string;
  /** Extra module-scope WGSL (functions) appended after the shared 64-bit helpers. */
  helperSource?: string;
  /** Optional per-row validity (non-zero = valid) with the same chunking as `words`. */
  validity?: GPUVector<'uint32'>;
  resultWordCount: number;
  initialWords?: Uint32Array;
  /**
   * One WGSL statement block per phase. Each block runs once per row with `index` (row in chunk),
   * `pair` (vec2<u32>, zero for null rows), `present` (bool), `wordOffset` (u32 index of the row's
   * low word in `words`), `words` (read-only u32 array) and `result` (array<atomic<u32>>).
   * Phases run in order over every chunk, so a later phase observes all earlier atomic updates.
   */
  phaseBodies: readonly string[];
};

/** Runs the phases over all chunks of a resident Int64/Uint64/Float64 word vector and reads back only `resultWordCount` words. */
export async function runWideNumericKernel(props: WideNumericKernelProps): Promise<Uint32Array> {
  const {device} = props;
  const byteLength = Math.max(16, props.resultWordCount * 4);
  const output = device.createBuffer({
    id: `${props.id}-result`,
    data: props.initialWords
      ? new Uint32Array(byteLength / 4).map((_, index) => props.initialWords![index] ?? 0)
      : new Uint32Array(byteLength / 4),
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const graph = new GPUCommandGraph(device, {id: props.id});
  let compiled: ReturnType<GPUCommandGraph['compile']> | undefined;
  try {
    const result = graph.createDataView(
      graph.importBuffer({id: 'result', byteLength, usage: output.usage}, output),
      {format: 'uint32', length: byteLength / 4}
    );
    const words = graph.importGPUVector('words', props.words);
    const validity = props.validity ? graph.importGPUVector('validity', props.validity) : undefined;
    const wordsPerRow = props.wordsPerRow ?? 2;
    for (const [phaseIndex, phaseBody] of props.phaseBodies.entries()) {
      let globalRow = 0;
      for (const [chunkIndex, chunk] of words.data.entries()) {
        const chunkRowBase = globalRow;
        globalRow += chunk.length;
        if (!chunk.length) continue;
        const valid = validity?.data[chunkIndex];
        const dispatch = getBoundedDispatchLayout(
          'WideNumericScan',
          chunk.length,
          256,
          device.limits.maxComputeWorkgroupsPerDimension
        );
        graph.addComputePass({
          id: `phase-${phaseIndex}-chunk-${chunkIndex}`,
          resources: [
            {buffer: chunk, usage: 'storage-read'},
            ...(valid ? [{buffer: valid, usage: 'storage-read' as const}] : []),
            {buffer: result, usage: 'storage-read-write'}
          ],
          compile: ({device: compileDevice}) => {
            const kernel = new Kernel(compileDevice, {
              source: `
${WIDE_NUMERIC_WGSL_HELPERS}
${props.helperSource ?? ''}
@group(0) @binding(0) var<storage,read> words: array<u32>;
@group(0) @binding(1) var<storage,read_write> result: array<atomic<u32>>;
${valid ? '@group(0) @binding(2) var<storage,read> validity: array<u32>;' : ''}
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) localInvocationIndex: u32, @builtin(workgroup_id) workgroupId: vec3<u32>) {
  ${getBoundedInvocationIndexSource(dispatch, 256)}
  if (index >= ${chunk.length}u) {return;}
  let wordOffset = ${chunk.byteOffset / 4}u + index * ${wordsPerRow}u;
  let globalRow = ${chunkRowBase}u + index;
  let present = ${valid ? `validity[${valid.byteOffset / 4}u + index] != 0u` : 'true'};
  let pair = select(vec2<u32>(0u, 0u), vec2<u32>(words[wordOffset], words[wordOffset + 1u]), present);
  let selected = ${props.selectionExpression ? `present && (${props.selectionExpression})` : 'true'};
  ${phaseBody}
}`,
              shaderLayout: {
                bindings: [
                  {name: 'words', location: 0, group: 0, type: 'storage'},
                  {name: 'result', location: 1, group: 0, type: 'storage'},
                  ...(valid
                    ? [{name: 'validity', location: 2, group: 0, type: 'storage' as const}]
                    : [])
                ]
              }
            });
            return {
              encode: ({computePass, getBuffer}) =>
                kernel.dispatch(computePass, {
                  bindings: {
                    words: getBuffer(chunk.buffer),
                    result: getBuffer(result.buffer),
                    ...(valid ? {validity: getBuffer(valid.buffer)} : {})
                  },
                  ...dispatch
                }),
              destroy: () => kernel.destroy()
            };
          }
        });
      }
    }
    compiled = graph.compile();
    const encoder = device.createCommandEncoder();
    compiled.encode(encoder, {parameters: undefined});
    device.submit(encoder.finish());
    const bytes = await output.readAsync(0, byteLength);
    return new Uint32Array(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + props.resultWordCount * 4)
    );
  } finally {
    compiled?.destroy();
    output.destroy();
  }
}
