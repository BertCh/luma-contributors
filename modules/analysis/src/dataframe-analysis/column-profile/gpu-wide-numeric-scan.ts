// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Kernel} from '@luma.gl/graph';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import {
  createGPUComputeCommandNode,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource
} from '@luma.gl/graph/gpu-core';
import {WIDE_NUMERIC_WGSL_HELPERS} from './wide-numeric-gpu-wgsl';

import {
  validateCapacity,
  validateRequirement,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
export type GPUWideNumericScanProps = {
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
  output: GraphDataView<'uint32'>;
  /**
   * One WGSL statement block per phase. Each block runs once per row with `index` (row in chunk),
   * `pair` (vec2<u32>, zero for null rows), `present` (bool), `wordOffset` (u32 index of the row's
   * low word in `words`), `words` (read-only u32 array) and `result` (array<atomic<u32>>).
   * Phases run in order over every chunk, so a later phase observes all earlier atomic updates.
   */
  phaseBodies: readonly string[];
};

/** Counter/carry exactness requires fewer than 2^32 total rows, including nulls. */
export const WIDE_NUMERIC_MAXIMUM_ROW_COUNT = 0xffffffff;

/**
 * Parallel word phases for exact wide column aggregates and narrowing diagnostics. Chunked inputs
 * are supported with matching packed validity chunks. All props are compile-time; there are no
 * per-frame parameters. The caller initializes output words before execution. No readback occurs
 * in this contributor; the domain aggregate/diagnostic adapters perform bounded summary reads.
 */
export class GPUWideNumericScan implements GPUCommandNodeProducer {
  readonly id: string;
  readonly estimatedCommandNodeCount: number;
  constructor(readonly props: GPUWideNumericScanProps) {
    this.id = props.id;
    const id = this.id;
    const wordsPerRow = props.wordsPerRow ?? 2;
    validateRequirement(id, wordsPerRow === 2 || wordsPerRow === 4, 'wordsPerRow must be 2 or 4');
    validateRequirement(
      id,
      props.words.format === `uint32x${wordsPerRow}`,
      'words format must match wordsPerRow'
    );
    validateCapacity(id, 'resultWordCount', props.resultWordCount);
    validateRequirement(
      id,
      props.output.format === 'uint32' &&
        props.output.byteOffset === 0 &&
        props.output.length >= props.resultWordCount &&
        props.output.byteStride === 4,
      'output must be packed uint32 at byteOffset 0 with length at least resultWordCount'
    );
    validateCapacity(id, 'rowCount', props.words.length, 0, WIDE_NUMERIC_MAXIMUM_ROW_COUNT);
    validateRequirement(
      id,
      !props.validity ||
        (props.validity.format === 'uint32' &&
          props.validity.data.length === props.words.data.length),
      'validity must have the same chunks as words'
    );
    for (const [index, chunk] of props.words.data.entries()) {
      validateRequirement(
        id,
        chunk.byteStride === wordsPerRow * 4 && chunk.byteOffset % 4 === 0,
        `words chunk ${index} must be packed and word aligned`
      );
      const validity = props.validity?.data[index];
      validateRequirement(
        id,
        !validity ||
          (validity.length === chunk.length &&
            validity.byteStride === 4 &&
            validity.byteOffset % 4 === 0),
        `validity chunk ${index} must be packed, word aligned and match words length`
      );
    }
    this.estimatedCommandNodeCount =
      props.phaseBodies.length * props.words.data.filter(chunk => chunk.length > 0).length;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }

  getCommandNodes<Parameters>(graph: GPUCommandGraph<Parameters>): GPUCommandNode<Parameters>[] {
    const device = graph.device;
    // Binding-size checks belong here, before any nodes, because they require the device.
    for (const chunk of this.props.words.data)
      validateGPUStorageBindingByteLength(
        device,
        this.id,
        'words chunk',
        chunk.buffer.byteLength,
        'Split the column into smaller chunks'
      );
    for (const chunk of this.props.validity?.data ?? [])
      validateGPUStorageBindingByteLength(
        device,
        this.id,
        'validity chunk',
        chunk.buffer.byteLength,
        'Split the column into smaller chunks'
      );
    validateGPUStorageBindingByteLength(
      device,
      this.id,
      'output',
      this.props.output.buffer.byteLength,
      'Reduce resultWordCount'
    );
    const result = this.props.output;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const words = graph.importGPUVector(`${this.id}-words`, this.props.words);
    const validity = this.props.validity
      ? graph.importGPUVector(`${this.id}-validity`, this.props.validity)
      : undefined;
    const wordsPerRow = this.props.wordsPerRow ?? 2;
    for (const [phaseIndex, phaseBody] of this.props.phaseBodies.entries()) {
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
        nodes.push(
          createGPUComputeCommandNode<Parameters>({
            id: `${this.id}-phase-${phaseIndex}-chunk-${chunkIndex}`,
            resources: [
              {buffer: chunk, usage: 'storage-read'},
              ...(valid ? [{buffer: valid, usage: 'storage-read' as const}] : []),
              {buffer: result, usage: 'storage-read-write'}
            ],
            compile: ({device: compileDevice}) => {
              const kernel = new Kernel(compileDevice, {
                source: `
${WIDE_NUMERIC_WGSL_HELPERS}
${this.props.helperSource ?? ''}
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
  let selected = ${this.props.selectionExpression ? `present && (${this.props.selectionExpression})` : 'true'};
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
          })
        );
      }
    }
    return nodes;
  }
}
