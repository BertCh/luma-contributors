// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView, GraphVectorView} from './gpu-command-graph';
import type {GPUCommandNode} from './gpu-command-node';
import {getViewElementOffset, validatePackedView} from './graph-data-view-utils';
import {alignGraphVectorViews, getGraphVectorData} from './graph-vector-view-utils';
import {createChunkNode, getGraphDataRange, validateChunkViews} from './gpu-chunk-utils';
import {getBoundedDispatchLayout, getBoundedInvocationIndexSource} from './gpu-dispatch-utils';

export type GPUJacobiPreconditionerProps = {
  id?: string;
  rowOffsets: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  columnIndices: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  values: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  /** Reciprocal of the sum of each row's diagonal entries; zero when the diagonal is missing. */
  inverseDiagonal: GraphDataView<'float32'> | GraphVectorView<'float32'>;
};

/** Builds a Jacobi reciprocal diagonal from independently chunked CSR storage. */
export class GPUJacobiPreconditioner {
  readonly id: string;
  constructor(readonly props: GPUJacobiPreconditionerProps) {
    this.id = props.id ?? 'gpu-jacobi-preconditioner';
    for (const [name, view] of Object.entries({
      rowOffsets: props.rowOffsets,
      columnIndices: props.columnIndices,
      values: props.values,
      inverseDiagonal: props.inverseDiagonal
    })) {
      if (typeof view === 'string') continue;
      for (const chunk of getGraphVectorData(view))
        validatePackedView(
          chunk,
          [name === 'rowOffsets' || name === 'columnIndices' ? 'uint32' : 'float32'],
          this.id
        );
    }
    if (
      props.rowOffsets.length !== props.inverseDiagonal.length + 1 ||
      props.columnIndices.length !== props.values.length
    )
      throw new Error('Jacobi CSR dimensions must match the destination');
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {rowOffsets, columnIndices, values, inverseDiagonal} = this.props;
    validateChunkViews(graph, [rowOffsets, columnIndices, values], [inverseDiagonal]);
    const rows = alignGraphVectorViews(graph, [
      getGraphDataRange(graph, rowOffsets, 0, inverseDiagonal.length),
      getGraphDataRange(graph, rowOffsets, 1, inverseDiagonal.length),
      inverseDiagonal
    ]);
    const entries = alignGraphVectorViews(graph, [columnIndices, values]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    let rowStart = 0;
    for (const [rowIndex, [begins, ends, output]] of rows.entries()) {
      let entryStart = 0;
      for (let entryIndex = 0; entryIndex < Math.max(entries.length, 1); entryIndex++) {
        const pair = entries[entryIndex];
        const final = entryIndex === Math.max(entries.length, 1) - 1;
        // Keep the original scalar kernel for small independently stored entry spans.
        const cooperative = Boolean(pair && pair[0].length >= 4096);
        const workgroupSize = cooperative ? 64 : 256;
        const dispatch = getBoundedDispatchLayout(
          this.id,
          output.length,
          workgroupSize,
          graph.device.limits.maxComputeWorkgroupsPerDimension
        );
        const source =
          cooperative && pair
            ? makeCooperativeJacobiShader({
                begins,
                ends,
                output,
                columns: pair[0],
                values: pair[1],
                rowStart,
                entryStart,
                initialize: entryIndex === 0,
                final,
                dispatch
              })
            : `
${
  pair
    ? `@group(0) @binding(0) var<storage, read> rowBegins: array<u32>;
@group(0) @binding(1) var<storage, read> rowEnds: array<u32>;
@group(0) @binding(2) var<storage, read> columns: array<u32>;
@group(0) @binding(3) var<storage, read> values: array<f32>;`
    : ''
}
@group(0) @binding(${pair ? 4 : 0}) var<storage, read_write> diagonalValues: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${getBoundedInvocationIndexSource(dispatch, 256)}
  if (index >= ${output.length}u) { return; }
  let destination = ${getViewElementOffset(output)}u + index;
  var diagonal = ${entryIndex === 0 ? '0.0' : 'diagonalValues[destination]'};
  ${
    pair
      ? `let begin = max(rowBegins[${getViewElementOffset(begins)}u + index], ${entryStart}u);
  let end = min(rowEnds[${getViewElementOffset(ends)}u + index], ${entryStart + pair[0].length}u);
  for (var entry = begin; entry < end; entry++) {
    let localEntry = entry - ${entryStart}u;
    if (columns[${getViewElementOffset(pair[0])}u + localEntry] == ${rowStart}u + index) {
      diagonal += values[${getViewElementOffset(pair[1])}u + localEntry];
    }
  }`
      : ''
  }
  ${final ? `if (diagonal != 0.0) { diagonal = 1.0 / diagonal; }` : ''}
  diagonalValues[destination] = diagonal;
}`;
        nodes.push(
          createChunkNode(graph, {
            id: `${this.id}-rows-${rowIndex}-entries-${entryIndex}${cooperative ? '-cooperative' : ''}`,
            source,
            // Empty entry domains only write zero; unused row bindings are absent from the shader interface.
            inputs: pair
              ? {rowBegins: begins, rowEnds: ends, columns: pair[0], values: pair[1]}
              : {},
            outputs: {diagonalValues: output},
            dispatch,
            workgroupSize
          })
        );
        entryStart += pair?.[0].length ?? 0;
      }
      rowStart += output.length;
    }
    return nodes;
  }
}

/** Short rows stay scalar; the workgroup cooperates on its locally queued long rows. */
function makeCooperativeJacobiShader(props: {
  begins: GraphDataView;
  ends: GraphDataView;
  output: GraphDataView;
  columns: GraphDataView;
  values: GraphDataView;
  rowStart: number;
  entryStart: number;
  initialize: boolean;
  final: boolean;
  dispatch: ReturnType<typeof getBoundedDispatchLayout>;
}): string {
  const {begins, ends, output, columns, values, rowStart, entryStart, initialize, final, dispatch} =
    props;
  return `
@group(0) @binding(0) var<storage, read> rowBegins: array<u32>;
@group(0) @binding(1) var<storage, read> rowEnds: array<u32>;
@group(0) @binding(2) var<storage, read> columns: array<u32>;
@group(0) @binding(3) var<storage, read> values: array<f32>;
@group(0) @binding(4) var<storage, read_write> diagonalValues: array<f32>;
var<workgroup> longRowBits: array<atomic<u32>, 2>;
var<workgroup> pendingRowBits: array<u32, 2>;
var<workgroup> partialDiagonal: array<vec2f, 64>;

fn sumDiagonal(row: u32, firstEntry: u32, stride: u32, initial: f32, entryLimit: u32) -> vec2f {
  let begin = max(rowBegins[${getViewElementOffset(begins)}u + row], ${entryStart}u);
  let end = min(rowEnds[${getViewElementOffset(ends)}u + row], ${entryStart + columns.length}u);
  var diagonal = initial;
  var foundNegative = initial < 0.0;
  var localEntry = firstEntry;
  loop {
    if (begin >= end || localEntry >= min(end - begin, entryLimit)) { break; }
    let entry = begin - ${entryStart}u + localEntry;
    if (columns[${getViewElementOffset(columns)}u + entry] == ${rowStart}u + row) {
      let value = values[${getViewElementOffset(values)}u + entry];
      foundNegative = foundNegative || value < 0.0;
      diagonal += value;
    }
    // Avoid wrapping the index at the final uint32 range.
    if (end - begin - localEntry <= stride) { break; }
    localEntry += stride;
  }
  return vec2f(diagonal, f32(foundNegative));
}

fn writeDiagonal(row: u32, value: f32) {
  var diagonal = value;
  ${final ? 'if (diagonal != 0.0) { diagonal = 1.0 / diagonal; }' : ''}
  diagonalValues[${getViewElementOffset(output)}u + row] = diagonal;
}

fn sumScalarDiagonal(row: u32, firstEntry: u32, initial: f32) -> f32 {
  let begin = max(rowBegins[${getViewElementOffset(begins)}u + row], ${entryStart}u);
  let end = min(rowEnds[${getViewElementOffset(ends)}u + row], ${entryStart + columns.length}u);
  var diagonal = initial;
  if (begin < end) {
    for (var entry = begin + min(firstEntry, end - begin); entry < end; entry++) {
      let localEntry = entry - ${entryStart}u;
      if (columns[${getViewElementOffset(columns)}u + localEntry] == ${rowStart}u + row) {
        diagonal += values[${getViewElementOffset(values)}u + localEntry];
      }
    }
  }
  return diagonal;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${getBoundedInvocationIndexSource(dispatch, 64)}
  if (localInvocationIndex < 2u) { atomicStore(&longRowBits[localInvocationIndex], 0u); }
  workgroupBarrier();
  if (index < ${output.length}u) {
    let begin = max(rowBegins[${getViewElementOffset(begins)}u + index], ${entryStart}u);
    let end = min(rowEnds[${getViewElementOffset(ends)}u + index], ${entryStart + columns.length}u);
    let initial = ${initialize ? '0.0' : `diagonalValues[${getViewElementOffset(output)}u + index]`};
    if (end - min(begin, end) <= 256u) {
      writeDiagonal(index, sumScalarDiagonal(index, 0u, initial));
    } else {
      atomicOr(&longRowBits[localInvocationIndex / 32u], 1u << (localInvocationIndex % 32u));
    }
  }
  workgroupBarrier();
  if (localInvocationIndex < 2u) {
    pendingRowBits[localInvocationIndex] = atomicLoad(&longRowBits[localInvocationIndex]);
  }
  let firstWord = workgroupUniformLoad(&pendingRowBits[0]);
  let secondWord = workgroupUniformLoad(&pendingRowBits[1]);
  // Many long rows already expose row parallelism; do not serialize them through one local queue.
  if (countOneBits(firstWord) + countOneBits(secondWord) > 4u) {
    let word = select(firstWord, secondWord, localInvocationIndex >= 32u);
    if ((word & (1u << (localInvocationIndex % 32u))) != 0u) {
      let initial = ${initialize ? '0.0' : `diagonalValues[${getViewElementOffset(output)}u + index]`};
      writeDiagonal(index, sumScalarDiagonal(index, 0u, initial));
    }
    return;
  }
  // Uniform loads keep every reduction barrier in workgroup-uniform control flow.
  for (var word = 0u; word < 2u; word++) {
    var pending = workgroupUniformLoad(&pendingRowBits[word]);
    while (pending != 0u) {
      let row = workgroupIndex * 64u + word * 32u + firstTrailingBit(pending);
      pending &= pending - 1u;
      if (localInvocationIndex == 0u) {
        let original = ${initialize ? '0.0' : `diagonalValues[${getViewElementOffset(output)}u + row]`};
        partialDiagonal[0] = sumDiagonal(row, 0u, 1u, original, 256u);
      }
      let prefix = workgroupUniformLoad(&partialDiagonal[0]);
      if (prefix.y != 0.0) {
        // A signed prefix already requires sequential rounding; avoid a speculative full scan.
        if (localInvocationIndex == 0u) {
          writeDiagonal(row, sumScalarDiagonal(row, 256u, prefix.x));
        }
        workgroupBarrier();
        continue;
      }
      var initial = 0.0;
      if (localInvocationIndex == 0u) { initial = prefix.x; }
      // Every invocation must finish reading the prefix before lane zero reuses its storage.
      workgroupBarrier();
      partialDiagonal[localInvocationIndex] = sumDiagonal(row, 256u + localInvocationIndex, 64u, initial, 0xffffffffu);
      workgroupBarrier();
      for (var stride = 32u; stride > 0u; stride /= 2u) {
        if (localInvocationIndex < stride) {
          let left = partialDiagonal[localInvocationIndex];
          let right = partialDiagonal[localInvocationIndex + stride];
          partialDiagonal[localInvocationIndex] = vec2f(left.x + right.x, max(left.y, right.y));
        }
        workgroupBarrier();
      }
      if (localInvocationIndex == 0u) {
        var diagonal = partialDiagonal[0].x;
        // Signed duplicate terms retain sequential rounding and zero/breakdown behavior.
        if (partialDiagonal[0].y != 0.0) {
          let original = ${initialize ? '0.0' : `diagonalValues[${getViewElementOffset(output)}u + row]`};
          diagonal = sumScalarDiagonal(row, 0u, original);
        }
        writeDiagonal(row, diagonal);
      }
      // Finish lane zero's read before the next queued row can overwrite shared partials.
      workgroupBarrier();
    }
  }
}`;
}
