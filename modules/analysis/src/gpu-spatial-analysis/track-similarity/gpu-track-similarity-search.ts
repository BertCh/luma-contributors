// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {
  createTransientView,
  doGraphDataViewsOverlap,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPULineGeometryPort, GPUPackedGeometryPort} from '../../analysis-core/contracts/index';
import {GPUTrackSimilarity} from './gpu-track-similarity';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validatePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';

export type GPUTrackSimilaritySearchProps = {
  id?: string;
  linesA: GPUPackedGeometryPort<GPULineGeometryPort>;
  linesB?: GPUPackedGeometryPort<GPULineGeometryPort>;
  pairA: GraphDataView<'uint32'>;
  pairB: GraphDataView<'uint32'>;
  metric: 'dynamic-time-warping' | 'frechet' | 'longest-common-subsequence';
  /** Inclusive maximum distance, or minimum LCSS similarity in [0,1]. */
  threshold: GraphDataView<'float32'>;
  maximumFrechetVertexCount?: number;
  dynamicTimeWarpingBandRadius?: number;
  lcssTolerance?: number;
  lcssBandRadius?: number;
  /** Stable GPU compaction bounds dynamic-program scratch. Defaults to pair count. */
  survivorCapacity?: number;
  /** Exact top-k among the supplied threshold-matching pairs, grouped by pairA ID. */
  topK?: number;
  output: {
    distance?: GraphDataView<'float32'>;
    lowerBound?: GraphDataView<'float32'>;
    /** Required for LCSS, normalized matched length / shorter length. */
    similarity?: GraphDataView<'float32'>;
    /** Required for LCSS; possible-match cardinality upper bound. */
    upperBound?: GraphDataView<'float32'>;
    matched: GraphDataView<'uint32'>;
    /** 0 exact evaluated, 1 bound-pruned, 2 invalid pair/data/threshold, 4 dynamic-program capacity exceeded, 8 survivor capacity exceeded. */
    status: GraphDataView<'uint32'>;
    /** Optional [survivor count, evaluated capacity, overflow count]. */
    survivorSummary?: GraphDataView<'uint32'>;
    /** Original pair row indices, ordered by metric then original row; absent slots are 0xffffffff. */
    topKIndices?: GraphDataView<'uint32'>;
    /** Per A trajectory [written count, complete]; invalid or overflow pairs make complete zero. */
    topKSummary?: GraphDataView<'uint32x2'>;
  };
};

/**
 * Batched planar threshold search composing the existing tiled dynamic programs. Endpoint
 * distance is an admissible lower bound for both discrete Frechet and sum-of-distances DTW,
 * including any band. LCSS uses a possible-match cardinality upper bound; evaluating it is O(n*m). Pruned pairs skip dynamic programming and have NaN distance, never zero.
 * Input pair IDs are preserved; explicit query batches bound scratch. No approximate pruning.
 */
export class GPUTrackSimilaritySearch implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUTrackSimilaritySearchProps;
  readonly value: GraphDataView<'float32'>;
  readonly bound: GraphDataView<'float32'>;
  constructor(props: GPUTrackSimilaritySearchProps) {
    this.id = props.id ?? 'track-similarity-search';
    this.props = props;
    const value =
      props.metric === 'longest-common-subsequence'
        ? props.output.similarity
        : props.output.distance;
    const bound =
      props.metric === 'longest-common-subsequence'
        ? props.output.upperBound
        : props.output.lowerBound;
    assert(value && bound);
    this.value = value;
    this.bound = bound;
    assert(
      Number.isFinite(Math.fround(props.lcssTolerance ?? 1)) && (props.lcssTolerance ?? 1) >= 0
    );
    for (const lines of [props.linesA, props.linesB ?? props.linesA]) {
      validatePackedView(lines.positions, ['float32x2'], this.id);
      validatePackedView(lines.lineOffsets, ['uint32'], this.id);
      assert(lines.lineOffsets.length >= 1);
    }
    for (const view of [props.pairA, props.pairB, props.output.matched, props.output.status]) {
      validatePackedView(view, ['uint32'], this.id);
      assert(view.length === props.pairA.length);
    }
    for (const view of [this.value, this.bound]) {
      validatePackedView(view, ['float32'], this.id);
      assert(view.length === props.pairA.length);
    }
    const capacity = props.survivorCapacity ?? props.pairA.length;
    assert(Number.isInteger(capacity) && capacity >= 0 && capacity <= props.pairA.length);
    if (props.output.survivorSummary) {
      validatePackedView(props.output.survivorSummary, ['uint32'], this.id);
      assert(props.output.survivorSummary.length === 3);
    }
    if (props.topK !== undefined) {
      assert(Number.isInteger(props.topK) && props.topK > 0 && props.topK <= 1024);
      assert(props.output.topKIndices && props.output.topKSummary);
      validatePackedView(props.output.topKIndices, ['uint32'], this.id);
      validatePackedView(props.output.topKSummary, ['uint32x2'], this.id);
      assert(
        props.output.topKIndices.length === (props.linesA.lineOffsets.length - 1) * props.topK
      );
      assert(props.output.topKSummary.length === props.linesA.lineOffsets.length - 1);
    } else {
      assert(!props.output.topKIndices && !props.output.topKSummary);
    }
    validatePackedView(props.threshold, ['float32'], this.id);
    assert(props.threshold.length === 1);
    // Separate passes cannot preserve independently named outputs in overlapping storage.
    const outputs = Object.values(props.output).filter(
      (view): view is NonNullable<typeof view> => view !== undefined
    );
    for (const [index, output] of outputs.entries()) {
      assert(!outputs.slice(0, index).some(previous => doGraphDataViewsOverlap(previous, output)));
    }
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.linesA.positions,
      props.linesA.lineOffsets,
      props.linesB?.positions,
      props.linesB?.lineOffsets,
      props.pairA,
      props.pairB,
      props.threshold
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, value, bound} = this;
    const subsequence = props.metric === 'longest-common-subsequence';
    const linesB = props.linesB ?? props.linesA;
    validateGraphViewsBelongToGraph(id, graph, [
      props.linesA.positions,
      props.linesA.lineOffsets,
      linesB.positions,
      linesB.lineOffsets,
      props.pairA,
      props.pairB,
      props.threshold,
      ...Object.values(props.output)
    ]);
    const capacity = props.survivorCapacity ?? props.pairA.length;
    const filteredA = createTransientView(graph, `${id}-filtered-a`, 'uint32', capacity);
    const filteredB = createTransientView(graph, `${id}-filtered-b`, 'uint32', capacity);
    const flags = createTransientView(graph, `${id}-flags`, 'uint32', props.pairA.length);
    const ranks = createTransientView(graph, `${id}-ranks`, 'uint32', props.pairA.length);
    const compactValue = createTransientView(graph, `${id}-compact-value`, 'float32', capacity);
    const survivorSummary =
      props.output.survivorSummary ??
      createTransientView(graph, `${id}-survivor-summary`, 'uint32', 3);
    const programStatus = createTransientView(graph, `${id}-program-status`, 'uint32', capacity);
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-bound`,
        operation: 'GPUTrackSimilaritySearch',
        variant: subsequence ? 'possible-match-bound' : 'endpoint-bound',
        invocationCount: props.pairA.length,
        bindings: [
          {name: 'positionsA', view: props.linesA.positions, type: 'f32', access: 'read'},
          {name: 'offsetsA', view: props.linesA.lineOffsets, type: 'u32', access: 'read'},
          {name: 'positionsB', view: linesB.positions, type: 'f32', access: 'read'},
          {name: 'offsetsB', view: linesB.lineOffsets, type: 'u32', access: 'read'},
          {name: 'pairA', view: props.pairA, type: 'u32', access: 'read'},
          {name: 'pairB', view: props.pairB, type: 'u32', access: 'read'},
          {name: 'bounds', view: bound, type: 'f32', access: 'read_write'}
        ],
        declarations: `fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn loadA(row: u32) -> vec2<f32> { return vec2<f32>(positionsA[positionsAOffset + 2u * row], positionsA[positionsAOffset + 2u * row + 1u]); }
fn loadB(row: u32) -> vec2<f32> { return vec2<f32>(positionsB[positionsBOffset + 2u * row], positionsB[positionsBOffset + 2u * row + 1u]); }`,
        body: `var nanBits = 0x7fc00000u; bounds[boundsOffset + index] = bitcast<f32>(nanBits);
let first = pairA[pairAOffset + index]; let second = pairB[pairBOffset + index];
if (first >= ${props.linesA.lineOffsets.length - 1}u || second >= ${linesB.lineOffsets.length - 1}u) { return; }
let startA = offsetsA[offsetsAOffset + first]; let endA = offsetsA[offsetsAOffset + first + 1u]; let startB = offsetsB[offsetsBOffset + second]; let endB = offsetsB[offsetsBOffset + second + 1u];
if (startA >= endA || endA > ${props.linesA.positions.length}u || startB >= endB || endB > ${linesB.positions.length}u) { return; }
for (var row = startA; row < endA; row++) { let position = loadA(row); if (!finite(position.x) || !finite(position.y)) { return; } }
for (var row = startB; row < endB; row++) { let position = loadB(row); if (!finite(position.x) || !finite(position.y)) { return; } }
let startDifference = loadA(startA) - loadB(startB); let endDifference = loadA(endA - 1u) - loadB(endB - 1u);
${
  subsequence
    ? `var possibleA = 0u; var possibleB = 0u;
for (var rowA = startA; rowA < endA; rowA++) {
  var possible = false;
  for (var rowB = startB; rowB < endB; rowB++) { let difference = loadA(rowA) - loadB(rowB); if (sqrt(dot(difference, difference)) <= f32(${props.lcssTolerance ?? 1})) { possible = true; break; } }
  if (possible) { possibleA++; }
}
for (var rowB = startB; rowB < endB; rowB++) {
  var possible = false;
  for (var rowA = startA; rowA < endA; rowA++) { let difference = loadA(rowA) - loadB(rowB); if (sqrt(dot(difference, difference)) <= f32(${props.lcssTolerance ?? 1})) { possible = true; break; } }
  if (possible) { possibleB++; }
}
bounds[boundsOffset + index] = f32(min(possibleA, possibleB)) / f32(min(endA - startA, endB - startB));`
    : 'bounds[boundsOffset + index] = max(sqrt(dot(startDifference, startDifference)), sqrt(dot(endDifference, endDifference)));'
}`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-prune`,
        operation: 'GPUTrackSimilaritySearch',
        variant: 'prune-pairs',
        invocationCount: props.pairA.length,
        bindings: [
          {name: 'bounds', view: bound, type: 'f32', access: 'read'},
          {name: 'threshold', view: props.threshold, type: 'f32', access: 'read'},
          {name: 'pairA', view: props.pairA, type: 'u32', access: 'read'},
          {name: 'pairB', view: props.pairB, type: 'u32', access: 'read'},
          {name: 'flags', view: flags, type: 'u32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        body: `let bound = bounds[boundsOffset + index]; let limit = threshold[thresholdOffset];
let valid = (bitcast<u32>(bound) & 0x7f800000u) != 0x7f800000u && (bitcast<u32>(limit) & 0x7f800000u) != 0x7f800000u && limit >= 0.0${subsequence ? ' && limit <= 1.0' : ''};
// The small conservative margin retains threshold-neighbor pairs under f32 roundoff.
let pruned = valid && ${subsequence ? 'bound < limit - 0.000001' : 'bound > limit + max(bound, 1.0) * 0.000001'};
status[statusOffset + index] = select(select(0u, 1u, pruned), 2u, !valid);
flags[flagsOffset + index] = select(0u, 1u, valid && !pruned);`
      })
    ];
    nodes.push(
      ...new GPUScan({
        id: `${id}-survivor-scan`,
        input: flags,
        output: ranks,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-clear-survivors`,
        operation: 'GPUTrackSimilaritySearch',
        variant: 'clear-survivors',
        invocationCount: Math.max(capacity, 1),
        bindings: [
          {name: 'filteredA', view: filteredA, type: 'u32', access: 'read_write'},
          {name: 'filteredB', view: filteredB, type: 'u32', access: 'read_write'},
          {name: 'ranks', view: ranks, type: 'u32', access: 'read'},
          {name: 'flags', view: flags, type: 'u32', access: 'read'},
          {name: 'summary', view: survivorSummary, type: 'u32', access: 'read_write'}
        ],
        body: `if (index < ${capacity}u) { filteredA[filteredAOffset + index] = 0xffffffffu; filteredB[filteredBOffset + index] = 0xffffffffu; }
if (index == 0u) {
  let count = ${props.pairA.length ? `ranks[ranksOffset + ${props.pairA.length - 1}u] + flags[flagsOffset + ${props.pairA.length - 1}u]` : '0u'};
  summary[summaryOffset] = count; summary[summaryOffset + 1u] = min(count, ${capacity}u); summary[summaryOffset + 2u] = count - min(count, ${capacity}u);
}`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-compact`,
        operation: 'GPUTrackSimilaritySearch',
        variant: 'stable-survivors',
        invocationCount: props.pairA.length,
        bindings: [
          {name: 'ranks', view: ranks, type: 'u32', access: 'read'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'},
          {name: 'pairA', view: props.pairA, type: 'u32', access: 'read'},
          {name: 'pairB', view: props.pairB, type: 'u32', access: 'read'},
          {name: 'filteredA', view: filteredA, type: 'u32', access: 'read_write'},
          {name: 'filteredB', view: filteredB, type: 'u32', access: 'read_write'}
        ],
        body: `if (status[statusOffset + index] != 0u) { return; } let rank = ranks[ranksOffset + index];
if (rank >= ${capacity}u) { status[statusOffset + index] = 8u; return; }
filteredA[filteredAOffset + rank] = pairA[pairAOffset + index]; filteredB[filteredBOffset + rank] = pairB[pairBOffset + index];`
      })
    );
    if (capacity > 0)
      nodes.push(
        ...new GPUTrackSimilarity({
          id: `${id}-metric`,
          linesA: props.linesA,
          linesB,
          pairA: filteredA,
          pairB: filteredB,
          maximumFrechetVertexCount: props.maximumFrechetVertexCount,
          dynamicTimeWarpingBandRadius: props.dynamicTimeWarpingBandRadius,
          lcssTolerance: props.lcssTolerance,
          lcssBandRadius: props.lcssBandRadius,
          output: {
            frechet: props.metric === 'frechet' ? compactValue : undefined,
            dynamicTimeWarping: props.metric === 'dynamic-time-warping' ? compactValue : undefined,
            longestCommonSubsequence: subsequence ? compactValue : undefined,
            pairStatus: programStatus
          }
        }).getCommandNodes(graph)
      );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish`,
        operation: 'GPUTrackSimilaritySearch',
        variant: 'threshold-match',
        invocationCount: props.pairA.length,
        bindings: [
          {name: 'distance', view: value, type: 'f32', access: 'read_write'},
          {name: 'compactValue', view: compactValue, type: 'f32', access: 'read'},
          {name: 'ranks', view: ranks, type: 'u32', access: 'read'},
          {name: 'threshold', view: props.threshold, type: 'f32', access: 'read'},
          {name: 'programStatus', view: programStatus, type: 'u32', access: 'read'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'},
          {name: 'matched', view: props.output.matched, type: 'u32', access: 'read_write'}
        ],
        body: `var nanBits = 0x7fc00000u; distance[distanceOffset + index] = bitcast<f32>(nanBits);
if (status[statusOffset + index] == 0u) { let rank = ranks[ranksOffset + index];
  distance[distanceOffset + index] = compactValue[compactValueOffset + rank];
  if (programStatus[programStatusOffset + rank] != 0u) { status[statusOffset + index] = select(2u, 4u, (programStatus[programStatusOffset + rank] & 2u) != 0u); }
}
matched[matchedOffset + index] = select(0u, 1u, status[statusOffset + index] == 0u && distance[distanceOffset + index] ${subsequence ? '>=' : '<='} threshold[thresholdOffset]);`
      })
    );
    if (props.topK !== undefined && props.output.topKIndices && props.output.topKSummary)
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-top-k`,
          operation: 'GPUTrackSimilaritySearch',
          variant: 'exact-stable-top-k',
          invocationCount: props.linesA.lineOffsets.length - 1,
          bindings: [
            {name: 'pairA', view: props.pairA, type: 'u32', access: 'read'},
            {name: 'values', view: value, type: 'f32', access: 'read'},
            {name: 'matched', view: props.output.matched, type: 'u32', access: 'read'},
            {name: 'status', view: props.output.status, type: 'u32', access: 'read'},
            {name: 'selected', view: props.output.topKIndices, type: 'u32', access: 'read_write'},
            {name: 'summary', view: props.output.topKSummary, type: 'u32', access: 'read_write'}
          ],
          body: `var complete = 1u; var count = 0u;
for (var row = 0u; row < ${props.pairA.length}u; row++) { if (pairA[pairAOffset + row] == index && status[statusOffset + row] > 1u) { complete = 0u; } }
for (var slot = 0u; slot < ${props.topK}u; slot++) {
  var best = 0xffffffffu; var bestValue = 0.0;
  for (var row = 0u; row < ${props.pairA.length}u; row++) {
    if (pairA[pairAOffset + row] != index || matched[matchedOffset + row] == 0u) { continue; }
    var used = false;
    for (var previous = 0u; previous < slot; previous++) { if (selected[selectedOffset + index * ${props.topK}u + previous] == row) { used = true; } }
    if (used) { continue; } let candidate = values[valuesOffset + row];
    if (best == 0xffffffffu || candidate ${subsequence ? '>' : '<'} bestValue) { best = row; bestValue = candidate; }
  }
  selected[selectedOffset + index * ${props.topK}u + slot] = best;
  if (best != 0xffffffffu) { count++; }
}
summary[summaryOffset + index * 2u] = count; summary[summaryOffset + index * 2u + 1u] = complete;`
        })
      );
    return nodes;
  }
}
