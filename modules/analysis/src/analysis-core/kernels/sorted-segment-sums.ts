// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandGraphNodeCondition,
  type GPUCommandNode,
  type GraphBufferUse,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from './wgsl-kernel-nodes';
import {getSortedSegmentOffsetNodes} from './sorted-segment-offsets';

/** Threads per segment-sum workgroup; shared with kernels that reuse the tree reduction. @internal */
export const SEGMENT_WORKGROUP_SIZE = 256;

/** Module-scope WGSL declarations for the workgroup partial sums used by {@link getSegmentTreeSumWGSL}. @internal */
export const SEGMENT_PARTIAL_SUMS_WGSL = `var<workgroup> partialSums: array<f32, ${SEGMENT_WORKGROUP_SIZE}>;`;

/**
 * WGSL statements that store `partialSum`, then reduce `partialSums` with the fixed binary tree
 * (strides 128 to 1). Shared by the segment-sum kernels so they stay bit-identical. Every
 * invocation of the workgroup must reach it (it contains barriers); the total is left in
 * `partialSums[0]`. @internal
 */
export function getSegmentTreeSumWGSL(): string {
  return `partialSums[localInvocationIndex] = partialSum;
  workgroupBarrier();
  for (var stride = ${SEGMENT_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }`;
}

/**
 * Returns the `[invocationCount, workgroupSize]` dispatch shape of the {@link createSegmentSumNode}
 * kernel, for the slot of a `createWGSLDispatchGate` that gates a segment sum. @internal
 */
export function getSegmentSumDispatchShape(
  segmentCount: number
): readonly [invocationCount: number, workgroupSize: number] {
  return [segmentCount * SEGMENT_WORKGROUP_SIZE, SEGMENT_WORKGROUP_SIZE];
}

/** Returns the number of key bits that hold values in `[0, maximumKey]` (at least one). @internal */
export function getSortKeyBits(maximumKey: number): number {
  return Math.max(1, Math.floor(maximumKey).toString(2).length);
}

/**
 * One deterministic per-segment sum computed by {@link getSortedSegmentSumNodes}.
 *
 * @internal
 */
export type SortedSegmentReduction = {
  /** Short name used in generated node and transient IDs. Must be unique per call. */
  name: string;
  /** Per-row contributions aligned with `segmentKeys`; invalid rows contribute 0. */
  contributions: GraphDataView<'float32'>;
  /** Per-segment sums (`segmentCount` rows). */
  output: GraphDataView<'float32'>;
};

/**
 * Shared sorted segmented reduction used by zonal statistics, raster zonal statistics, flow
 * aggregation and spatial clustering. Reduces per-segment sums in a fixed order: stable sort by
 * segment key, segment offsets by binary search over the sorted keys, gather of contributions in
 * sorted order, and one segmented sum, so the results are bitwise reproducible.
 *
 * `segmentKeys` holds a segment key per row; keys at or above `segmentCount` sort last and never
 * contribute. Callers need no per-segment counts (no atomics, no scan).
 *
 * @internal
 */
export function getSortedSegmentSumNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    /** Operation name reported in workload estimates. */
    operation: string;
    segmentCount: number;
    segmentKeys: GraphDataView<'uint32'>;
    sumContributions?: GraphDataView<'float32'>;
    weightContributions?: GraphDataView<'float32'>;
    sums?: GraphDataView<'float32'>;
    weightSums?: GraphDataView<'float32'>;
    /** Additional named reductions sharing the same sort and segment offsets. */
    reductions?: readonly SortedSegmentReduction[];
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation, segmentCount, segmentKeys} = props;
  const pointCount = segmentKeys.length;
  const nodes: GPUCommandNode<Parameters>[] = [];
  const sortKeys = createTransientView(graph, `${id}-sort-keys`, 'uint32', pointCount);
  const sortIndices = createTransientView(graph, `${id}-sort-indices`, 'uint32', pointCount);
  const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', pointCount);
  const sortedIndices = createTransientView(graph, `${id}-sorted-indices`, 'uint32', pointCount);
  // Unassigned and out-of-range rows share the key `segmentCount`, which sorts after every segment.
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-sort-prepare`,
      operation,
      variant: 'sort-prepare',
      bindings: [
        {name: 'pointFeatureRows', view: segmentKeys, type: 'u32', access: 'read'},
        {name: 'sortKeys', view: sortKeys, type: 'u32', access: 'read_write'},
        {name: 'sortIndices', view: sortIndices, type: 'u32', access: 'read_write'}
      ],
      invocationCount: pointCount,
      declarations: `const FEATURE_COUNT: u32 = ${segmentCount}u;`,
      body: `sortKeys[sortKeysOffset + index] = min(pointFeatureRows[pointFeatureRowsOffset + index], FEATURE_COUNT);
  sortIndices[sortIndicesOffset + index] = index;`
    })
  );
  nodes.push(
    ...new GPUSort({
      id: `${id}-sort`,
      keys: sortKeys,
      values: sortIndices,
      outputKeys: sortedKeys,
      outputValues: sortedIndices,
      keyBits: getSortKeyBits(segmentCount)
    }).getCommandNodes(graph)
  );

  // The keys are sorted, so each segment offset is the lower bound of its key: one binary search
  // per segment replaces the per-row atomic counting, the scan and the total fixup.
  const segmentOffsets = createTransientView(
    graph,
    `${id}-segment-offsets`,
    'uint32',
    segmentCount + 1
  );
  nodes.push(
    ...getSortedSegmentOffsetNodes<Parameters>(graph, {
      id,
      operation,
      groupCount: segmentCount,
      sortedGroupKeys: sortedKeys,
      segmentOffsets
    })
  );

  const reductions: SortedSegmentReduction[] = [];
  if (props.sums && props.sumContributions) {
    reductions.push({name: 'sums', contributions: props.sumContributions, output: props.sums});
  }
  if (props.weightSums && props.weightContributions) {
    reductions.push({
      name: 'weight-sums',
      contributions: props.weightContributions,
      output: props.weightSums
    });
  }
  reductions.push(...(props.reductions ?? []));
  for (const reduction of reductions) {
    const sortedContributions = createTransientView(
      graph,
      `${id}-sorted-${reduction.name}`,
      'float32',
      pointCount
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-gather-${reduction.name}`,
        operation,
        variant: 'gather',
        bindings: [
          {name: 'sortedIndices', view: sortedIndices, type: 'u32', access: 'read'},
          {name: 'contributions', view: reduction.contributions, type: 'f32', access: 'read'},
          {
            name: 'sortedContributions',
            view: sortedContributions,
            type: 'f32',
            access: 'read_write'
          }
        ],
        invocationCount: pointCount,
        body: `sortedContributions[sortedContributionsOffset + index] =
    contributions[contributionsOffset + sortedIndices[sortedIndicesOffset + index]];`
      })
    );
    nodes.push(
      createSegmentSumNode<Parameters>(graph, {
        id: `${id}-reduce-${reduction.name}`,
        operation,
        segmentCount,
        input: sortedContributions,
        segmentOffsets,
        output: reduction.output,
        // Offsets are lower bounds in sorted keys, so they never exceed `pointCount`.
        offsetsWithinInput: true
      })
    );
  }
  return nodes;
}

/**
 * Sums each segment of `input` with one 256-thread workgroup: a strided per-thread partial sum
 * followed by a fixed binary tree. The order depends only on the segment bounds, so the result is
 * bitwise reproducible. Segments are workgroups of a bounded 3D dispatch, so the segment count is
 * not limited, and `end` is clamped to `input.length`. The segment count is always a build-time
 * number here; there is no indirect (GPU-only count) variant.
 *
 * With a `condition` (the sums of an unrolled loop with a data-dependent exit skip converged
 * rounds) the node is gated. An indirect condition supplies the dispatch dimensions of the node,
 * so its triple must hold the layout of {@link getSegmentSumDispatchShape}.
 *
 * `offsetsWithinInput` and `implementation` are accepted for call-site stability and ignored:
 * published luma.gl does not export `GPUSegmentedReduction`, so the local kernel always runs.
 *
 * @internal
 */
export function createSegmentSumNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    /** Operation name reported in workload estimates. */
    operation: string;
    segmentCount: number;
    input: GraphDataView<'float32'>;
    segmentOffsets: GraphDataView<'uint32'>;
    output: GraphDataView<'float32'>;
    /** Ignored; retained for call-site stability. */
    offsetsWithinInput?: boolean;
    /** Ignored; retained for call-site stability. */
    implementation?: 'auto' | 'kernel';
    /** GPU-state gate of the node. */
    condition?: GPUCommandGraphNodeCondition<Parameters>;
    /** Resources the `condition` reads (for example the `'indirect'` use of its dispatch buffer). */
    extraResources?: readonly GraphBufferUse[];
  }
): GPUCommandNode<Parameters> {
  // The offsets and output views must cover every segment; the kernel reads and
  // write `segmentCount + 1` offsets and `segmentCount` sums without bounds checks of their own.
  assert(props.segmentOffsets.length >= props.segmentCount + 1);
  assert(props.output.length >= props.segmentCount);
  return createSegmentSumKernelNode(graph, props);
}

function createSegmentSumKernelNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    segmentCount: number;
    input: GraphDataView<'float32'>;
    segmentOffsets: GraphDataView<'uint32'>;
    output: GraphDataView<'float32'>;
    condition?: GPUCommandGraphNodeCondition<Parameters>;
    extraResources?: readonly GraphBufferUse[];
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'segment-sum',
    condition: props.condition,
    extraResources: props.extraResources,
    bindings: [
      {name: 'segmentInput', view: props.input, type: 'f32', access: 'read'},
      {name: 'segmentOffsets', view: props.segmentOffsets, type: 'u32', access: 'read'},
      {name: 'segmentOutput', view: props.output, type: 'f32', access: 'read_write'}
    ],
    workgroupSize: SEGMENT_WORKGROUP_SIZE,
    invocationCount: props.segmentCount * SEGMENT_WORKGROUP_SIZE,
    guardIndex: false,
    declarations: `const INPUT_COUNT: u32 = ${props.input.length}u;
${SEGMENT_PARTIAL_SUMS_WGSL}`,
    // No early return: every invocation of a workgroup must reach the barriers.
    body: `let segment = index / ${SEGMENT_WORKGROUP_SIZE}u;
  let isInRange = index < INVOCATION_COUNT;
  var partialSum = 0.0;
  if (isInRange) {
    let begin = segmentOffsets[segmentOffsetsOffset + segment];
    let end = min(segmentOffsets[segmentOffsetsOffset + segment + 1u], INPUT_COUNT);
    for (var row = begin + localInvocationIndex; row < end; row += ${SEGMENT_WORKGROUP_SIZE}u) {
      partialSum += segmentInput[segmentInputOffset + row];
    }
  }
  ${getSegmentTreeSumWGSL()}
  if (isInRange && localInvocationIndex == 0u) {
    segmentOutput[segmentOutputOffset + segment] = partialSums[0];
  }`
  });
}
