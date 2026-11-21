// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getBoundedDispatchLayout,
  GPUGather,
  GPUSort,
  GPUSegmentedReduction,
  type GPUCommandGraph,
  type GPUCommandGraphNodeCondition,
  type GPUCommandNode,
  type GraphBufferUse,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement,
  validateGPUStorageBindingByteLength,
  validateSinglePackedView,
  validateGraphViewsBelongToGraph
} from './gpu-contributor-utils';
import {validateGraphChunkBindingRanges} from './graph-chunk-dispatch';
import {
  createWGSLKernelNode,
  getAutomaticWGSLKernelWorkgroupSize,
  isRecordingWGSLKernelNodes
} from './wgsl-kernel-nodes';
import {getSortedSegmentOffsetNodes} from './sorted-segment-offsets';

/** Reduction applied by the segment tree kernel. @internal */
export type SegmentReduction = 'sum' | 'min' | 'max';

const SEGMENT_REDUCTION_IDENTITY: Record<SegmentReduction, string> = {
  sum: '0.0',
  min: '3.402823466e+38',
  max: '-3.402823466e+38'
};

/** Threads per segment-sum workgroup; shared with kernels that reuse the tree reduction. @internal */
export const SEGMENT_WORKGROUP_SIZE = 256;

/** Most segment workgroups dispatched at once; more segments are reduced in strided passes. */
const MAXIMUM_SEGMENT_WORKGROUPS = Math.floor(0xffffffff / SEGMENT_WORKGROUP_SIZE);

/** Module-scope WGSL declarations for the workgroup partial sums used by {@link getSegmentTreeSumWGSL}. @internal */
export const SEGMENT_PARTIAL_SUMS_WGSL = `var<workgroup> partialSums: array<f32, ${SEGMENT_WORKGROUP_SIZE}>;`;

/**
 * WGSL statements that store `partialSum`, then reduce `partialSums` with the fixed binary tree
 * (strides 128 to 1). Shared by the segment-sum kernels so they stay bit-identical. Every
 * invocation of the workgroup must reach it (it contains barriers); the total is left in
 * `partialSums[0]`. @internal
 */
export function getSegmentTreeSumWGSL(reduction: SegmentReduction = 'sum'): string {
  const combine =
    reduction === 'sum'
      ? 'partialSums[localInvocationIndex] + partialSums[localInvocationIndex + stride]'
      : `${reduction}(partialSums[localInvocationIndex], partialSums[localInvocationIndex + stride])`;
  return `partialSums[localInvocationIndex] = partialSum;
  workgroupBarrier();
  for (var stride = ${SEGMENT_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] = ${combine};
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
  return [
    Math.min(segmentCount, MAXIMUM_SEGMENT_WORKGROUPS) * SEGMENT_WORKGROUP_SIZE,
    SEGMENT_WORKGROUP_SIZE
  ];
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
  /** Covers every key row; word-aligned float32 strides are supported. Invalid keys contribute 0. */
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
 * contribute. Keys and outputs must be packed scalar views. Contributions must cover every key
 * row and may use word-aligned float32 strides. Empty runs do not bind contribution storage.
 * Callers need no per-segment counts (no atomics, no scan).
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
  validateCapacity(id, 'segmentCount', segmentCount, 0);
  validateRequirement(
    id,
    props.sums === undefined || props.sumContributions !== undefined,
    'sums requires sumContributions'
  );
  validateRequirement(
    id,
    props.weightSums === undefined || props.weightContributions !== undefined,
    'weightSums requires weightContributions'
  );
  const pointCount = segmentKeys.length;
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
  // Validate every reduction before the first graph declaration, so a refused call reserves no
  // resource IDs and a corrected call with the same ID can follow.
  validateSinglePackedView(segmentKeys, ['uint32'], `${id} segmentKeys`);
  const reductionNames = new Set<string>();
  for (const reduction of reductions) {
    validateRequirement(
      id,
      !reductionNames.has(reduction.name),
      `reduction name "${reduction.name}" must be unique`
    );
    reductionNames.add(reduction.name);
    validateSinglePackedView(reduction.output, ['float32'], `${id} ${reduction.name} output`);
    validateMaximumCount(
      id,
      `${reduction.name} contributions required rows`,
      pointCount,
      reduction.contributions.length
    );
    // Gather permits strided scalar sources, provided every row is word aligned. Its constructor
    // runs even for empty indices; validate that layout here before declaring any shared scratch.
    const contributions = reduction.contributions;
    validateRequirement(
      id,
      contributions.format === 'float32' &&
        contributions.rowByteLength === 4 &&
        contributions.byteStride >= 4 &&
        contributions.byteStride % 4 === 0 &&
        contributions.byteOffset % 4 === 0,
      `${reduction.name} contributions must have uint32-aligned float32 rows and word-aligned stride`
    );
    validateGraphViewsBelongToGraph(id, graph, [contributions]);
    validateMaximumCount(
      `${id}-reduce-${reduction.name}`,
      'segmentCount',
      segmentCount,
      reduction.output.length
    );
  }
  // Caller views can be larger than the active prefix, or have an unaligned binding prefix.
  // Nonempty gather and the WGSL nodes bind those whole ranges. Check them, and both scratch shapes,
  // before reserving the sort IDs so a corrected same-ID call remains possible.
  validateGraphChunkBindingRanges(graph, id, {
    segmentKeys,
    ...Object.fromEntries(
      reductions.flatMap(reduction => [
        ...(pointCount > 0 ? [[`${reduction.name} contributions`, reduction.contributions]] : []),
        [`${reduction.name} output`, reduction.output]
      ])
    )
  });
  for (const [name, length] of [
    ['sort scratch', Math.max(pointCount, 1)],
    ['segment offsets', segmentCount + 1]
  ] as const) {
    const byteLength = length * Uint32Array.BYTES_PER_ELEMENT;
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      byteLength,
      'reduce the row or segment count'
    );
    validateMaximumCount(id, `${name} byteLength`, byteLength, graph.device.limits.maxBufferSize);
  }
  validateMaximumCount(
    id,
    'sort preparation storage bindings',
    3,
    graph.device.limits.maxStorageBuffersPerShaderStage
  );
  // GPUSort's internally generated views are packed, equally sized, disjoint and graph-owned.
  // Its radix histogram/scan scratch (at most 16 * ceil(pointCount / 256) words) is no larger
  // than sort scratch on the radix route (pointCount > 256).
  validateMaximumCount(id, 'pointCount', pointCount, 0x80000000);
  for (const invocationCount of [pointCount, segmentCount + 1]) {
    getBoundedDispatchLayout(
      `${id} ${operation}`,
      Math.max(invocationCount, 1),
      getAutomaticWGSLKernelWorkgroupSize(invocationCount),
      graph.device.limits.maxComputeWorkgroupsPerDimension
    );
  }
  if (reductions.length > 0) {
    const [invocationCount, workgroupSize] = getSegmentSumDispatchShape(segmentCount);
    getBoundedDispatchLayout(
      `${id} ${operation}`,
      Math.max(invocationCount, 1),
      workgroupSize,
      graph.device.limits.maxComputeWorkgroupsPerDimension
    );
  }
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

  for (const reduction of reductions) {
    const sortedContributions = createTransientView(
      graph,
      `${id}-reduction-${reduction.name}-sorted`,
      'float32',
      pointCount
    );
    nodes.push(
      // `sortedIndices` is a permutation of the rows, so no index is out of range.
      ...new GPUGather({
        id: `${id}-gather-${reduction.name}`,
        source: reduction.contributions,
        indices: sortedIndices,
        output: sortedContributions
      }).getCommandNodes(graph),
      createSegmentSumNode<Parameters>(graph, {
        id: `${id}-reduce-${reduction.name}`,
        operation,
        segmentCount,
        input: sortedContributions,
        segmentOffsets,
        output: reduction.output
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
 * Ungated sums with packed scalar views and a separate output buffer delegate to the published
 * GPUSegmentedReduction with active offset/output prefixes. Zero segments, other reductions,
 * aliased storage, conditions, extra resources and recording graphs retain the local node and its
 * dispatch shape. Public dispatches whose padded invocation count exceeds uint32 also stay local.
 * All three direct views must be packed scalars; use the sorted-gather helper for strided input.
 *
 * With a `condition` (the sums of an unrolled loop with a data-dependent exit skip converged
 * rounds) the node is gated. An indirect condition supplies the dispatch dimensions of the node,
 * so its triple must hold the layout of {@link getSegmentSumDispatchShape}.
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
    /** Reduction operation; defaults to `'sum'`. Empty segments write 0 for every operation. */
    reduction?: SegmentReduction;
    /** GPU-state gate of the node. */
    condition?: GPUCommandGraphNodeCondition<Parameters>;
    /** Resources the `condition` reads (for example the `'indirect'` use of its dispatch buffer). */
    extraResources?: readonly GraphBufferUse[];
  }
): GPUCommandNode<Parameters> {
  validateCapacity(props.id, 'segmentCount', props.segmentCount, 0);
  validateSinglePackedView(props.input, ['float32'], `${props.id} input`);
  validateSinglePackedView(props.segmentOffsets, ['uint32'], `${props.id} segmentOffsets`);
  validateSinglePackedView(props.output, ['float32'], `${props.id} output`);
  // The offsets and output views must cover every segment; the kernel reads and
  // write `segmentCount + 1` offsets and `segmentCount` sums without bounds checks of their own.
  validateMaximumCount(
    props.id,
    'segmentCount + 1',
    props.segmentCount + 1,
    props.segmentOffsets.length
  );
  validateMaximumCount(props.id, 'segmentCount', props.segmentCount, props.output.length);
  // Keep the local declaration preflight for every route (binding ranges and dispatch limits
  // included). It reserves no resources. The published sum uses exactly the same
  // per-lane strided additions and 128..1 tree; gates and extra uses remain analysis-owned.
  const localNode = createSegmentSumKernelNode(graph, props);
  if (
    props.segmentCount > 0 &&
    props.segmentCount <= MAXIMUM_SEGMENT_WORKGROUPS &&
    !isRecordingWGSLKernelNodes(graph) &&
    (props.reduction ?? 'sum') === 'sum' &&
    props.condition === undefined &&
    props.extraResources === undefined &&
    props.input.byteStride === 4 &&
    props.input.rowByteLength === 4 &&
    props.segmentOffsets.byteStride === 4 &&
    props.segmentOffsets.rowByteLength === 4 &&
    props.output.byteStride === 4 &&
    props.output.rowByteLength === 4 &&
    props.output.buffer !== props.input.buffer &&
    props.output.buffer !== props.segmentOffsets.buffer
  ) {
    // This public planner gives exactly the published reducer's workgroup layout when the
    // logical invocation count is segmentCount * 256. Its padded product can exceed uint32
    // even though the logical count fits. The local kernel guards and strides that case.
    const layout = getBoundedDispatchLayout(
      props.id,
      props.segmentCount * SEGMENT_WORKGROUP_SIZE,
      SEGMENT_WORKGROUP_SIZE,
      graph.device.limits.maxComputeWorkgroupsPerDimension
    );
    if (layout.x * layout.y * layout.z * SEGMENT_WORKGROUP_SIZE > 0xffffffff) return localNode;
    const [node] = new GPUSegmentedReduction({
      id: props.id,
      input: props.input,
      segmentOffsets: graph.createDataView(props.segmentOffsets.buffer, {
        format: 'uint32',
        length: props.segmentCount + 1,
        byteOffset: props.segmentOffsets.byteOffset
      }),
      output: graph.createDataView(props.output.buffer, {
        format: 'float32',
        length: props.segmentCount,
        byteOffset: props.output.byteOffset
      }),
      operation: 'sum'
    }).getCommandNodes(graph);
    return {...node, workload: {...node.workload, operation: props.operation}};
  }
  return localNode;
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
    reduction?: SegmentReduction;
    condition?: GPUCommandGraphNodeCondition<Parameters>;
    extraResources?: readonly GraphBufferUse[];
  }
): GPUCommandNode<Parameters> {
  const reduction = props.reduction ?? 'sum';
  const combine = reduction === 'sum' ? 'partialSum + next' : `${reduction}(partialSum, next)`;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: `segment-${reduction}`,
    condition: props.condition,
    extraResources: props.extraResources,
    bindings: [
      {name: 'segmentInput', view: props.input, type: 'f32', access: 'read'},
      {name: 'segmentOffsets', view: props.segmentOffsets, type: 'u32', access: 'read'},
      {name: 'segmentOutput', view: props.output, type: 'f32', access: 'read_write'}
    ],
    workgroupSize: SEGMENT_WORKGROUP_SIZE,
    invocationCount:
      Math.min(props.segmentCount, MAXIMUM_SEGMENT_WORKGROUPS) * SEGMENT_WORKGROUP_SIZE,
    guardIndex: false,
    declarations: `const INPUT_COUNT: u32 = ${props.input.length}u;
const SEGMENT_COUNT: u32 = ${props.segmentCount}u;
${SEGMENT_PARTIAL_SUMS_WGSL}`,
    // No early return: every invocation of a workgroup must reach the barriers.
    body: `let firstSegment = index / ${SEGMENT_WORKGROUP_SIZE}u;
  let segmentStride = INVOCATION_COUNT / ${SEGMENT_WORKGROUP_SIZE}u;
  // Segment counts beyond one dispatch are reduced in strided passes with a uniform trip count.
  for (var segmentPass = 0u; segmentPass < ${Math.ceil(props.segmentCount / MAXIMUM_SEGMENT_WORKGROUPS)}u; segmentPass++) {
  let segment = firstSegment + segmentPass * segmentStride;
  let isInRange = index < INVOCATION_COUNT && segment < SEGMENT_COUNT;
  var partialSum = ${SEGMENT_REDUCTION_IDENTITY[reduction]};
  var isEmpty = true;
  if (isInRange) {
    let begin = segmentOffsets[segmentOffsetsOffset + segment];
    let end = min(segmentOffsets[segmentOffsetsOffset + segment + 1u], INPUT_COUNT);
    for (var row = begin + localInvocationIndex; row < end; row += ${SEGMENT_WORKGROUP_SIZE}u) {
      let next = segmentInput[segmentInputOffset + row];
      partialSum = ${combine};
    }
    isEmpty = begin >= end;
  }
  ${getSegmentTreeSumWGSL(reduction)}
  if (isInRange && localInvocationIndex == 0u) {
    segmentOutput[segmentOutputOffset + segment] = select(partialSums[0], 0.0, isEmpty);
  }
  workgroupBarrier();
  }`
  });
}
