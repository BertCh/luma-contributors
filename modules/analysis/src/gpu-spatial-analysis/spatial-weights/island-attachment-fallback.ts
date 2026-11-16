// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCompaction,
  getViewBindingRange,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLActiveCountDispatch,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createStatisticsKernelNode as createWGSLKernelNode} from './statistics-kernel-node';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';

/**
 * Scaled nearest comparison shared with the reference scan. Halving before subtraction avoids
 * overflow for opposite finite coordinates; comparing at one normal scale preserves tiny norms.
 * @internal
 */
export const ISLAND_ATTACHMENT_NEAREST_DECLARATIONS = `${IS_FINITE_BITS_WGSL}
const NO_ATTACHMENT: u32 = 0xffffffffu;
fn isCloserDifference(difference: vec2<f32>, bestDifference: vec2<f32>, isFirst: bool) -> bool {
 let scale = max(max(abs(difference.x), abs(difference.y)), max(abs(bestDifference.x), abs(bestDifference.y)));
 var closer = isFirst;
 if (scale > 0.0) { let exponent = clamp(i32((bitcast<u32>(scale) >> 23u) & 255u) - 127, -120, 120); let normalizer = bitcast<f32>(u32(127 - exponent) << 23u); let normalized = difference * normalizer; let bestNormalized = bestDifference * normalizer; closer = closer || dot(normalized, normalized) < dot(bestNormalized, bestNormalized); }
 return closer;
}`;

/** Portable workgroup size; the cooperative reduction occupies 3 KiB of workgroup memory. */
export const ISLAND_ATTACHMENT_WORKGROUP_SIZE = 256;
/** Each lane tests at most 256 representatives in one dispatch, bounding watchdog work. */
export const ISLAND_ATTACHMENT_REPRESENTATIVE_BATCH_SIZE = 65536;
/** Keeps the linear invocation count and the rounded dispatch count inside uint32. */
export const ISLAND_ATTACHMENT_QUERY_BATCH_SIZE = 16777215;

/** Allows the logarithmic published compaction scan in the uint32 row space (node tests check it). */
const ISLAND_ATTACHMENT_FALLBACK_PRIMITIVE_NODE_ALLOWANCE = 256;

/** Conservative number of fallback dispatches, independent of the runtime unresolved count. */
export function getIslandAttachmentFallbackBatchCount(
  rows: number,
  queryBatchSize = ISLAND_ATTACHMENT_QUERY_BATCH_SIZE,
  representativeBatchSize = ISLAND_ATTACHMENT_REPRESENTATIVE_BATCH_SIZE
): number {
  return Math.ceil(rows / representativeBatchSize) * Math.ceil(rows / queryBatchSize);
}

/**
 * Compact unresolved queries and resolve them in bounded representative batches. A GPU cursor
 * supplies every batch base and count; all repeated nodes have identical shaders and constants.
 * Only unresolved queries launch workgroups, and zero unresolved queries launch none. Query
 * batches also bound the linear invocation space on devices with very large storage bindings.
 * @internal
 */
export function createIslandAttachmentFallbackNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: GraphDataView<'float32x2'>;
    unresolved: GraphDataView<'uint32'>;
    attachment: GraphDataView<'uint32'>;
    uncertainCount?: GraphDataView<'uint32'>;
    nearestDeclarations: string;
    /** Internal dispatch caps for small rollover tests; production uses the named maxima. */
    queryBatchSize?: number;
    representativeBatchSize?: number;
  }
): GPUCommandNode<Parameters>[] {
  const {id, positions, unresolved, attachment, nearestDeclarations} = props;
  const rows = positions.length;
  const maximumQueryBatchSize = props.queryBatchSize ?? ISLAND_ATTACHMENT_QUERY_BATCH_SIZE;
  const maximumRepresentativeBatchSize =
    props.representativeBatchSize ?? ISLAND_ATTACHMENT_REPRESENTATIVE_BATCH_SIZE;
  validateCapacity(
    id,
    'queryBatchSize',
    maximumQueryBatchSize,
    1,
    ISLAND_ATTACHMENT_QUERY_BATCH_SIZE
  );
  validateCapacity(
    id,
    'representativeBatchSize',
    maximumRepresentativeBatchSize,
    1,
    ISLAND_ATTACHMENT_REPRESENTATIVE_BATCH_SIZE
  );
  const queryBatchSize = Math.min(rows, maximumQueryBatchSize);
  const representativeBatchSize = Math.min(rows, maximumRepresentativeBatchSize);
  const batchCount = getIslandAttachmentFallbackBatchCount(
    rows,
    queryBatchSize,
    representativeBatchSize
  );
  validateCommandNodeBudget(
    id,
    ISLAND_ATTACHMENT_FALLBACK_PRIMITIVE_NODE_ALLOWANCE + 3 * batchCount
  );
  for (const [name, view] of [
    ['positions', positions],
    ['unresolved', unresolved],
    ['attachment', attachment],
    ['uncertainCount', props.uncertainCount]
  ] as const) {
    if (view)
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        getViewBindingRange(view).size,
        'reduce rows'
      );
  }
  for (const [name, byteLength] of [
    ['fallback rows', rows * 4],
    ['fallback cursor', 20],
    ['fallback dispatch', 12]
  ] as const) {
    validateGPUStorageBindingByteLength(graph.device, id, name, byteLength, 'reduce rows');
  }
  const rowIds = createTransientView(graph, `${id}-fallback-row-ids`, 'uint32', rows);
  const queries = createTransientView(graph, `${id}-fallback-queries`, 'uint32', rows);
  const count =
    props.uncertainCount ?? createTransientView(graph, `${id}-fallback-count`, 'uint32', 1);
  // representative base, query base, current query count, representative cap, query cap.
  const cursor = createTransientView(graph, `${id}-fallback-cursor`, 'uint32', 5);
  const binding = (
    name: string,
    view: GraphDataView,
    access: 'read' | 'read_write' = 'read',
    type: 'f32' | 'u32' = 'u32'
  ): WGSLKernelBinding => ({name, view, access, type});
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-fallback-row-ids`,
      operation: 'GPUIslandAttachment',
      invocationCount: rows,
      bindings: [binding('rowIds', rowIds, 'read_write')],
      body: 'rowIds[rowIdsOffset + index] = index;'
    }),
    ...new GPUCompaction({
      id: `${id}-fallback-compact`,
      input: rowIds,
      flags: unresolved,
      output: queries,
      count
    }).getCommandNodes(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-fallback-initialize`,
      operation: 'GPUIslandAttachment',
      invocationCount: 1,
      bindings: [binding('cursor', cursor, 'read_write'), binding('count', count)],
      body: `cursor[cursorOffset] = 0u;
cursor[cursorOffset + 1u] = 0u;
cursor[cursorOffset + 2u] = min(count[countOffset], ${queryBatchSize}u);
cursor[cursorOffset + 3u] = ${representativeBatchSize}u;
cursor[cursorOffset + 4u] = ${queryBatchSize}u;`
    })
  ];
  for (let batch = 0; batch < batchCount; batch++) {
    const dispatch = createWGSLActiveCountDispatch(graph, {
      id: `${id}-fallback-dispatch-${batch}`,
      operation: 'GPUIslandAttachment',
      count: cursor,
      countRow: 2,
      maximumItemCount: queryBatchSize,
      invocationsPerItem: ISLAND_ATTACHMENT_WORKGROUP_SIZE,
      workgroupSize: ISLAND_ATTACHMENT_WORKGROUP_SIZE
    });
    nodes.push(
      dispatch.updateNode,
      createWGSLKernelNode(graph, {
        id: `${id}-resolve-unproven-${batch}`,
        operation: 'GPUIslandAttachment',
        variant: 'resolve-unproven',
        invocationCount: queryBatchSize * ISLAND_ATTACHMENT_WORKGROUP_SIZE,
        workgroupSize: ISLAND_ATTACHMENT_WORKGROUP_SIZE,
        guardIndex: false,
        condition: dispatch.condition,
        bindings: [
          binding('positions', positions, 'read', 'f32'),
          binding('queries', queries),
          binding('cursor', cursor),
          binding('attachment', attachment, 'read_write')
        ],
        declarations: `${nearestDeclarations}
const WORKGROUP_SIZE: u32 = ${ISLAND_ATTACHMENT_WORKGROUP_SIZE}u;
var<workgroup> nearestRows: array<u32, ${ISLAND_ATTACHMENT_WORKGROUP_SIZE}>;
var<workgroup> nearestDifferences: array<vec2<f32>, ${ISLAND_ATTACHMENT_WORKGROUP_SIZE}>;`,
        // Padded workgroups participate in every barrier; only in-range queries read/write rows.
        body: `let querySlot = workgroupIndex;
let lane = localInvocationIndex;
let inRange = querySlot < cursor[cursorOffset + 2u];
let rowCount = (arrayLength(&positions) - positionsOffset) / 2u;
let representativeBase = cursor[cursorOffset];
let representativeCount = min(cursor[cursorOffset + 3u], rowCount - representativeBase);
var island = NO_ATTACHMENT;
var position = vec2<f32>(0.0);
if (inRange) {
 island = queries[queriesOffset + cursor[cursorOffset + 1u] + querySlot];
 position = vec2<f32>(positions[positionsOffset + 2u * island], positions[positionsOffset + 2u * island + 1u]);
}
var nearest = NO_ATTACHMENT;
var bestDifference = vec2<f32>(0.0);
if (inRange) {
 for (var slot = lane; slot < representativeCount; slot += WORKGROUP_SIZE) {
  let other = representativeBase + slot;
  let candidate = vec2<f32>(positions[positionsOffset + 2u * other], positions[positionsOffset + 2u * other + 1u]);
  if (other != island && isFiniteBits(candidate.x) && isFiniteBits(candidate.y)) {
   let difference = position * 0.5 - candidate * 0.5;
   if (isCloserDifference(difference, bestDifference, nearest == NO_ATTACHMENT)) {
    nearest = other; bestDifference = difference;
   }
  }
 }
}
nearestRows[lane] = nearest;
nearestDifferences[lane] = bestDifference;
workgroupBarrier();
for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
 if (lane < stride) {
  let other = nearestRows[lane + stride];
  let difference = nearestDifferences[lane + stride];
  nearest = nearestRows[lane];
  bestDifference = nearestDifferences[lane];
  if (other != NO_ATTACHMENT &&
      (isCloserDifference(difference, bestDifference, nearest == NO_ATTACHMENT) ||
       (other < nearest && !isCloserDifference(bestDifference, difference, false)))) {
   nearestRows[lane] = other;
   nearestDifferences[lane] = difference;
  }
 }
 workgroupBarrier();
}
if (inRange && lane == 0u) {
 nearest = nearestRows[0];
 bestDifference = nearestDifferences[0];
 let previous = attachment[attachmentOffset + island];
 if (previous != NO_ATTACHMENT) {
  let candidate = vec2<f32>(positions[positionsOffset + 2u * previous], positions[positionsOffset + 2u * previous + 1u]);
  let difference = position * 0.5 - candidate * 0.5;
  if (isCloserDifference(difference, bestDifference, nearest == NO_ATTACHMENT) ||
      (previous < nearest && !isCloserDifference(bestDifference, difference, false))) { nearest = previous; }
 }
 attachment[attachmentOffset + island] = nearest;
}`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-fallback-advance-${batch}`,
        operation: 'GPUIslandAttachment',
        invocationCount: 1,
        bindings: [
          binding('cursor', cursor, 'read_write'),
          binding('count', count),
          binding('positions', positions, 'read', 'f32')
        ],
        body: `let rowCount = (arrayLength(&positions) - positionsOffset) / 2u;
let nextRepresentative = cursor[cursorOffset] + min(cursor[cursorOffset + 3u], rowCount - cursor[cursorOffset]);
if (nextRepresentative == rowCount) {
 cursor[cursorOffset] = 0u;
 let queryBase = min(count[countOffset], cursor[cursorOffset + 1u] + cursor[cursorOffset + 4u]);
 cursor[cursorOffset + 1u] = queryBase;
 cursor[cursorOffset + 2u] = min(cursor[cursorOffset + 4u], count[countOffset] - queryBase);
} else { cursor[cursorOffset] = nextRepresentative; }`
      })
    );
  }
  return nodes;
}
