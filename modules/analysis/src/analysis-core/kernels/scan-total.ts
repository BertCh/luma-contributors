// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScanUint64,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {validateMaximumCount} from './gpu-contributor-utils';
import {createFillNode, createWGSLKernelNode} from './wgsl-kernel-nodes';

/**
 * Value of a saturated scan total: the true sum of the counts reached `2^32`. It is above every
 * capacity (a uint32 capacity is at most `2^32 - 2` once a 128 MiB binding holds its slots), so a
 * `total > capacity` check reports overflow and a `requiredCount` copied from it saturates.
 *
 * @internal
 */
export const SATURATED_SCAN_TOTAL = 0xffffffff;

/** Properties for the count-scan and scan-total helpers. @internal */
export type ScanTotalProps = {
  /** Node ID prefix. */
  id: string;
  /** Operation name reported in the workload estimate. */
  operation: string;
  /** Per-row uint32 counts. */
  counts: GraphDataView<'uint32'>;
  /** Exclusive offsets: read by total-only helpers, written by the saturated count scan. */
  offsets: GraphDataView<'uint32'>;
  /** Number of scanned rows, at most the length of both views. */
  rowCount: number;
  /** uint32 view holding the total word. */
  total: GraphDataView<'uint32'>;
  /** Row of `total` holding the total. Defaults to 0. */
  totalRow?: number;
};

/**
 * Saturates the total of an exclusive uint32 count scan whose true sum reached `2^32` (S-8).
 *
 * `GPUScan` wraps modulo `2^32`, so when the counts sum to `2^32` or more the usual total
 * `offsets[last] + counts[last]` is a small number, a `total > capacity` check passes and later
 * rows write over earlier slots with no overflow flag. A count is below `2^32`, so the true prefix
 * crosses a multiple of `2^32` inside row `i` exactly when `offsets[i] + counts[i] < offsets[i]` in
 * uint32 arithmetic, and the true sum reaches `2^32` exactly when some row carries. This node checks
 * every row and stores {@link SATURATED_SCAN_TOTAL} into `total[totalRow]` when one does.
 *
 * Run it after the node that writes the wrapped total and before every node that reads it.
 *
 * @internal
 */
export function createScanCarryNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: ScanTotalProps
): GPUCommandNode<Parameters> {
  // Callers scan `rowCount` rows of both views and own the total row.
  validateMaximumCount(props.id, 'rowCount', props.rowCount, props.counts.length);
  validateMaximumCount(props.id, 'rowCount', props.rowCount, props.offsets.length);
  validateMaximumCount(props.id, 'totalRow', props.totalRow ?? 0, props.total.length - 1);
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-carry`,
    operation: props.operation,
    variant: 'scan-carry',
    bindings: [
      {name: 'counts', view: props.counts, type: 'u32', access: 'read'},
      {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
      {name: 'total', view: props.total, type: 'atomic<u32>', access: 'read_write'}
    ],
    invocationCount: props.rowCount,
    declarations: 'override TOTAL_ROW: u32;',
    constants: {TOTAL_ROW: props.totalRow ?? 0},
    body: `let offset = offsets[offsetsOffset + index];
  if (offset + counts[countsOffset + index] < offset) {
    atomicStore(&total[totalOffset + TOTAL_ROW], ${SATURATED_SCAN_TOTAL}u);
  }`
  });
}

/**
 * Writes the total of an exclusive uint32 count scan into `total[totalRow]`, saturated at
 * {@link SATURATED_SCAN_TOTAL} when the true sum of the counts reaches `2^32` (S-8): one node
 * writes `offsets[last] + counts[last]` (0 for no rows), then {@link createScanCarryNode} saturates
 * it. Run them after the scan and before every node that reads the total.
 *
 * @internal
 */
export function createSaturatedScanTotalNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: ScanTotalProps
): GPUCommandNode<Parameters>[] {
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-total`,
      operation: props.operation,
      variant: 'scan-total',
      bindings: [
        {name: 'counts', view: props.counts, type: 'u32', access: 'read'},
        {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
        {name: 'total', view: props.total, type: 'u32', access: 'read_write'}
      ],
      invocationCount: 1,
      declarations: 'override ROW_COUNT: u32;\noverride TOTAL_ROW: u32;',
      constants: {ROW_COUNT: props.rowCount, TOTAL_ROW: props.totalRow ?? 0},
      body: `var sum = 0u;
  if (ROW_COUNT > 0u) {
    sum = offsets[offsetsOffset + ROW_COUNT - 1u] + counts[countsOffset + ROW_COUNT - 1u];
  }
  total[totalOffset + TOTAL_ROW] = sum;`
    }),
    createScanCarryNode(graph, props)
  ];
}

/**
 * Saturating arithmetic for scan offsets, counts and resize bounds. Clamp operands before
 * evaluating the arithmetic: WGSL select evaluates both values, including the unselected one.
 * @internal
 */
export const SCAN_TOTAL_WGSL = /* wgsl */ `
fn getSaturatedScanSum(offset: u32, count: u32) -> u32 {
  return offset + min(count, 0xffffffffu - offset);
}
fn getSaturatedScanProduct(count: u32, multiplier: u32) -> u32 {
  if (multiplier == 0u) { return 0u; }
  let maximumCount = 0xffffffffu / multiplier;
  return select(min(count, maximumCount) * multiplier, 0xffffffffu, count > maximumCount);
}
`;

/**
 * Scans counts into saturated exclusive offsets and a saturated total. Unlike a total-only carry
 * check, this preserves monotonic offsets after any number of uint32 wraps, so a later row cannot
 * overwrite an earlier output prefix. Compose the published inclusive uint64 scan with a shift and
 * uint32 saturation. Counts must fit in uint32 and there can be at most uint32 rows, so the true sum
 * fits in uint64. Consumers must also saturate offset-plus-count arithmetic (SCAN_TOTAL_WGSL).
 *
 * @internal
 */
export function createSaturatedCountScanNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: ScanTotalProps
): GPUCommandNode<Parameters>[] {
  validateMaximumCount(props.id, 'rowCount', props.rowCount, SATURATED_SCAN_TOTAL);
  validateMaximumCount(props.id, 'rowCount', props.rowCount, props.counts.length);
  validateMaximumCount(props.id, 'rowCount', props.rowCount, props.offsets.length);
  validateMaximumCount(props.id, 'totalRow', props.totalRow ?? 0, props.total.length - 1);
  const counts = graph.createDataView(props.counts.buffer, {
    format: 'uint32',
    byteOffset: props.counts.byteOffset,
    length: props.rowCount
  });
  const highInput = createTransientView(graph, `${props.id}-high-input`, 'uint32', props.rowCount);
  const low = createTransientView(graph, `${props.id}-prefix-low`, 'uint32', props.rowCount);
  const high = createTransientView(graph, `${props.id}-prefix-high`, 'uint32', props.rowCount);
  return [
    createFillNode<Parameters>(graph, {
      id: `${props.id}-clear-high`,
      operation: props.operation,
      view: highInput,
      type: 'u32',
      value: '0u'
    }),
    ...new GPUScanUint64({
      id: props.id,
      inputLow: counts,
      inputHigh: highInput,
      outputLow: low,
      outputHigh: high
    }).getCommandNodes(graph),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-saturate`,
      operation: props.operation,
      variant: 'saturated-count-scan',
      bindings: [
        {name: 'low', view: low, type: 'u32', access: 'read'},
        {name: 'high', view: high, type: 'u32', access: 'read'},
        {name: 'offsets', view: props.offsets, type: 'u32', access: 'read_write'},
        {name: 'total', view: props.total, type: 'u32', access: 'read_write'}
      ],
      invocationCount: Math.max(1, props.rowCount),
      declarations: 'override ROW_COUNT: u32;\noverride TOTAL_ROW: u32;',
      constants: {ROW_COUNT: props.rowCount, TOTAL_ROW: props.totalRow ?? 0},
      body: `if (index == 0u) {
    var sum = 0u;
    if (ROW_COUNT > 0u) {
      sum = select(low[lowOffset + ROW_COUNT - 1u], 0xffffffffu, high[highOffset + ROW_COUNT - 1u] != 0u);
    }
    total[totalOffset + TOTAL_ROW] = sum;
  }
  if (index < ROW_COUNT) {
    var offset = 0u;
    if (index > 0u) {
      offset = select(low[lowOffset + index - 1u], 0xffffffffu, high[highOffset + index - 1u] != 0u);
    }
    offsets[offsetsOffset + index] = offset;
  }`
    })
  ];
}
