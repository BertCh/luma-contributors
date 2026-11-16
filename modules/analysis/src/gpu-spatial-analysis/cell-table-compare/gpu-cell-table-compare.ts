// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {clamp} from '@math.gl/core';
import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  getCellDispatchBuffer,
  validateCellDispatchOutputs
} from '../cell-indexing/cell-dispatch-validation';
import type {GPUCardinalityPort, GPUOverflowStatusPort} from '../../analysis-core/contracts/status';
import {
  getCellTableViews,
  validateCellTable,
  type GPUCellTable
} from '../cell-aggregation/cell-table';
import {
  getFixedPointWGSL,
  GPU_CELL_DEFAULT_SUM_SCALE,
  ORDERED_KEY_WGSL,
  validateSumScale
} from '../../analysis-core/cell-key-wgsl';

const OPERATION = 'GPUCellTableCompare';
/** Two-word row addressing and the sum of both input cardinalities must fit uint32. */
const MAXIMUM_CELL_TABLE_ROW_COUNT = 0x7fffffff;
const STATISTICS_WORKGROUP_SIZE = 256;
/** Upper bound of the workgroups sharing the standardized z-score reductions. */
const STATISTICS_MAXIMUM_GROUPS = 256;

/** `presence` bit set when the cell occurs in the `before` table. */
export const GPU_CELL_COMPARE_PRESENT_BEFORE = 1;

/** `presence` bit set when the cell occurs in the `after` table. */
export const GPU_CELL_COMPARE_PRESENT_AFTER = 2;

/** Cell table column that {@link GPUCellTableCompare} compares. */
export type GPUCellTableCompareMeasure = 'count' | 'sum';

/**
 * Score written to `output.zScore`: `'poisson'` is `delta / sqrt(before + after)` (0 when both
 * are 0), `'standardized'` is `(delta - mean) / sd` over the union rows.
 *
 * `'poisson'` is a statistic of counts. With `measure: 'sum'` it is meaningful only for sums of
 * non-negative values: a row where `before + after` is negative, or is 0 while the two sums
 * differ, gets NaN.
 */
export type GPUCellTableCompareZScore = 'poisson' | 'standardized';

/** Caller-owned, capacity-bounded outputs of {@link GPUCellTableCompare}. */
export type GPUCellTableCompareOutput = {
  /** Union cell keys ascending as little-endian `(low, high)` words; capacity is its length. */
  cells: GraphDataView<'uint32x2'>;
  /** Per cell: bit 1 (`GPU_CELL_COMPARE_PRESENT_BEFORE`) in `before`, bit 2 in `after`. */
  presence?: GraphDataView<'uint32'>;
  /** Per cell: 1 for a union row below `count`, 0 for the empty tail. Same length as `cells`. */
  mask?: GraphDataView<'uint32'>;
  /** Measure in the `before` table, 0 when absent. */
  before?: GraphDataView<'float32'>;
  /** Measure in the `after` table, 0 when absent. */
  after?: GraphDataView<'float32'>;
  /**
   * `after - before`, computed exactly in integers (64-bit fixed point for sums) and then rounded
   * to f32.
   */
  delta?: GraphDataView<'float32'>;
  /** `after / before`, NaN when `before` is 0. */
  ratio?: GraphDataView<'float32'>;
  /** `100 * (after - before) / before`, NaN when `before` is 0. */
  percentChange?: GraphDataView<'float32'>;
  /** The configured z-score, see {@link GPUCellTableCompareZScore}. */
  zScore?: GraphDataView<'float32'>;
  /**
   * One-row status scalars: `count` receives `min(requiredCount, cells.length)`, `overflow`
   * receives 1 when union cells were dropped or either input table overflowed, and the optional
   * `requiredCount` receives the unclamped number of union cells.
   */
  status: GPUCardinalityPort & GPUOverflowStatusPort;
};

/**
 * Properties for {@link GPUCellTableCompare}.
 *
 * Per-frame (no recompile): the contents of both tables. Topology: table capacities, which
 * columns are present, `measure`, `sumScale`, `zScore`, and the output capacity.
 */
export type GPUCellTableCompareProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'cell-table-compare'`. */
  id?: string;
  /**
   * First table, written by {@link GPUCellAggregation} or {@link GPUCellRollup}: ascending keys in
   * rows `[0, count)`. Both tables must use the same grid family and resolution; the contributor
   * cannot check that.
   */
  before: GPUCellTable;
  /** Second table, same family and resolution as `before`. */
  after: GPUCellTable;
  /** Compared column. Defaults to `'count'`; `'sum'` needs `sums` on both tables. */
  measure?: GPUCellTableCompareMeasure;
  /** Fixed-point scale of both tables' `sums` (for `'sum'`). Defaults to 65536. */
  sumScale?: number;
  /** Defaults to `'poisson'` for counts and `'standardized'` for sums. */
  zScore?: GPUCellTableCompareZScore;
  /** Caller-owned union table. */
  output: GPUCellTableCompareOutput;
};

/**
 * Outer-join diff of two sorted cell tables (period-over-period or A-versus-B compare maps).
 *
 * Both tables are sorted ascending with distinct keys, so the union order follows from binary
 * searches and one scan of matched `before` rows: a `before` row `i` lands at
 * `i + lowerBound(after, key) - matchesBelow(i)` and an `after` row `j` whose key is absent from
 * `before` at `j + lowerBound(before, key) - matchesBelow(lowerBound(before, key))`, where
 * `matchesBelow(k)` counts `before` rows below `k` that also occur in `after`. Equal keys are
 * emitted once, by the `before` side, which also reads the matching `after` row. Output rows are
 * written by exactly one invocation each, so results do not depend on thread order. `delta` is an
 * exact integer difference (64-bit fixed point for sums) before it is rounded to f32. The
 * standardized z-score mean and standard deviation come from one workgroup in a fixed order (strided partial
 * sums then a binary tree), so a second encoding is bitwise identical.
 *
 * The union keeps the smallest keys when it exceeds the output capacity, and the overflow flag
 * includes the input tables' own overflow flags. Rows past `count` hold the empty key, presence 0,
 * `before`/`after` 0 and NaN for derived columns. Keys that are the reserved all-ones key inside
 * `[0, count)` are not supported. Inputs must be single packed views.
 *
 * Consumes cell keys and table values only, so it takes no `spatialContext`.
 */
export class GPUCellTableCompare implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCellTableCompareProps;
  /** Compared column. */
  readonly measure: GPUCellTableCompareMeasure;
  /** Fixed-point scale of the sums. */
  readonly sumScale: number;
  /** Resolved z-score kind. */
  readonly zScoreKind: GPUCellTableCompareZScore;
  private readonly stageMeasures: boolean;

  constructor(props: GPUCellTableCompareProps) {
    this.id = props.id ?? 'cell-table-compare';
    this.props = props;
    const id = this.id;
    this.measure = props.measure ?? 'count';
    if (this.measure !== 'count' && this.measure !== 'sum') {
      throw new Error(`${id} measure must be 'count' or 'sum'`);
    }
    this.sumScale = props.sumScale ?? GPU_CELL_DEFAULT_SUM_SCALE;
    validateSumScale(id, this.sumScale);
    this.zScoreKind = props.zScore ?? (this.measure === 'sum' ? 'standardized' : 'poisson');
    if (this.zScoreKind !== 'poisson' && this.zScoreKind !== 'standardized') {
      throw new Error(`${id} zScore must be 'poisson' or 'standardized'`);
    }
    validateCellTable(id, 'before', props.before);
    validateCellTable(id, 'after', props.after);
    validateCapacity(
      id,
      'before.cells.length',
      props.before.cells.length,
      1,
      MAXIMUM_CELL_TABLE_ROW_COUNT
    );
    validateCapacity(
      id,
      'after.cells.length',
      props.after.cells.length,
      1,
      MAXIMUM_CELL_TABLE_ROW_COUNT
    );
    if (this.measure === 'sum' && (!props.before.sums || !props.after.sums)) {
      throw new Error(`${id} measure 'sum' needs sums on both before and after`);
    }
    const {output} = props;
    validateSinglePackedView(output.cells, ['uint32x2'], `${id} output.cells`);
    const capacity = output.cells.length;
    validateCapacity(id, 'output.cells.length', capacity, 1, MAXIMUM_CELL_TABLE_ROW_COUNT);
    if (output.presence) {
      validateSinglePackedView(output.presence, ['uint32'], `${id} output.presence`);
    }
    if (output.mask) {
      validateSinglePackedView(output.mask, ['uint32'], `${id} output.mask`);
    }
    for (const name of ['before', 'after', 'delta', 'ratio', 'percentChange', 'zScore'] as const) {
      const view = output[name];
      if (view) {
        validateSinglePackedView(view, ['float32'], `${id} output.${name}`);
      }
    }
    for (const name of [
      'presence',
      'mask',
      'before',
      'after',
      'delta',
      'ratio',
      'percentChange',
      'zScore'
    ] as const) {
      const view = output[name];
      if (view && view.length !== capacity) {
        throw new Error(`${id} output.${name} needs output.cells.length rows`);
      }
    }
    for (const name of ['count', 'overflow', 'requiredCount'] as const) {
      const view = output.status[name];
      if (!view) {
        continue;
      }
      validateSinglePackedView(view, ['uint32'], `${id} output.status.${name}`);
      if (view.length < 1) {
        throw new Error(`${id} output.status.${name} needs one row`);
      }
    }
    const {status, ...columns} = output;
    const measureBuffers = new Set(
      [output.before, output.after, output.delta].flatMap(view =>
        view ? [getCellDispatchBuffer(view)] : []
      )
    );
    this.stageMeasures = [output.ratio, output.percentChange, output.zScore].some(
      view => view && measureBuffers.has(getCellDispatchBuffer(view))
    );
    validateCellDispatchOutputs(id, {...columns, ...status}, [
      {cells: output.cells, presence: output.presence, mask: output.mask},
      this.stageMeasures
        ? {}
        : {
            before: output.before,
            after: output.after,
            delta: output.delta
          },
      {
        ratio: output.ratio,
        percentChange: output.percentChange,
        zScore: output.zScore
      },
      {...status}
    ]);
    validateGraphOutputsDisjointFromInputs(id, getOutputViews(output), [
      ...getCellTableViews(props.before),
      ...getCellTableViews(props.after)
    ]);
  }

  /**
   * Returns match, scan, merge, measure, statistics, derived-column, tail, and publish nodes in
   * order. A standardized z-score adds three statistics nodes (`-statistics-sums`,
   * `-statistics-squares`, `-statistics`) that reduce in fixed order across workgroups.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, measure} = this;
    const {before, after, output} = props;
    const views: Record<string, GraphDataView | undefined> = {};
    for (const [name, table] of [
      ['before', before],
      ['after', after]
    ] as const) {
      views[`${name}.cells`] = table.cells;
      views[`${name}.measure`] = measure === 'sum' ? table.sums : table.counts;
      views[`${name}.status.count`] = table.status.count;
      views[`${name}.status.overflow`] = table.status.overflow;
    }
    for (const [name, view] of Object.entries(getOutputViews(output)))
      views[`output ${name}`] = view;
    validateGraphChunkBindingRanges(graph, id, views);
    // The bounds plane includes the final lower bound and two clamped input counts.
    const scratch = [
      ['match bounds scratch', (before.cells.length + 3) * Uint32Array.BYTES_PER_ELEMENT],
      ['output columns scratch', output.cells.length * Float32Array.BYTES_PER_ELEMENT]
    ] as const;
    for (const [name, byteLength] of scratch) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        byteLength,
        'reduce table capacities or use a device with larger bindings'
      );
      validateCapacity(id, `${name} byteLength`, byteLength, 0, graph.device.limits.maxBufferSize);
    }
    if (output.zScore && this.zScoreKind === 'standardized') {
      const groups = clamp(
        Math.ceil(output.cells.length / STATISTICS_WORKGROUP_SIZE),
        1,
        STATISTICS_MAXIMUM_GROUPS
      );
      const byteLength = 2 * groups * Float32Array.BYTES_PER_ELEMENT;
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'statistics partials scratch',
        byteLength,
        'reduce output.cells.length or use a device with larger bindings'
      );
      validateCapacity(
        id,
        'statistics partials scratch byteLength',
        byteLength,
        0,
        graph.device.limits.maxBufferSize
      );
    }
    validateGraphViewsBelongToGraph(id, graph, [
      ...getCellTableViews(before),
      ...getCellTableViews(after),
      ...getOutputViews(output)
    ]);
    const capacity = output.cells.length;
    const capacityBefore = before.cells.length;
    const capacityAfter = after.cells.length;
    const transient = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);
    const read = (name: string, view: GraphDataView, type: 'u32' | 'f32' = 'u32') =>
      ({name, view, type, access: 'read'}) as WGSLKernelBinding;
    const write = (name: string, view: GraphDataView, type: 'u32' | 'f32' = 'u32') =>
      ({name, view, type, access: 'read_write'}) as WGSLKernelBinding;
    const nodes: GPUCommandNode<Parameters>[] = [];

    const total = transient('total', 'uint32', 1);
    const beforeRows = transient('before-rows', 'uint32', capacity);
    const afterRows = transient('after-rows', 'uint32', capacity);
    // Derived columns must not bind a caller buffer both read-only and writable, even at
    // disjoint offsets. Stage only this packed-layout case; ordinary layouts keep their route.
    const beforeValues =
      (!this.stageMeasures && output.before) || transient('before-values', 'float32', capacity);
    const afterValues =
      (!this.stageMeasures && output.after) || transient('after-values', 'float32', capacity);
    const deltaValues =
      (!this.stageMeasures && output.delta) || transient('delta-values', 'float32', capacity);

    // 1. Matches: before rows whose key also occurs in `after`, an exclusive scan of them, and
    // the clamped row counts. The scan has one extra row, so `prefix[capacityBefore]` is the total.
    const flags = transient('matched-flags', 'uint32', capacityBefore + 1);
    const prefix = transient('matched-prefix', 'uint32', capacityBefore + 1);
    // `bounds[i]` is the lower bound of before row `i` in `after`, found once here and reused by the
    // merge pass. The last two rows hold the clamped row counts of `before` and `after`.
    const bounds = transient('after-bounds', 'uint32', capacityBefore + 3);
    const keyDeclarations = `const CAPACITY: u32 = ${capacity}u;
const CAPACITY_BEFORE: u32 = ${capacityBefore}u;
const CAPACITY_AFTER: u32 = ${capacityAfter}u;
const NO_ROW: u32 = 0xffffffffu;
fn getBeforeKey(row: u32) -> vec2u {
  return vec2u(beforeCells[beforeCellsOffset + 2u * row + 1u], beforeCells[beforeCellsOffset + 2u * row]);
}
fn getAfterKey(row: u32) -> vec2u {
  return vec2u(afterCells[afterCellsOffset + 2u * row + 1u], afterCells[afterCellsOffset + 2u * row]);
}
fn isKeyLess(left: vec2u, right: vec2u) -> bool {
  return left.x < right.x || (left.x == right.x && left.y < right.y);
}`;
    const searchFunction = (
      name: string,
      table: string
    ) => `fn ${name}(key: vec2u, count: u32) -> u32 {
  var low = 0u;
  var high = count;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (isKeyLess(get${table}Key(middle), key)) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-flags`,
        operation: OPERATION,
        variant: 'matches',
        bindings: [
          read('beforeCells', before.cells),
          read('beforeCount', before.status.count),
          read('afterCells', after.cells),
          read('afterCount', after.status.count),
          write('flags', flags),
          write('bounds', bounds)
        ],
        invocationCount: capacityBefore + 1,
        // In-kernel lower bound over two-word (high, low) keys; upstream GPUGallopingSearch is a batch node (extra pass and buffer) and takes only uint32/float32 keys.
        declarations: `${keyDeclarations}
${searchFunction('lowerBoundAfter', 'After')}`,
        body: `let countBefore = min(beforeCount[beforeCountOffset], CAPACITY_BEFORE);
  let countAfter = min(afterCount[afterCountOffset], CAPACITY_AFTER);
  if (index == 0u) {
    bounds[boundsOffset + CAPACITY_BEFORE + 1u] = countBefore;
    bounds[boundsOffset + CAPACITY_BEFORE + 2u] = countAfter;
  }
  var isMatch = false;
  if (index < countBefore) {
    let key = getBeforeKey(index);
    let bound = lowerBoundAfter(key, countAfter);
    bounds[boundsOffset + index] = bound;
    isMatch = bound < countAfter && all(getAfterKey(bound) == key);
  }
  flags[flagsOffset + index] = select(0u, 1u, isMatch);`
      }),
      ...new GPUScan({
        id: `${id}-scan`,
        input: flags,
        output: prefix,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );

    // 2. Merge path. A `before` row lands after the `before` rows below it and the `after` rows
    // below it, minus the matched pairs below it (counted on both sides). An unmatched `after`
    // row likewise subtracts the matched pairs below its key.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-merge`,
        operation: OPERATION,
        variant: 'merge-path',
        bindings: [
          read('beforeCells', before.cells),
          read('afterCells', after.cells),
          read('bounds', bounds),
          read('prefix', prefix),
          write('cellsOut', output.cells),
          write('beforeRows', beforeRows),
          write('afterRows', afterRows),
          write('total', total)
        ],
        invocationCount: capacityBefore + capacityAfter,
        // In-kernel lower bound over two-word (high, low) keys; upstream GPUGallopingSearch is a batch node (extra pass and buffer) and takes only uint32/float32 keys.
        declarations: `${keyDeclarations}
${searchFunction('lowerBoundBefore', 'Before')}`,
        body: `let countBefore = bounds[boundsOffset + CAPACITY_BEFORE + 1u];
  let countAfter = bounds[boundsOffset + CAPACITY_BEFORE + 2u];
  if (index == 0u) {
    total[totalOffset] = countBefore + countAfter - prefix[prefixOffset + CAPACITY_BEFORE];
  }
  var key = vec2u(0u);
  var position = 0u;
  var beforeRow = NO_ROW;
  var afterRow = NO_ROW;
  if (index < CAPACITY_BEFORE) {
    if (index >= countBefore) {
      return;
    }
    key = getBeforeKey(index);
    let bound = bounds[boundsOffset + index];
    position = index + bound - prefix[prefixOffset + index];
    beforeRow = index;
    if (bound < countAfter && all(getAfterKey(bound) == key)) {
      afterRow = bound;
    }
  } else {
    let row = index - CAPACITY_BEFORE;
    if (row >= countAfter) {
      return;
    }
    key = getAfterKey(row);
    let bound = lowerBoundBefore(key, countBefore);
    if (bound < countBefore && all(getBeforeKey(bound) == key)) {
      return;
    }
    position = row + bound - prefix[prefixOffset + bound];
    afterRow = row;
  }
  if (position < CAPACITY) {
    cellsOut[cellsOutOffset + 2u * position] = key.y;
    cellsOut[cellsOutOffset + 2u * position + 1u] = key.x;
    beforeRows[beforeRowsOffset + position] = beforeRow;
    afterRows[afterRowsOffset + position] = afterRow;
  }`
      })
    );

    // 3. Measures and the exact delta.
    const isSum = measure === 'sum';
    const beforeMeasure = isSum ? before.sums! : before.counts;
    const afterMeasure = isSum ? after.sums! : after.counts;
    const measureAt = (table: string, row: string) =>
      isSum
        ? `vec2u(${table}Measure[${table}MeasureOffset + 2u * ${row} + 1u], ${table}Measure[${table}MeasureOffset + 2u * ${row}])`
        : `vec2u(0u, ${table}Measure[${table}MeasureOffset + ${row}])`;
    const toFloat = (value: string) =>
      isSum
        ? `getScaledQuotient(cellI64ToF32(${value}), SUM_SCALE, 1.0, 0)`
        : `cellI64ToF32(${value})`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-measures`,
        operation: OPERATION,
        variant: isSum ? 'sum-measures' : 'count-measures',
        bindings: [
          read('beforeRows', beforeRows),
          read('afterRows', afterRows),
          read('beforeMeasure', beforeMeasure),
          read('afterMeasure', afterMeasure),
          read('total', total),
          write('beforeOut', beforeValues, 'f32'),
          write('afterOut', afterValues, 'f32'),
          write('deltaOut', deltaValues, 'f32')
        ],
        invocationCount: capacity,
        declarations: `const CAPACITY: u32 = ${capacity}u;
const NO_ROW: u32 = 0xffffffffu;
${ORDERED_KEY_WGSL}
${getFixedPointWGSL(this.sumScale)}
${SCALED_QUOTIENT_WGSL}`,
        body: `if (index >= min(total[totalOffset], CAPACITY)) {
    beforeOut[beforeOutOffset + index] = 0.0;
    afterOut[afterOutOffset + index] = 0.0;
    deltaOut[deltaOutOffset + index] = getNaN();
    return;
  }
  let beforeRow = beforeRows[beforeRowsOffset + index];
  let afterRow = afterRows[afterRowsOffset + index];
  var beforeValue = vec2u(0u);
  var afterValue = vec2u(0u);
  if (beforeRow != NO_ROW) {
    beforeValue = ${measureAt('before', 'beforeRow')};
  }
  if (afterRow != NO_ROW) {
    afterValue = ${measureAt('after', 'afterRow')};
  }
  let difference = cellAddI64(afterValue, cellNegateI64(beforeValue));
  beforeOut[beforeOutOffset + index] = ${toFloat('beforeValue')};
  afterOut[afterOutOffset + index] = ${toFloat('afterValue')};
  deltaOut[deltaOutOffset + index] = ${toFloat('difference')};`
      })
    );

    // 4. Standardized z-score statistics in three fixed-order stages: per-workgroup partial sums,
    // per-workgroup partial squares around a mean every workgroup derives identically from the
    // combined partial sums, then one workgroup combining both into mean and population deviation.
    const wantsStandardized = Boolean(output.zScore) && this.zScoreKind === 'standardized';
    const statistics = wantsStandardized ? transient('statistics', 'float32', 2) : undefined;
    if (statistics) {
      const groups = clamp(
        Math.ceil(capacity / STATISTICS_WORKGROUP_SIZE),
        1,
        STATISTICS_MAXIMUM_GROUPS
      );
      // Words [0, groups) are partial sums and [groups, 2 * groups) partial squares.
      const partialsView = transient('statistics-partials', 'float32', 2 * groups);
      const tree = (name: string) => `partials[localInvocationIndex] = ${name};
  workgroupBarrier();
  for (var stride = ${STATISTICS_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partials[localInvocationIndex] += partials[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }`;
      // Lanes load the partials of workgroups `0..GROUPS` at `base` (zero past GROUPS), then tree-sum.
      const combine = (base: string) => `partials[localInvocationIndex] = 0.0;
  if (localInvocationIndex < GROUPS) {
    partials[localInvocationIndex] = groupPartials[groupPartialsOffset + ${base} + localInvocationIndex];
  }
  workgroupBarrier();
  for (var stride = ${STATISTICS_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partials[localInvocationIndex] += partials[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }`;
      const declarations = `const CAPACITY: u32 = ${capacity}u;
const GROUPS: u32 = ${groups}u;
const STRIDE: u32 = ${groups * STATISTICS_WORKGROUP_SIZE}u;
var<workgroup> partials: array<f32, ${STATISTICS_WORKGROUP_SIZE}>;`;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-statistics-sums`,
          operation: OPERATION,
          variant: 'statistics-sums',
          bindings: [
            read('deltas', deltaValues, 'f32'),
            read('total', total),
            write('groupPartials', partialsView, 'f32')
          ],
          workgroupSize: STATISTICS_WORKGROUP_SIZE,
          invocationCount: groups * STATISTICS_WORKGROUP_SIZE,
          guardIndex: false,
          declarations,
          body: `let rowCount = min(total[totalOffset], CAPACITY);
  var partialSum = 0.0;
  for (var row = index; row < rowCount; row += STRIDE) {
    partialSum += deltas[deltasOffset + row];
  }
  ${tree('partialSum')}
  if (localInvocationIndex == 0u) {
    groupPartials[groupPartialsOffset + index / ${STATISTICS_WORKGROUP_SIZE}u] = partials[0];
  }`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-statistics-squares`,
          operation: OPERATION,
          variant: 'statistics-squares',
          bindings: [
            read('deltas', deltaValues, 'f32'),
            read('total', total),
            write('groupPartials', partialsView, 'f32')
          ],
          workgroupSize: STATISTICS_WORKGROUP_SIZE,
          invocationCount: groups * STATISTICS_WORKGROUP_SIZE,
          guardIndex: false,
          // Words [0, GROUPS) are only read and [GROUPS, 2 * GROUPS) only written.
          declarations,
          body: `let rowCount = min(total[totalOffset], CAPACITY);
  ${combine('0u')}
  let mean = select(0.0, partials[0] / f32(rowCount), rowCount > 0u);
  workgroupBarrier();
  var partialSquares = 0.0;
  for (var row = index; row < rowCount; row += STRIDE) {
    let centered = deltas[deltasOffset + row] - mean;
    partialSquares += centered * centered;
  }
  ${tree('partialSquares')}
  if (localInvocationIndex == 0u) {
    groupPartials[groupPartialsOffset + GROUPS + index / ${STATISTICS_WORKGROUP_SIZE}u] = partials[0];
  }`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-statistics`,
          operation: OPERATION,
          variant: 'statistics',
          bindings: [
            read('groupPartials', partialsView, 'f32'),
            read('total', total),
            write('statistics', statistics, 'f32')
          ],
          workgroupSize: STATISTICS_WORKGROUP_SIZE,
          invocationCount: STATISTICS_WORKGROUP_SIZE,
          guardIndex: false,
          declarations,
          body: `let rowCount = min(total[totalOffset], CAPACITY);
  ${combine('0u')}
  let mean = select(0.0, partials[0] / f32(rowCount), rowCount > 0u);
  workgroupBarrier();
  ${combine('GROUPS')}
  if (localInvocationIndex == 0u) {
    statistics[statisticsOffset] = mean;
    statistics[statisticsOffset + 1u] = select(0.0, sqrt(partials[0] / f32(rowCount)), rowCount > 0u);
  }`
        })
      );
    }

    // 5. Ratio, percent change and z-score.
    if (output.ratio || output.percentChange || output.zScore) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-derived`,
          operation: OPERATION,
          variant: 'derived',
          bindings: [
            read('beforeValues', beforeValues, 'f32'),
            read('afterValues', afterValues, 'f32'),
            read('deltaValues', deltaValues, 'f32'),
            read('total', total),
            ...(output.ratio ? [write('ratioOut', output.ratio, 'f32')] : []),
            ...(output.percentChange ? [write('percentOut', output.percentChange, 'f32')] : []),
            ...(output.zScore ? [write('zScoreOut', output.zScore, 'f32')] : []),
            ...(statistics ? [read('statistics', statistics, 'f32')] : [])
          ],
          invocationCount: capacity,
          declarations: `const CAPACITY: u32 = ${capacity}u;
${ORDERED_KEY_WGSL}
${IS_FINITE_VALUE_WGSL}
${SCALED_QUOTIENT_WGSL}
fn getCompareQuotient(value: f32, divisor: f32) -> f32 {
  if (isFiniteValue(value) && isFiniteValue(divisor) && divisor != 0.0) {
    return getScaledQuotient(value, divisor, 1.0, 0);
  }
  return value / divisor;
}
fn getComparePercent(value: f32, divisor: f32) -> f32 {
  if (isFiniteValue(value) && isFiniteValue(divisor) && divisor != 0.0) {
    let parts = frexp(value);
    return getScaledQuotient(parts.fract * 100.0, divisor, 1.0, parts.exp);
  }
  return (100.0 * value) / divisor;
}`,
          body: `let isOccupied = index < min(total[totalOffset], CAPACITY);
  let nan = getNaN();
  let beforeValue = beforeValues[beforeValuesOffset + index];
  let afterValue = afterValues[afterValuesOffset + index];
  let delta = deltaValues[deltaValuesOffset + index];
  ${output.ratio ? 'ratioOut[ratioOutOffset + index] = select(nan, getCompareQuotient(afterValue, beforeValue), isOccupied && beforeValue != 0.0);' : ''}
  ${output.percentChange ? 'percentOut[percentOutOffset + index] = select(nan, getComparePercent(delta, beforeValue), isOccupied && beforeValue != 0.0);' : ''}
  ${
    output.zScore
      ? this.zScoreKind === 'poisson'
        ? `// A count statistic: defined for before + after > 0, and 0 when both are 0. Sums can make
  // the denominator zero or negative with a non-zero delta; that has no value (never sqrt of a
  // negative number, which WGSL leaves indeterminate).
  let denominator = beforeValue + afterValue;
  let undefinedScore = select(nan, 0.0, beforeValue == 0.0 && afterValue == 0.0);
  let score = select(undefinedScore, getCompareQuotient(delta, sqrt(max(denominator, 0.0))), denominator > 0.0);
  zScoreOut[zScoreOutOffset + index] = select(nan, score, isOccupied);`
        : `let mean = statistics[statisticsOffset];
  let deviation = statistics[statisticsOffset + 1u];
  zScoreOut[zScoreOutOffset + index] = select(nan, select(0.0, getCompareQuotient(delta - mean, deviation), deviation != 0.0), isOccupied);`
      : ''
  }`
        })
      );
    }

    if (this.stageMeasures) {
      for (const [name, source, destination] of [
        ['before', beforeValues, output.before],
        ['after', afterValues, output.after],
        ['delta', deltaValues, output.delta]
      ] as const)
        if (destination)
          nodes.push(
            createWGSLKernelNode(graph, {
              id: `${id}-publish-${name}`,
              operation: OPERATION,
              variant: 'publish-measures',
              bindings: [
                read('sourceValues', source, 'f32'),
                write('destinationValues', destination, 'f32')
              ],
              invocationCount: capacity,
              body: 'destinationValues[destinationValuesOffset + index] = sourceValues[sourceValuesOffset + index];'
            })
          );
    }

    // 6. Presence, and empty keys past the union count.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-tail`,
        operation: OPERATION,
        variant: 'tail',
        bindings: [
          read('beforeRows', beforeRows),
          read('afterRows', afterRows),
          read('total', total),
          write('cellsOut', output.cells),
          ...(output.presence ? [write('presenceOut', output.presence)] : []),
          ...(output.mask ? [write('maskOut', output.mask)] : [])
        ],
        invocationCount: capacity,
        declarations: `const CAPACITY: u32 = ${capacity}u;
const NO_ROW: u32 = 0xffffffffu;`,
        body: `if (index >= min(total[totalOffset], CAPACITY)) {
    cellsOut[cellsOutOffset + 2u * index] = 0xffffffffu;
    cellsOut[cellsOutOffset + 2u * index + 1u] = 0xffffffffu;
    ${output.presence ? 'presenceOut[presenceOutOffset + index] = 0u;' : ''}
    ${output.mask ? 'maskOut[maskOutOffset + index] = 0u;' : ''}
    return;
  }
  ${output.mask ? 'maskOut[maskOutOffset + index] = 1u;' : ''}
  ${
    output.presence
      ? `presenceOut[presenceOutOffset + index] =
    select(0u, ${GPU_CELL_COMPARE_PRESENT_BEFORE}u, beforeRows[beforeRowsOffset + index] != NO_ROW) |
    select(0u, ${GPU_CELL_COMPARE_PRESENT_AFTER}u, afterRows[afterRowsOffset + index] != NO_ROW);`
      : ''
  }`
      })
    );

    // 7. Clamped count and overflow, including the input tables' own overflow.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        variant: 'publish',
        bindings: [
          {name: 'totalIn', view: total, type: 'u32', access: 'read'},
          {name: 'beforeOverflow', view: before.status.overflow, type: 'u32', access: 'read'},
          {name: 'afterOverflow', view: after.status.overflow, type: 'u32', access: 'read'},
          {name: 'countOut', view: output.status.count, type: 'u32', access: 'read_write'},
          {name: 'overflowOut', view: output.status.overflow, type: 'u32', access: 'read_write'},
          ...(output.status.requiredCount
            ? [
                {
                  name: 'totalOut',
                  view: output.status.requiredCount,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: 1,
        declarations: `const CAPACITY: u32 = ${output.cells.length}u;`,
        body: `let required = totalIn[totalInOffset];
  countOut[countOutOffset] = min(required, CAPACITY);
  overflowOut[overflowOutOffset] = select(0u, 1u, required > CAPACITY || beforeOverflow[beforeOverflowOffset] != 0u || afterOverflow[afterOverflowOffset] != 0u);
  ${output.status.requiredCount ? 'totalOut[totalOutOffset] = required;' : ''}`
      })
    );
    return nodes;
  }
}

function getOutputViews(output: GPUCellTableCompareOutput): (GraphDataView | undefined)[] {
  return [
    output.cells,
    output.presence,
    output.mask,
    output.before,
    output.after,
    output.delta,
    output.ratio,
    output.percentChange,
    output.zScore,
    output.status.count,
    output.status.overflow,
    output.status.requiredCount
  ];
}
