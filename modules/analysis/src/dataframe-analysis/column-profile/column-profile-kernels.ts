// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {COLUMN_ORDERED_KEY_WGSL} from '../column-classification/column-classification-shared';
import {
  GPU_COLUMN_PROFILE_NULL_CATEGORY,
  GPU_COLUMN_PROFILE_PARAMETERS_PER_COLUMN,
  GPU_COLUMN_PROFILE_STATISTIC,
  GPU_COLUMN_PROFILE_STATISTIC_COUNT
} from './column-profile-parameters';
import {
  validateGPUStorageBindingByteLength,
  validateMaximumCount
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';

/** Workgroup size of the reduction kernels. */
export const COLUMN_PROFILE_WORKGROUP_SIZE = 256;
/** Rows each invocation folds sequentially before the workgroup tree. */
export const COLUMN_PROFILE_ROWS_PER_INVOCATION = 8;
/** Rows reduced by one workgroup tile. */
export const COLUMN_PROFILE_TILE_SIZE =
  COLUMN_PROFILE_WORKGROUP_SIZE * COLUMN_PROFILE_ROWS_PER_INVOCATION;
/** Words per moment record (three uint32 counts followed by four float32 bit patterns): `[count, nullCount, finiteCount, mean, m2, sum, origin]`. */
const RECORD_STRIDE = 7;
/** Keys per column: `[minimumKey, maximumKey, finiteMinimumKey, finiteMaximumKey]`. */
const KEY_STRIDE = 4;
/** Floats per resolved domain: `[lo, hi, width, mode]`; mode 0 invalid, 1 regular, 2 `lo == hi`. */
const DOMAIN_STRIDE = 4;
/** A histogram domain wider than this (2^100) is scaled before the bin products. @internal */
export const COLUMN_PROFILE_LARGE_RANGE = 2 ** 100;
/** Exact scale (2^-24) of a large range and of the differences inside it. @internal */
export const COLUMN_PROFILE_LARGE_RANGE_SCALE = 2 ** -24;

/** One normalized profile column. @internal */
export type ColumnProfileColumn = {
  kind: 'numeric' | 'category';
  values: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  categoryCount: number;
  /** Offset of this column's counts in the shared category count array. */
  categoryBase: number;
};

/** Inputs of {@link getColumnProfileNodes}. @internal */
export type ColumnProfileNodeProps = {
  id: string;
  operation: string;
  columns: readonly ColumnProfileColumn[];
  rowCount: number;
  mask?: GraphDataView<'uint32'>;
  parameters?: GraphDataView<'float32'>;
  histogramBinCount: number;
  hyperLogLogPrecision: number;
  topCategoryCount: number;
  output: {
    statistics: GraphDataView<'float32'>;
    counts?: GraphDataView<'uint32'>;
    histograms?: GraphDataView<'uint32'>;
    hyperLogLogRegisters?: GraphDataView<'uint32'>;
    topCategories?: GraphDataView<'uint32'>;
    topCategoryCounts?: GraphDataView<'uint32'>;
  };
};

/**
 * Moment record merged by a fixed-order tree: counts remain uint32 through both reduction levels;
 * `(finiteCount, mean, m2)` follow the Chan, Golub and LeVeque pairwise
 * update with the mean stored relative to the first finite value (`origin`). Keeping the
 * origin separate avoids rounding small mean updates at the scale of an offset such as an
 * epoch timestamp. Only the public mean adds the origin back; M2 never does.
 * `sum` is a plain fixed-order sum. Category columns carry their overflow count in
 * `finiteCount` with `mean = m2 = sum = 0`.
 */
const MOMENTS_WGSL = /* wgsl */ `
struct Moments {
  count: u32,
  nullCount: u32,
  finiteCount: u32,
  mean: f32,
  m2: f32,
  sum: f32,
  origin: f32,
}

fn emptyMoments() -> Moments {
  return Moments(0u, 0u, 0u, 0.0, 0.0, 0.0, 0.0);
}

fn mergeMoments(left: Moments, right: Moments) -> Moments {
  var result = emptyMoments();
  result.count = left.count + right.count;
  result.nullCount = left.nullCount + right.nullCount;
  result.sum = left.sum + right.sum;
  if (right.finiteCount == 0u) {
    result.finiteCount = left.finiteCount;
    result.origin = left.origin;
    result.mean = left.mean;
    result.m2 = left.m2;
  } else if (left.finiteCount == 0u) {
    result.finiteCount = right.finiteCount;
    result.origin = right.origin;
    result.mean = right.mean;
    result.m2 = right.m2;
  } else {
    let count = left.finiteCount + right.finiteCount;
    let delta = (right.origin - left.origin) + (right.mean - left.mean);
    result.origin = left.origin;
    result.finiteCount = count;
    result.mean = left.mean + delta * (f32(right.finiteCount) / f32(count));
    result.m2 = left.m2 + right.m2 + delta * delta * (f32(left.finiteCount) * f32(right.finiteCount) / f32(count));
  }
  return result;
}

var<workgroup> sharedMoments: array<Moments, ${COLUMN_PROFILE_WORKGROUP_SIZE}>;

fn reduceSharedMoments(local: u32) {
  for (var stride = ${COLUMN_PROFILE_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride >> 1u) {
    workgroupBarrier();
    if (local < stride) {
      sharedMoments[local] = mergeMoments(sharedMoments[local], sharedMoments[local + stride]);
    }
  }
}
`;

/**
 * Equal-width bin of a value inside `[minimum, maximum]`, with the convention of the official
 * `GPUHistogram` of `@luma.gl/graph/gpu-core`: bin `floor((value - minimum) / range * BIN_COUNT)`,
 * the maximum in the last bin. The official kernel evaluates that expression with one float32
 * division, which WGSL does not require to be correctly rounded. Here the division only gives an
 * estimate; the bin is the `k` with `k * range <= (value - minimum) * BIN_COUNT < (k + 1) * range`,
 * decided by correctly rounded float32 products. A value on an edge
 * `minimum + k * range / BIN_COUNT` therefore falls in bin `k` for every bin count, the result is
 * the same on every adapter, and a CPU mirror with `Math.fround` reproduces it. (The earlier rule
 * compared with `fround(k * width)` for a width rounded twice, which drifted by a unit in the
 * last place per bin when the bin count is not a power of two.)
 *
 * A range above 2^100 is scaled by 2^-24 first (exact) so that the products stay finite. The
 * caller has already rejected values outside the domain.
 */
const BIN_WGSL = /* wgsl */ `
fn getBinIndex(value: f32, minimum: f32, maximum: f32) -> u32 {
  var difference = value - minimum;
  var range = maximum - minimum;
  if (range > ${getWGSLFloatLiteral(COLUMN_PROFILE_LARGE_RANGE)}) {
    difference = difference * ${getWGSLFloatLiteral(COLUMN_PROFILE_LARGE_RANGE_SCALE)};
    range = range * ${getWGSLFloatLiteral(COLUMN_PROFILE_LARGE_RANGE_SCALE)};
  }
  let scaled = difference * f32(BIN_COUNT);
  var bin = clamp(floor(scaled / range), 0.0, f32(BIN_COUNT - 1u));
  if (scaled < bin * range) {
    bin = bin - 1.0;
  } else if (scaled >= (bin + 1.0) * range) {
    bin = bin + 1.0;
  }
  return u32(clamp(bin, 0.0, f32(BIN_COUNT - 1u)));
}
`;

/** Murmur3 `fmix32` avalanche hash, applied to the value bits XOR a per-column seed. */
const HASH_WGSL = /* wgsl */ `
fn hashBits(bits: u32) -> u32 {
  var h = bits;
  h = h ^ (h >> 16u);
  h = h * 0x85ebca6bu;
  h = h ^ (h >> 13u);
  h = h * 0xc2b2ae35u;
  h = h ^ (h >> 16u);
  return h;
}
`;

/** Largest slot count counted in workgroup memory (4 KiB of `atomic<u32>`). */
const PRIVATIZED_SLOT_LIMIT = 1024;

/**
 * Builds the `declarations` and `body` of a one-row-one-slot counting kernel. `slotBody` is the
 * body of `fn getSlot(index: u32, column: u32) -> u32` that returns the slot or `NO_SLOT`. The
 * body starts by reading `column` from the column cursor. Small tables are counted in a
 * workgroup-private atomic table and flushed with one global atomic per non-empty slot per
 * workgroup, so global atomic traffic drops from one per row to at most one per slot per
 * workgroup (skewed columns no longer serialise on a few hot counters). Large tables keep one
 * global atomic per row.
 *
 * `maximumSlotCount` sizes the workgroup table and selects the strategy, so every column of the
 * graph shares one kernel source; `slotCount` is a WGSL `u32` expression of the runtime column's
 * slot count that bounds the clear and flush loops.
 */
function getPrivatizedCountingKernel(options: {
  maximumSlotCount: number;
  slotCount: string;
  targetName: string;
  /** WGSL expression of the first slot's index in the target, in terms of `column`. */
  targetBase: string;
  declarations: string;
  slotBody: string;
}): {declarations: string; body: string; guardIndex?: boolean; workgroupSize: number} {
  const {maximumSlotCount, slotCount, targetName, targetBase, slotBody} = options;
  const slotFunction = `const NO_SLOT: u32 = 0xffffffffu;
fn getSlot(index: u32, column: u32) -> u32 {
  ${slotBody}
}`;
  const readColumn = 'let column = columnCursor[columnCursorOffset];';
  const target = `${targetName}[${targetName}Offset + ${targetBase} + `;
  if (maximumSlotCount > PRIVATIZED_SLOT_LIMIT || maximumSlotCount < 1) {
    return {
      workgroupSize: COLUMN_PROFILE_WORKGROUP_SIZE,
      declarations: `${options.declarations}\n${slotFunction}`,
      body: `${readColumn}
  let slot = getSlot(index, column);
  if (slot != NO_SLOT) {
    atomicAdd(&${target}slot], 1u);
  }`
    };
  }
  return {
    declarations: `${options.declarations}
${slotFunction}
var<workgroup> localCounts: array<atomic<u32>, ${maximumSlotCount}>;`,
    // No early return: every invocation of a workgroup must reach the barriers.
    body: `${readColumn}
  let slotCount = ${slotCount};
  for (var slot = localInvocationIndex; slot < slotCount; slot += ${COLUMN_PROFILE_WORKGROUP_SIZE}u) {
    atomicStore(&localCounts[slot], 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    let slot = getSlot(index, column);
    if (slot != NO_SLOT) {
      atomicAdd(&localCounts[slot], 1u);
    }
  }
  workgroupBarrier();
  for (var slot = localInvocationIndex; slot < slotCount; slot += ${COLUMN_PROFILE_WORKGROUP_SIZE}u) {
    let count = atomicLoad(&localCounts[slot]);
    if (count != 0u) {
      atomicAdd(&${target}slot], count);
    }
  }`,
    guardIndex: false,
    workgroupSize: COLUMN_PROFILE_WORKGROUP_SIZE
  };
}

/** Per-column HyperLogLog hash seed, `(column + 1) * 0x9e3779b9` modulo 2^32. @internal */
export function getColumnProfileHashSeed(column: number): number {
  return Math.imul(column + 1, 0x9e3779b9) >>> 0;
}

/** HyperLogLog bias constant alpha_m. @internal */
export function getHyperLogLogAlpha(registerCount: number): number {
  if (registerCount === 16) {
    return 0.673;
  }
  if (registerCount === 32) {
    return 0.697;
  }
  if (registerCount === 64) {
    return 0.709;
  }
  return 0.7213 / (1 + 1.079 / registerCount);
}

/** Number of u32 count slots a profile needs for its category columns. @internal */
export function getColumnProfileCategoryTotal(columns: readonly ColumnProfileColumn[]): number {
  return columns.reduce(
    (total, column) => total + (column.kind === 'category' ? column.categoryCount : 0),
    0
  );
}

const read = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
  name,
  view,
  type,
  access: 'read'
});
const write = (
  name: string,
  view: GraphDataView,
  type: 'u32' | 'f32' | 'atomic<u32>' = 'u32'
): WGSLKernelBinding => ({name, view, type, access: 'read_write'});

/**
 * Module-scope constants shared by every per-column kernel: the column count and the per-column
 * category metadata. They depend only on the column set, so all columns of a graph compile the
 * same kernel source and read their own entry through the runtime column cursor.
 */
function getColumnConstantsWGSL(columns: readonly ColumnProfileColumn[]): string {
  const list = (values: number[]) =>
    `array<u32, ${columns.length}>(${values.map(value => `${value}u`).join(', ')})`;
  // Private variables, not const arrays: a runtime index into a module-scope const array is
  // valid WGSL but has been a driver-compiler weak spot; a private copy is always indexable.
  return `const COLUMN_COUNT: u32 = ${columns.length}u;
var<private> columnCategoryCounts: array<u32, ${columns.length}> = ${list(columns.map(column => column.categoryCount))};
var<private> columnCategoryBases: array<u32, ${columns.length}> = ${list(columns.map(column => column.categoryBase))};`;
}

/** Reads the column that the current per-column kernel processes. */
const READ_COLUMN_WGSL = 'let column = columnCursor[columnCursorOffset];';

/** Preflights caller views and every scratch binding before declaring resources. @internal */
function validateColumnProfileBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: ColumnProfileNodeProps
): void {
  const {id, columns, output} = props;
  validateGraphChunkBindingRanges(graph, id, {
    ...Object.fromEntries(
      columns.map((column, index) => [`columns[${index}].values`, column.values])
    ),
    mask: props.mask,
    parameters: props.parameters,
    ...output
  });
  const columnCount = columns.length;
  const tileCount = Math.ceil(props.rowCount / COLUMN_PROFILE_TILE_SIZE);
  const scratchLengths = {
    partials: columnCount * tileCount * RECORD_STRIDE,
    records: columnCount * RECORD_STRIDE,
    keys: columnCount * KEY_STRIDE,
    domains: columnCount * DOMAIN_STRIDE,
    registers: output.hyperLogLogRegisters ? 0 : columnCount * 2 ** props.hyperLogLogPrecision,
    categoryCounts:
      output.histograms || output.topCategories ? getColumnProfileCategoryTotal(columns) : 0,
    columnCursor: 1
  };
  for (const [name, length] of Object.entries(scratchLengths)) {
    if (length > 0) {
      validateMaximumCount(
        id,
        `${name} scratch byteLength`,
        length * 4,
        graph.device.limits.maxBufferSize
      );
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        `${name} scratch`,
        length * Uint32Array.BYTES_PER_ELEMENT,
        'reduce the column row count or category count'
      );
    }
  }
}

/**
 * Returns every command node of a column profile, in execution order.
 *
 * The per-column kernels do not bake their column index: a one-word GPU cursor names the current
 * column, and one-thread advance steps (or the finish kernel) move it, so pipelines do not grow
 * with the column count.
 *
 * @internal
 */
export function getColumnProfileNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: ColumnProfileNodeProps
): GPUCommandNode<Parameters>[] {
  validateColumnProfileBindings(graph, props);
  const {id, operation, columns, rowCount, mask, parameters, output} = props;
  const {histogramBinCount, hyperLogLogPrecision, topCategoryCount} = props;
  const columnCount = columns.length;
  const registerCount = 2 ** hyperLogLogPrecision;
  const tileCount = Math.ceil(rowCount / COLUMN_PROFILE_TILE_SIZE);
  const categoryTotal = getColumnProfileCategoryTotal(columns);
  const hasHistograms = Boolean(output.histograms);
  const hasTopCategories = Boolean(output.topCategories && output.topCategoryCounts);
  const hasCategoryColumn = columns.some(column => column.kind === 'category');
  const needsCategoryCounts = hasCategoryColumn && (hasHistograms || hasTopCategories);
  const maskBinding = mask ? [read('rowMask', mask, 'u32')] : [];
  const valuesType = (column: ColumnProfileColumn) => (column.kind === 'numeric' ? 'f32' : 'u32');

  const u32 = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', length);
  const f32 = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'float32', length);
  const partials = u32('partials', columnCount * tileCount * RECORD_STRIDE);
  const records = u32('records', columnCount * RECORD_STRIDE);
  const keys = u32('keys', columnCount * KEY_STRIDE);
  const domains = f32('domains', columnCount * DOMAIN_STRIDE);
  const registers = output.hyperLogLogRegisters ?? u32('registers', columnCount * registerCount);
  const categoryCounts = needsCategoryCounts ? u32('category-counts', categoryTotal) : undefined;
  const columnCursor = u32('column-cursor', 1);
  const columnConstants = getColumnConstantsWGSL(columns);
  const cursorRead = read('columnCursor', columnCursor, 'u32');
  const nodes: GPUCommandNode<Parameters>[] = [];

  // Init: every accumulator is rewritten on every encoding.
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-init`,
      operation,
      variant: 'init',
      bindings: [write('keys', keys), write('columnCursor', columnCursor)],
      invocationCount: columnCount * KEY_STRIDE,
      body: `keys[keysOffset + index] = select(0xffffffffu, 0u, (index & 1u) == 1u);
  if (index == 0u) {
    columnCursor[columnCursorOffset] = 0u;
  }`
    })
  );
  const fill = (name: string, view: GraphDataView, length: number, value: string) =>
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-${name}`,
        operation,
        view,
        type: 'u32',
        value,
        componentCount: length
      })
    );
  fill('clear-registers', registers, columnCount * registerCount, '0u');
  if (output.histograms) {
    fill('clear-histograms', output.histograms, columnCount * histogramBinCount, '0u');
  }
  if (categoryCounts) {
    fill('clear-category-counts', categoryCounts, categoryTotal, '0u');
  }
  if (hasTopCategories) {
    fill(
      'clear-top-categories',
      output.topCategories!,
      columnCount * topCategoryCount,
      '0xffffffffu'
    );
    fill(
      'clear-top-category-counts',
      output.topCategoryCounts!,
      columnCount * topCategoryCount,
      '0u'
    );
  }

  // Moments tiles, one node per column.
  for (const [columnIndex, column] of columns.entries()) {
    nodes.push(
      getMomentTileNode(graph, {
        id: `${id}-column-${columnIndex}-moments-tile`,
        operation,
        column,
        rowCount,
        tileCount,
        maskBinding,
        partials,
        keys,
        columnCursor,
        columnConstants
      })
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-column-${columnIndex}-advance`,
        operation,
        variant: 'column-advance',
        bindings: [write('columnCursor', columnCursor)],
        invocationCount: 1,
        declarations: `const COLUMN_COUNT: u32 = ${columnCount}u;`,
        body: `columnCursor[columnCursorOffset] = (columnCursor[columnCursorOffset] + 1u) % COLUMN_COUNT;`
      })
    );
  }
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-moments-merge`,
      operation,
      variant: 'moments-merge',
      bindings: [read('partials', partials, 'u32'), write('records', records)],
      invocationCount: columnCount * COLUMN_PROFILE_WORKGROUP_SIZE,
      workgroupSize: COLUMN_PROFILE_WORKGROUP_SIZE,
      guardIndex: false,
      declarations: `const TILE_COUNT: u32 = ${tileCount}u;
const STEP_COUNT: u32 = ${Math.ceil(tileCount / COLUMN_PROFILE_WORKGROUP_SIZE)}u;
${MOMENTS_WGSL}`,
      body: `let local = localInvocationIndex;
  let column = workgroupIndex;
  var accumulated = emptyMoments();
  for (var step = 0u; step < STEP_COUNT; step = step + 1u) {
    let tile = step * ${COLUMN_PROFILE_WORKGROUP_SIZE}u + local;
    if (tile < TILE_COUNT) {
      let base = partialsOffset + (column * TILE_COUNT + tile) * ${RECORD_STRIDE}u;
      accumulated = mergeMoments(
        accumulated,
        Moments(partials[base], partials[base + 1u], partials[base + 2u], bitcast<f32>(partials[base + 3u]), bitcast<f32>(partials[base + 4u]), bitcast<f32>(partials[base + 5u]), bitcast<f32>(partials[base + 6u]))
      );
    }
  }
  sharedMoments[local] = accumulated;
  reduceSharedMoments(local);
  if (local == 0u) {
    let record = sharedMoments[0];
    let base = recordsOffset + column * ${RECORD_STRIDE}u;
    records[base] = record.count;
    records[base + 1u] = record.nullCount;
    records[base + 2u] = record.finiteCount;
    records[base + 3u] = bitcast<u32>(record.mean);
    records[base + 4u] = bitcast<u32>(record.m2);
    records[base + 5u] = bitcast<u32>(record.sum);
    records[base + 6u] = bitcast<u32>(record.origin);
  }`
    })
  );

  if (hasHistograms) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-domain`,
        operation,
        variant: 'domain',
        bindings: [
          read('keys', keys, 'u32'),
          ...(parameters ? [read('params', parameters, 'f32')] : []),
          write('domains', domains, 'f32')
        ],
        invocationCount: columnCount,
        declarations: `const INVERSE_BIN_COUNT: f32 = ${getWGSLFloatLiteral(1 / histogramBinCount)};
${COLUMN_ORDERED_KEY_WGSL}
${IS_FINITE_BITS_WGSL}`,
        body: `let column = index;
  let keyBase = keysOffset + column * ${KEY_STRIDE}u;
  var lo = getNaN();
  var hi = getNaN();
  ${
    parameters
      ? `lo = params[paramsOffset + column * ${GPU_COLUMN_PROFILE_PARAMETERS_PER_COLUMN}u];
  hi = params[paramsOffset + column * ${GPU_COLUMN_PROFILE_PARAMETERS_PER_COLUMN}u + 1u];`
      : ''
  }
  let finiteMinimumKey = keys[keyBase + 2u];
  let finiteMaximumKey = keys[keyBase + 3u];
  if (isNanBits(lo)) {
    lo = select(getNaN(), decodeOrderedKey(finiteMinimumKey), finiteMinimumKey != 0xffffffffu);
  }
  if (isNanBits(hi)) {
    hi = select(getNaN(), decodeOrderedKey(finiteMaximumKey), finiteMaximumKey != 0u);
  }
  var mode = 0.0;
  var width = 0.0;
  if (isFiniteBits(lo) && isFiniteBits(hi) && lo <= hi) {
    if (lo == hi) {
      mode = 2.0;
    } else {
      let range = hi - lo;
      width = range * INVERSE_BIN_COUNT;
      if (isFiniteBits(range) && isFiniteBits(width) && width > 0.0) {
        mode = 1.0;
      }
    }
  }
  let base = domainsOffset + column * ${DOMAIN_STRIDE}u;
  domains[base] = lo;
  domains[base + 1u] = hi;
  domains[base + 2u] = width;
  domains[base + 3u] = mode;`
      })
    );
  }

  // The finish kernel advances the cursor, so every other kernel of a column runs before it.
  const maximumCategoryCount = Math.max(
    0,
    ...columns.map(column => (column.kind === 'category' ? column.categoryCount : 0))
  );
  for (const [columnIndex, column] of columns.entries()) {
    const columnId = `${id}-column-${columnIndex}`;
    const isNumeric = column.kind === 'numeric';
    const type = valuesType(column);
    if (hasHistograms && isNumeric) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${columnId}-histogram`,
          operation,
          variant: 'histogram',
          bindings: [
            read('values', column.values, 'f32'),
            ...maskBinding,
            read('domains', domains, 'f32'),
            write('histogram', output.histograms!, 'atomic<u32>'),
            cursorRead
          ],
          invocationCount: rowCount,
          ...getPrivatizedCountingKernel({
            maximumSlotCount: histogramBinCount,
            slotCount: `${histogramBinCount}u`,
            targetName: 'histogram',
            targetBase: `column * BIN_COUNT`,
            declarations: `const BIN_COUNT: u32 = ${histogramBinCount}u;
${BIN_WGSL}`,
            slotBody: `${mask ? 'if (rowMask[rowMaskOffset + index] == 0u) {\n    return NO_SLOT;\n  }' : ''}
  let value = values[valuesOffset + index];
  if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) {
    return NO_SLOT;
  }
  let base = domainsOffset + column * ${DOMAIN_STRIDE}u;
  let mode = domains[base + 3u];
  let lo = domains[base];
  let hi = domains[base + 1u];
  if (mode == 0.0 || value < lo || value > hi) {
    return NO_SLOT;
  }
  var bin = 0u;
  if (mode == 1.0) {
    bin = getBinIndex(value, lo, hi);
  }
  return bin;`
          })
        })
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${columnId}-hyperloglog`,
        operation,
        variant: 'hyperloglog',
        bindings: [
          read('values', column.values, type),
          ...maskBinding,
          write('registers', registers, 'atomic<u32>'),
          cursorRead
        ],
        invocationCount: rowCount,
        declarations: `const PRECISION: u32 = ${hyperLogLogPrecision}u;
const REGISTER_COUNT: u32 = ${registerCount}u;
${HASH_WGSL}`,
        body: `${READ_COLUMN_WGSL}
  // Equals getColumnProfileHashSeed(column).
  let seed = (column + 1u) * 0x9e3779b9u;
  ${mask ? 'if (rowMask[rowMaskOffset + index] == 0u) {\n    return;\n  }' : ''}
  ${
    isNumeric
      ? `var bits = bitcast<u32>(values[valuesOffset + index]);
  if ((bits & 0x7fffffffu) > 0x7f800000u) {
    return;
  }
  if ((bits & 0x7fffffffu) == 0u) {
    bits = 0u;
  }`
      : `let bits = values[valuesOffset + index];
  if (bits == ${GPU_COLUMN_PROFILE_NULL_CATEGORY}u) {
    return;
  }`
  }
  let hash = hashBits(bits ^ seed);
  let registerIndex = hash >> (32u - PRECISION);
  let rank = min(countLeadingZeros(hash << PRECISION), 32u - PRECISION) + 1u;
  atomicMax(&registers[registersOffset + column * REGISTER_COUNT + registerIndex], rank);`
      })
    );
    if (categoryCounts && !isNumeric) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${columnId}-category-counts`,
          operation,
          variant: 'category-counts',
          bindings: [
            read('values', column.values, 'u32'),
            ...maskBinding,
            write('categoryCounts', categoryCounts, 'atomic<u32>'),
            cursorRead
          ],
          invocationCount: rowCount,
          ...getPrivatizedCountingKernel({
            maximumSlotCount: maximumCategoryCount,
            slotCount: 'columnCategoryCounts[column]',
            targetName: 'categoryCounts',
            targetBase: 'columnCategoryBases[column]',
            declarations: columnConstants,
            slotBody: `${mask ? 'if (rowMask[rowMaskOffset + index] == 0u) {\n    return NO_SLOT;\n  }' : ''}
  let code = values[valuesOffset + index];
  if (code < columnCategoryCounts[column]) {
    return code;
  }
  return NO_SLOT;`
          })
        })
      );
    }
    if (hasTopCategories && categoryCounts && !isNumeric) {
      nodes.push(
        getTopCategoriesNode(graph, {
          id: `${columnId}-top-categories`,
          operation,
          topCategoryCount,
          columnConstants,
          columnCursor,
          categoryCounts,
          topCategories: output.topCategories!,
          topCategoryCounts: output.topCategoryCounts!
        })
      );
    }
    nodes.push(
      getFinishNode(graph, {
        id: `${columnId}-finish`,
        operation,
        column,
        columnConstants,
        columnCursor,
        registerCount,
        hyperLogLogPrecision,
        histogramBinCount,
        records,
        keys,
        registers,
        categoryCounts,
        output
      })
    );
  }
  return nodes;
}

function getMomentTileNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    column: ColumnProfileColumn;
    rowCount: number;
    tileCount: number;
    maskBinding: WGSLKernelBinding[];
    partials: GraphDataView<'uint32'>;
    keys: GraphDataView<'uint32'>;
    columnCursor: GraphDataView<'uint32'>;
    columnConstants: string;
  }
): GPUCommandNode<Parameters> {
  const {column, rowCount, tileCount, maskBinding} = props;
  const isNumeric = column.kind === 'numeric';
  const hasMask = maskBinding.length > 0;
  const rowBody = isNumeric
    ? `let bits = bitcast<u32>(value);
      if ((bits & 0x7fffffffu) > 0x7f800000u) {
        accumulated.nullCount = accumulated.nullCount + 1u;
      } else {
        accumulated.count = accumulated.count + 1u;
        let key = getOrderedKey(value);
        minimumKey = min(minimumKey, key);
        maximumKey = max(maximumKey, key);
        if ((bits & 0x7fffffffu) != 0x7f800000u) {
          finiteMinimumKey = min(finiteMinimumKey, key);
          finiteMaximumKey = max(finiteMaximumKey, key);
          if (accumulated.finiteCount == 0u) {
            accumulated.origin = value;
          }
          let shiftedValue = value - accumulated.origin;
          let count = accumulated.finiteCount + 1u;
          let delta = shiftedValue - accumulated.mean;
          accumulated.finiteCount = count;
          accumulated.mean = accumulated.mean + delta / f32(count);
          accumulated.m2 = accumulated.m2 + delta * (shiftedValue - accumulated.mean);
          accumulated.sum = accumulated.sum + value;
        }
      }`
    : `if (value == ${GPU_COLUMN_PROFILE_NULL_CATEGORY}u) {
        accumulated.nullCount = accumulated.nullCount + 1u;
      } else {
        accumulated.count = accumulated.count + 1u;
        if (value >= categoryCount) {
          accumulated.finiteCount = accumulated.finiteCount + 1u;
        }
        minimumKey = min(minimumKey, value);
        maximumKey = max(maximumKey, value);
      }`;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'moments-tile',
    bindings: [
      read('values', column.values, isNumeric ? 'f32' : 'u32'),
      ...maskBinding,
      write('partials', props.partials),
      write('keys', props.keys, 'atomic<u32>'),
      read('columnCursor', props.columnCursor, 'u32')
    ],
    invocationCount: tileCount * COLUMN_PROFILE_WORKGROUP_SIZE,
    workgroupSize: COLUMN_PROFILE_WORKGROUP_SIZE,
    guardIndex: false,
    declarations: `const ROW_COUNT: u32 = ${rowCount}u;
const TILE_COUNT: u32 = ${tileCount}u;
const TILE_SIZE: u32 = ${COLUMN_PROFILE_TILE_SIZE}u;
const ROWS_PER_INVOCATION: u32 = ${COLUMN_PROFILE_ROWS_PER_INVOCATION}u;
${props.columnConstants}
${COLUMN_ORDERED_KEY_WGSL}
${MOMENTS_WGSL}
var<workgroup> sharedKeys: array<atomic<u32>, ${KEY_STRIDE}>;`,
    body: `let local = localInvocationIndex;
  let tile = workgroupIndex;
  ${READ_COLUMN_WGSL}
  let categoryCount = columnCategoryCounts[column];
  if (local < ${KEY_STRIDE}u) {
    atomicStore(&sharedKeys[local], select(0xffffffffu, 0u, (local & 1u) == 1u));
  }
  workgroupBarrier();
  var accumulated = emptyMoments();
  var minimumKey = 0xffffffffu;
  var maximumKey = 0u;
  var finiteMinimumKey = 0xffffffffu;
  var finiteMaximumKey = 0u;
  for (var step = 0u; step < ROWS_PER_INVOCATION; step = step + 1u) {
    // Uniform trip count: rows past the end are skipped by a predicate, never by a break.
    let row = tile * TILE_SIZE + step * ${COLUMN_PROFILE_WORKGROUP_SIZE}u + local;
    if (row < ROW_COUNT${hasMask ? ' && rowMask[rowMaskOffset + row] != 0u' : ''}) {
      let value = values[valuesOffset + row];
      ${rowBody}
    }
  }
  sharedMoments[local] = accumulated;
  atomicMin(&sharedKeys[0], minimumKey);
  atomicMax(&sharedKeys[1], maximumKey);
  atomicMin(&sharedKeys[2], finiteMinimumKey);
  atomicMax(&sharedKeys[3], finiteMaximumKey);
  reduceSharedMoments(local);
  workgroupBarrier();
  if (local == 0u) {
    let record = sharedMoments[0];
    let base = partialsOffset + (column * TILE_COUNT + tile) * ${RECORD_STRIDE}u;
    partials[base] = record.count;
    partials[base + 1u] = record.nullCount;
    partials[base + 2u] = record.finiteCount;
    partials[base + 3u] = bitcast<u32>(record.mean);
    partials[base + 4u] = bitcast<u32>(record.m2);
    partials[base + 5u] = bitcast<u32>(record.sum);
    partials[base + 6u] = bitcast<u32>(record.origin);
    let keyBase = keysOffset + column * ${KEY_STRIDE}u;
    atomicMin(&keys[keyBase], atomicLoad(&sharedKeys[0]));
    atomicMax(&keys[keyBase + 1u], atomicLoad(&sharedKeys[1]));
    atomicMin(&keys[keyBase + 2u], atomicLoad(&sharedKeys[2]));
    atomicMax(&keys[keyBase + 3u], atomicLoad(&sharedKeys[3]));
  }`
  });
}

function getTopCategoriesNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    topCategoryCount: number;
    columnConstants: string;
    columnCursor: GraphDataView<'uint32'>;
    categoryCounts: GraphDataView<'uint32'>;
    topCategories: GraphDataView<'uint32'>;
    topCategoryCounts: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {topCategoryCount} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'top-categories',
    bindings: [
      read('categoryCounts', props.categoryCounts, 'u32'),
      write('topCategories', props.topCategories),
      write('topCategoryCounts', props.topCategoryCounts),
      read('columnCursor', props.columnCursor, 'u32')
    ],
    invocationCount: COLUMN_PROFILE_WORKGROUP_SIZE,
    workgroupSize: COLUMN_PROFILE_WORKGROUP_SIZE,
    guardIndex: false,
    declarations: `${props.columnConstants}
const TOP_COUNT: u32 = ${topCategoryCount}u;
var<workgroup> bestCounts: array<u32, ${COLUMN_PROFILE_WORKGROUP_SIZE}>;
var<workgroup> bestCodes: array<u32, ${COLUMN_PROFILE_WORKGROUP_SIZE}>;`,
    body: `let local = localInvocationIndex;
  ${READ_COLUMN_WGSL}
  let categoryCount = columnCategoryCounts[column];
  let base = columnCategoryBases[column];
  let stepCount = (categoryCount + ${COLUMN_PROFILE_WORKGROUP_SIZE - 1}u) / ${COLUMN_PROFILE_WORKGROUP_SIZE}u;
  var previousCount = 0u;
  var previousCode = 0u;
  var hasPrevious = false;
  for (var rank = 0u; rank < TOP_COUNT; rank = rank + 1u) {
    var bestCount = 0u;
    var bestCode = 0xffffffffu;
    for (var step = 0u; step < stepCount; step = step + 1u) {
      let code = step * ${COLUMN_PROFILE_WORKGROUP_SIZE}u + local;
      if (code < categoryCount) {
        let count = categoryCounts[categoryCountsOffset + base + code];
        // Strictly after the previous pick in (count descending, code ascending) order.
        let isCandidate = count > 0u && (!hasPrevious || count < previousCount || (count == previousCount && code > previousCode));
        if (isCandidate && count > bestCount) {
          bestCount = count;
          bestCode = code;
        }
      }
    }
    bestCounts[local] = bestCount;
    bestCodes[local] = bestCode;
    for (var stride = ${COLUMN_PROFILE_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride >> 1u) {
      workgroupBarrier();
      if (local < stride) {
        let otherCount = bestCounts[local + stride];
        let otherCode = bestCodes[local + stride];
        if (otherCount > bestCounts[local] || (otherCount == bestCounts[local] && otherCode < bestCodes[local])) {
          bestCounts[local] = otherCount;
          bestCodes[local] = otherCode;
        }
      }
    }
    workgroupBarrier();
    let pickedCount = bestCounts[0];
    let pickedCode = bestCodes[0];
    if (local == 0u) {
      topCategories[topCategoriesOffset + column * TOP_COUNT + rank] = select(0xffffffffu, pickedCode, pickedCount > 0u);
      topCategoryCounts[topCategoryCountsOffset + column * TOP_COUNT + rank] = pickedCount;
    }
    previousCount = pickedCount;
    previousCode = pickedCode;
    hasPrevious = true;
    workgroupBarrier();
  }`
  });
}

function getFinishNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    column: ColumnProfileColumn;
    columnConstants: string;
    columnCursor: GraphDataView<'uint32'>;
    registerCount: number;
    hyperLogLogPrecision: number;
    histogramBinCount: number;
    records: GraphDataView<'uint32'>;
    keys: GraphDataView<'uint32'>;
    registers: GraphDataView<'uint32'>;
    categoryCounts?: GraphDataView<'uint32'>;
    output: ColumnProfileNodeProps['output'];
  }
): GPUCommandNode<Parameters> {
  const {column, registerCount, histogramBinCount, output} = props;
  const isNumeric = column.kind === 'numeric';
  const alphaTimesSquare = getHyperLogLogAlpha(registerCount) * registerCount * registerCount;
  const copyHistogram = !isNumeric && output.histograms && props.categoryCounts;
  const stat = GPU_COLUMN_PROFILE_STATISTIC;
  const bindings: WGSLKernelBinding[] = [
    read('records', props.records, 'u32'),
    read('keys', props.keys, 'u32'),
    // Registers and statistics may occupy disjoint writable ranges in one buffer.
    write('registers', props.registers, 'u32'),
    write('statistics', output.statistics, 'f32')
  ];
  if (output.counts) {
    bindings.push(write('countsOut', output.counts));
  }
  if (copyHistogram) {
    bindings.push(
      read('categoryCounts', props.categoryCounts!, 'u32'),
      write('histogram', output.histograms!)
    );
  }
  bindings.push(write('columnCursor', props.columnCursor));
  const statisticsBody = isNumeric
    ? `let hasValues = count > 0.0;
    let hasFinite = finiteCount > 0.0;
    let variance = select(nan, m2 / finiteCount, hasFinite);
    statistics[statisticsBase + ${stat.minimum}u] = select(nan, decodeOrderedKey(minimumKey), hasValues);
    statistics[statisticsBase + ${stat.maximum}u] = select(nan, decodeOrderedKey(maximumKey), hasValues);
    statistics[statisticsBase + ${stat.sum}u] = select(nan, sum, hasFinite);
    statistics[statisticsBase + ${stat.mean}u] = select(nan, mean, hasFinite);
    statistics[statisticsBase + ${stat.variance}u] = variance;
    statistics[statisticsBase + ${stat.sampleVariance}u] = select(nan, m2 / f32(exactFiniteCount - min(exactFiniteCount, 1u)), exactFiniteCount >= 2u);
    statistics[statisticsBase + ${stat.standardDeviation}u] = select(nan, sqrt(variance), hasFinite);
    statistics[statisticsBase + ${stat.overflowCount}u] = 0.0;`
    : `let hasValues = count > 0.0;
    statistics[statisticsBase + ${stat.minimum}u] = select(nan, f32(minimumKey), hasValues);
    statistics[statisticsBase + ${stat.maximum}u] = select(nan, f32(maximumKey), hasValues);
    statistics[statisticsBase + ${stat.sum}u] = nan;
    statistics[statisticsBase + ${stat.mean}u] = nan;
    statistics[statisticsBase + ${stat.variance}u] = nan;
    statistics[statisticsBase + ${stat.sampleVariance}u] = nan;
    statistics[statisticsBase + ${stat.standardDeviation}u] = nan;
    statistics[statisticsBase + ${stat.overflowCount}u] = finiteCount;`;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'finish',
    bindings,
    invocationCount: COLUMN_PROFILE_WORKGROUP_SIZE,
    workgroupSize: COLUMN_PROFILE_WORKGROUP_SIZE,
    guardIndex: false,
    declarations: `${props.columnConstants}
const REGISTER_COUNT: u32 = ${registerCount}u;
const REGISTER_STEP_COUNT: u32 = ${Math.ceil(registerCount / COLUMN_PROFILE_WORKGROUP_SIZE)}u;
const ALPHA_TIMES_SQUARE: f32 = ${getWGSLFloatLiteral(alphaTimesSquare)};
const BIN_COUNT: u32 = ${histogramBinCount}u;
${COLUMN_ORDERED_KEY_WGSL}
var<workgroup> sharedColumn: u32;
var<workgroup> sharedInverseSums: array<f32, ${COLUMN_PROFILE_WORKGROUP_SIZE}>;
var<workgroup> sharedZeroCounts: array<u32, ${COLUMN_PROFILE_WORKGROUP_SIZE}>;`,
    // One workgroup: only invocation 0 touches the storage cursor, so advancing it at the end
    // cannot race with the other invocations' column reads.
    body: `let local = localInvocationIndex;
  if (local == 0u) {
    sharedColumn = columnCursor[columnCursorOffset];
  }
  let column = workgroupUniformLoad(&sharedColumn);
  var inverseSum = 0.0;
  var zeroCount = 0u;
  for (var step = 0u; step < REGISTER_STEP_COUNT; step = step + 1u) {
    let registerIndex = step * ${COLUMN_PROFILE_WORKGROUP_SIZE}u + local;
    if (registerIndex < REGISTER_COUNT) {
      let rank = registers[registersOffset + column * REGISTER_COUNT + registerIndex];
      // 2^-rank is exact: rank is at most ${33 - props.hyperLogLogPrecision}.
      inverseSum = inverseSum + bitcast<f32>((127u - rank) << 23u);
      zeroCount = zeroCount + select(0u, 1u, rank == 0u);
    }
  }
  sharedInverseSums[local] = inverseSum;
  sharedZeroCounts[local] = zeroCount;
  for (var stride = ${COLUMN_PROFILE_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride >> 1u) {
    workgroupBarrier();
    if (local < stride) {
      sharedInverseSums[local] = sharedInverseSums[local] + sharedInverseSums[local + stride];
      sharedZeroCounts[local] = sharedZeroCounts[local] + sharedZeroCounts[local + stride];
    }
  }
  ${
    copyHistogram
      ? `for (var bin = local; bin < BIN_COUNT; bin = bin + ${COLUMN_PROFILE_WORKGROUP_SIZE}u) {
    histogram[histogramOffset + column * BIN_COUNT + bin] = select(0u, categoryCounts[categoryCountsOffset + columnCategoryBases[column] + bin], bin < columnCategoryCounts[column]);
  }`
      : ''
  }
  if (local == 0u) {
    let recordBase = recordsOffset + column * ${RECORD_STRIDE}u;
    let exactCount = records[recordBase];
    let count = f32(exactCount);
    let exactNullCount = records[recordBase + 1u];
    let nullCount = f32(exactNullCount);
    let exactFiniteCount = records[recordBase + 2u];
    let finiteCount = f32(exactFiniteCount);
    let mean = bitcast<f32>(records[recordBase + 6u]) + bitcast<f32>(records[recordBase + 3u]);
    let m2 = bitcast<f32>(records[recordBase + 4u]);
    let sum = bitcast<f32>(records[recordBase + 5u]);
    let keyBase = keysOffset + column * ${KEY_STRIDE}u;
    let minimumKey = keys[keyBase];
    let maximumKey = keys[keyBase + 1u];
    let nan = getNaN();
    let statisticsBase = statisticsOffset + column * ${GPU_COLUMN_PROFILE_STATISTIC_COUNT}u;
    statistics[statisticsBase + ${stat.count}u] = count;
    statistics[statisticsBase + ${stat.nullCount}u] = nullCount;
    ${statisticsBody}
    let registerCountFloat = f32(REGISTER_COUNT);
    var estimate = ALPHA_TIMES_SQUARE / sharedInverseSums[0];
    let zeros = sharedZeroCounts[0];
    if (zeros == REGISTER_COUNT) {
      estimate = 0.0;
    } else if (estimate <= 2.5 * registerCountFloat && zeros > 0u) {
      // Linear counting m * ln(m / zeros). For few occupied registers the ratio is close to 1 and
      // log() cancels, so use the series of ln(1 + u) with the exact integer u = (m - zeros) / zeros.
      let occupied = f32(REGISTER_COUNT - zeros);
      let ratio = occupied / f32(zeros);
      if (ratio < 0.0625) {
        let r2 = ratio * ratio;
        estimate = registerCountFloat * (ratio - r2 * (0.5 - ratio * (0.33333334 - ratio * (0.25 - ratio * 0.2))));
      } else {
        estimate = registerCountFloat * log(registerCountFloat / f32(zeros));
      }
    }
    statistics[statisticsBase + ${stat.distinctEstimate}u] = estimate;
    ${
      output.counts
        ? `countsOut[countsOutOffset + column * 2u] = exactCount;
    countsOut[countsOutOffset + column * 2u + 1u] = exactNullCount;`
        : ''
    }
    columnCursor[columnCursorOffset] = (column + 1u) % COLUMN_COUNT;
  }`
  });
}
