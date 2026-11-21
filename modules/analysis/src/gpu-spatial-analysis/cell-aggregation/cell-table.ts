// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  GPUSort,
  GPUUint32Gather,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelProps,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {validateSinglePackedView} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {
  GPUCardinalityPort,
  GPUCellTablePort,
  GPUOverflowStatusPort
} from '../../analysis-core/contracts/index';
import {CELL_KEY_WGSL, type CellKeyLayout} from './cell-keys';
import {createCellLevelKernelNode} from './cell-level-kernel';
import {getFixedPointWGSL, ORDERED_KEY_WGSL} from '../../analysis-core/kernels/cell-key-wgsl';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';

/** Key word written to every word of an unused cell table row. */
export const GPU_CELL_EMPTY_KEY_WORD = 0xffffffff;

/**
 * Caller-owned, capacity-bounded cell table written by {@link GPUCellAggregation} and
 * {@link GPUCellRollup}.
 *
 * Capacity is `cells.length`. Rows `[0, count)` hold one occupied cell each, in ascending key
 * order; later rows hold the empty key (`0xffffffff` in both words), a zero count and sum, and
 * NaN extremes. When more cells exist than fit, the cells with the largest keys are dropped and
 * `status.overflow` is 1. Every column must hold `cells.length` rows.
 */
export type GPUCellTable = GPUCellTablePort & {
  /**
   * 64-bit cell keys as little-endian `(low, high)` `uint32` words, the layout of Arrow `Uint64`
   * columns and of `GPUH3CellProjection` input, so an H3 table feeds the projection directly.
   */
  cells: GraphDataView<'uint32x2'>;
  /** Rows aggregated into each cell. */
  counts: GraphDataView<'uint32'>;
  /**
   * Fixed-point sums: signed 64-bit two's-complement integers as little-endian words, equal
   * to the sum over rows of `roundHalfEven(fround(value * sumScale))`. Integer sums are
   * associative, so a roll-up equals a direct aggregation bit for bit. The integer sum is exact;
   * each row is first rounded to a multiple of `1 / sumScale` (an error of up to `2^-17` per row
   * at the default 65536), so choose `sumScale` for the data (`getGPUCellSumScale`). Required on a
   * source table to roll sums up to a coarser level.
   *
   * Decode a readback with `decodeGPUGroupSum(words, row, sumScale)` from
   * `@luma-contributors/analysis/dataframe-analysis` (the shared
   * [group-sum decoder](../../dataframe-analysis/group-statistics/decode-gpu-group-sum.ts)).
   * Each pair is low word then high word; bit 31 of the high word is the sign of the
   * two's-complement integer. The decoder returns that exact `scaledInteger` as bigint and
   * `value = scaledInteger / sumScale` in the original measure's units (rounded to binary64).
   * For example `[0xffff0000, 0xffffffff]` represents `-65536n`, hence `-1` at scale 65536.
   * An unsaturated total requires `status.sumOverflow == 0`; if the flag was not requested,
   * the caller must independently guarantee row and total headroom. Decoding cannot recover
   * clipped values or source cells lost to either capacity flag.
   *
   * Headroom: a scaled row saturates at `CELL_MAXIMUM_SCALED_VALUE` (just under `2^62`, a row of
   * about `7.04e13` at the default scale) and a cell sum saturates at `±(2^63 - 1)` (about
   * `1.4e14` at the default scale) instead of wrapping; either sets `status.sumOverflow`.
   */
  sums?: GraphDataView<'uint32x2'>;
  /** `sums / sumScale` as f32, for styling. */
  sumValues?: GraphDataView<'float32'>;
  /** Minimum value per cell. */
  minimums?: GraphDataView<'float32'>;
  /** Maximum value per cell. */
  maximums?: GraphDataView<'float32'>;
  /**
   * One-row status scalars: `count` receives the occupied row count, `min(requiredCount,
   * cells.length)`. `overflow` reports only this table's output capacity; `candidateOverflow`
   * reports cells lost in source tables. `requiredCount` counts retained-input occupied cells
   * before destination truncation; it is a lower bound when `candidateOverflow` is nonzero.
   * `candidateOverflow` is required on rollup outputs and coarser pyramid levels; raw aggregation
   * accepts it optionally and writes zero each submission. Missing source flags mean no prior loss.
   * `sumOverflow` receives 1 when a retained cell's scaled row or sum saturated (see `sums`), or when a
   * source table reported it; the saturated sums are written either way.
   */
  status: GPUCardinalityPort &
    GPUOverflowStatusPort & {
      candidateOverflow?: GraphDataView<'uint32'>;
      sumOverflow?: GraphDataView<'uint32'>;
    };
};

/** Capture declared status ports once, including inherited getters, without metadata. @internal */
export function snapshotCellTableStatus<Status extends GPUCellTable['status']>(
  status: Status
): Status {
  return snapshotConstructionFields(status, [
    'count',
    'overflow',
    'candidateOverflow',
    'requiredCount',
    'sumOverflow'
  ]);
}

/** Capture table topology once; graph views remain borrowed and their contents remain dynamic. @internal */
export function snapshotCellTable(table: GPUCellTable): GPUCellTable {
  const fields = snapshotConstructionFields(table, [
    'cells',
    'counts',
    'sums',
    'sumValues',
    'minimums',
    'maximums',
    'status'
  ]);
  return Object.freeze({...fields, status: snapshotCellTableStatus(fields.status)});
}

/** Returns every view of a cell table, for graph membership and aliasing checks. @internal */
export function getCellTableViews(table: GPUCellTable): (GraphDataView | undefined)[] {
  return [
    table.cells,
    table.counts,
    table.sums,
    table.sumValues,
    table.minimums,
    table.maximums,
    table.status.count,
    table.status.overflow,
    table.status.candidateOverflow,
    table.status.requiredCount,
    table.status.sumOverflow
  ];
}

/** Returns the first row of `table.cells` within its buffer, for instanced draws. */
export function getCellTableFirstRow(table: GPUCellTable): number {
  return table.cells.byteOffset / table.cells.rowByteLength;
}

/** Validates formats, packing and row counts of a cell table. @internal */
export function validateCellTable(id: string, name: string, table: GPUCellTable): void {
  validateSinglePackedView(table.cells, ['uint32x2'], `${id} ${name}.cells`);
  const capacity = table.cells.length;
  if (capacity < 1) {
    throw new Error(`${id} ${name}.cells must hold at least one row`);
  }
  validatePackedUint32View(table.counts, `${id} ${name}.counts`);
  if (table.sums) {
    validateSinglePackedView(table.sums, ['uint32x2'], `${id} ${name}.sums`);
  }
  for (const column of ['sumValues', 'minimums', 'maximums'] as const) {
    const view = table[column];
    if (view) {
      validateSinglePackedView(view, ['float32'], `${id} ${name}.${column}`);
    }
  }
  for (const column of ['counts', 'sums', 'sumValues', 'minimums', 'maximums'] as const) {
    const view = table[column];
    if (view && view.length !== capacity) {
      throw new Error(`${id} ${name}.${column} must have the same length as ${name}.cells`);
    }
  }
  for (const column of [
    'count',
    'overflow',
    'candidateOverflow',
    'requiredCount',
    'sumOverflow'
  ] as const) {
    const view = table.status[column];
    if (!view) {
      continue;
    }
    validatePackedUint32View(view, `${id} ${name}.status.${column}`);
    if (view.length < 1) {
      throw new Error(`${id} ${name}.status.${column} must contain one uint32 row`);
    }
  }
}

/** Per-row source of the reductions: raw rows or the cells of a finer table. @internal */
export type CellTableReductionSource =
  | {
      kind: 'rows';
      /** Per-row values; required for sums and extremes. */
      values?: GraphDataView<'float32'>;
    }
  | {kind: 'cells'; table: GPUCellTable};

/** Private tile accumulator: preserve all 96 bits and per-key row clamping until union. @internal */
export type CellTableSumAccumulator = {
  guards: GraphDataView<'uint32'>;
  rowOverflow: GraphDataView<'uint32'>;
};

/** Properties of {@link getCellTableNodes}. @internal */
export type CellTableNodesProps = {
  id: string;
  operation: string;
  /** Layout of the table's resolution. */
  layout: CellKeyLayout;
  /** Compact keys (low words) per row; the invalid key has bit `layout.width` set. */
  keyLow: GraphDataView<'uint32'>;
  /** High words of the compact keys; present when `layout.width + 1 > 32`. */
  keyHigh?: GraphDataView<'uint32'>;
  /** Row indices `0..n-1`, needed when `sorted` is false. */
  rowIds?: GraphDataView<'uint32'>;
  /** True when rows are already in ascending key order with invalid rows last. */
  sorted: boolean;
  source: CellTableReductionSource;
  sumScale: number;
  output: GPUCellTable;
  /** Source loss flags propagated to `output.status.candidateOverflow`. */
  candidateSources?: readonly GraphDataView<'uint32'>[];
  /** Extra one-row destinations of the clamped count. */
  extraCounts?: readonly GraphDataView<'uint32'>[];
  /** Source sum-saturation flags propagated to `output.status.sumOverflow`. */
  sumOverflowSources?: readonly GraphDataView<'uint32'>[];
  /** Internal tiles defer 64-bit saturation until the final retained-key merge. */
  sumAccumulator?: CellTableSumAccumulator;
};

/** Rows folded per workgroup by the accumulation passes of {@link getCellTableNodes}. */
const ACCUMULATE_TILE_SIZE = 256;

/*
 * Per-cell float extremes use the tile run-folding kernel (`accumulate-extremes`) rather than
 * upstream `GPUGroupAggregation` min/max: one fused pass for both extremes and one atomic per run
 * of equal cells per tile. The upstream path (gather + two passes, one atomic per row) gave
 * identical bits but measured 4.6-5.2 ms vs 3.1-3.3 ms per frame at 1M rows with a 70% hot cell,
 * and was slower on uniform data too (2026-10-13).
 */

/**
 * 96-bit accumulation of the signed 64-bit fixed-point sums: a guard word above `(high, low)`
 * keeps the carries out of the high word, so a sum outside the signed 64-bit range is detected
 * (the guard differs from the sign extension of the high word) instead of wrapping. The scaled
 * rows are below `2^62` and at most `2^32` rows meet in one cell, so the 96-bit total is exact.
 * Values are `vec3u(guard, high, low)`. @internal
 */
export const CELL_SUM_GUARD_WGSL = /* wgsl */ `
fn cellGetSignWord(high: u32) -> u32 {
  return select(0u, 0xffffffffu, (high & 0x80000000u) != 0u);
}

/** Sign-extends a two's-complement 64-bit \`vec2u(high, low)\` to \`vec3u(guard, high, low)\`. */
fn cellExtendI64(value: vec2u) -> vec3u {
  return vec3u(cellGetSignWord(value.x), value.x, value.y);
}

/** \`left + right\` modulo 2^96. */
fn cellAddI96(left: vec3u, right: vec3u) -> vec3u {
  let low = left.z + right.z;
  let partialHigh = left.y + right.y;
  let high = partialHigh + select(0u, 1u, low < left.z);
  let highCarry = select(0u, 1u, partialHigh < left.y) + select(0u, 1u, high < partialHigh);
  return vec3u(left.x + right.x + highCarry, high, low);
}

/** Whether a 96-bit sum is outside the signed 64-bit range. */
fn cellIsI64Overflow(guardWord: u32, high: u32) -> bool {
  return guardWord != cellGetSignWord(high);
}

/** \`2^63 - 1\` or \`-(2^63 - 1)\` as \`vec2u(high, low)\`, by the sign of the 96-bit guard. */
fn cellGetSaturatedI64(guardWord: u32) -> vec2u {
  let isNegative = (guardWord & 0x80000000u) != 0u;
  return select(vec2u(0x7fffffffu, 0xffffffffu), vec2u(0x80000000u, 1u), isNegative);
}
`;

/** Returns whether compact keys at this layout need a high word. @internal */
export function hasCellKeyHighWord(layout: CellKeyLayout): boolean {
  return layout.width + 1 > 32;
}

/** WGSL statement assigning `isInvalid` for the compact key `(high, low)`. */
function getInvalidKeyExpression(layout: CellKeyLayout, high: string, low: string): string {
  return hasCellKeyHighWord(layout)
    ? `(${high} >> (CELL_WIDTH - 32u)) != 0u`
    : `(${low} >> CELL_WIDTH) != 0u`;
}

/**
 * Sorts compact keys (unless already sorted), finds cell boundaries, and writes the cell table:
 * keys, counts, fixed-point sums, f32 sums, extremes, count and overflow. Rows accumulate into
 * their table row with integer atomics (64-bit sums as two words with an explicit carry, extremes
 * as order-preserving u32 keys), so every result is independent of thread order. Work is
 * O(rows + capacity).
 *
 * @internal
 */
export function getCellTableNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: CellTableNodesProps
): GPUCommandNode<Parameters>[] {
  const {id, operation, layout, output, source} = props;
  const rowCount = props.keyLow.length;
  const capacity = output.cells.length;
  const twoWords = hasCellKeyHighWord(layout);
  const u32 = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', length);
  const read = (name: string, view: GraphDataView, type: 'u32' | 'f32' = 'u32') =>
    ({name, view, type, access: 'read'}) as WGSLKernelBinding;
  const write = (name: string, view: GraphDataView, type: 'u32' | 'f32' = 'u32') =>
    ({name, view, type, access: 'read_write'}) as WGSLKernelBinding;
  const nodes: GPUCommandNode<Parameters>[] = [];
  const createTableKernel = (
    kernelProps: WGSLKernelProps<Parameters>
  ): GPUCommandNode<Parameters> => {
    const result = createCellLevelKernelNode(graph, kernelProps, {
      ROW_COUNT: rowCount,
      CAPACITY: capacity,
      CELL_WIDTH: layout.width,
      CELL_LOW_BIT: layout.lowBit,
      CELL_HEADER: layout.headerHigh,
      CELL_RESOLUTION: layout.resolution
    });
    nodes.push(...result.uploads);
    return result.node;
  };

  // 1. Ascending (high, low) order by two stable LSD sorts, or one when the key fits a word.
  let sortedLow = props.keyLow;
  let sortedHigh = props.keyHigh;
  let permutation: GraphDataView<'uint32'> | undefined;
  if (!props.sorted) {
    if (!props.rowIds) {
      throw new Error(`${id} needs row IDs to sort`);
    }
    const lowKeyBits = twoWords ? 32 : layout.width + 1;
    const firstKeys = u32('sorted-low-first', rowCount);
    const firstRows = u32('sorted-rows-first', rowCount);
    nodes.push(
      ...new GPUSort({
        id: `${id}-sort-low`,
        keys: props.keyLow,
        values: props.rowIds,
        outputKeys: firstKeys,
        outputValues: firstRows,
        keyBits: lowKeyBits
      }).getCommandNodes(graph)
    );
    if (twoWords && props.keyHigh) {
      const highByFirst = u32('high-by-low-order', rowCount);
      nodes.push(
        // `firstRows` holds the row ids of a sort permutation, so no index is out of range.
        ...new GPUUint32Gather({
          id: `${id}-gather-high`,
          source: props.keyHigh,
          indices: firstRows,
          output: highByFirst
        }).getCommandNodes(graph)
      );
      sortedHigh = u32('sorted-high', rowCount);
      permutation = u32('permutation', rowCount);
      nodes.push(
        ...new GPUSort({
          id: `${id}-sort-high`,
          keys: highByFirst,
          values: firstRows,
          outputKeys: sortedHigh,
          outputValues: permutation,
          keyBits: layout.width + 1 - 32
        }).getCommandNodes(graph)
      );
      sortedLow = u32('sorted-low', rowCount);
      nodes.push(
        ...new GPUUint32Gather({
          id: `${id}-gather-low`,
          source: props.keyLow,
          indices: permutation,
          output: sortedLow
        }).getCommandNodes(graph)
      );
    } else {
      sortedLow = firstKeys;
      permutation = firstRows;
    }
  }

  const sortedKeyBindings = (): WGSLKernelBinding[] => [
    read('sortedLow', sortedLow),
    ...(twoWords && sortedHigh ? [read('sortedHigh', sortedHigh)] : [])
  ];
  const keyAt = (row: string) =>
    twoWords
      ? `vec2u(sortedHigh[sortedHighOffset + ${row}], sortedLow[sortedLowOffset + ${row}])`
      : `vec2u(0u, sortedLow[sortedLowOffset + ${row}])`;

  // 2. Segment heads and the number of valid rows (valid rows are a prefix of the sorted order).
  const heads = u32('heads', rowCount);
  const validCount = u32('valid-count', 1);
  nodes.push(
    createTableKernel({
      id: `${id}-heads`,
      operation,
      variant: 'heads',
      bindings: [...sortedKeyBindings(), write('heads', heads), write('validCount', validCount)],
      invocationCount: rowCount,
      declarations: `fn isInvalidKey(key: vec2u) -> bool {
  return ${getInvalidKeyExpression(layout, 'key.x', 'key.y')};
}`,
      body: `let key = ${keyAt('index')};
  let isValid = !isInvalidKey(key);
  var isHead = isValid;
  if (isValid && index > 0u) {
    isHead = any(key != ${keyAt('index - 1u')});
  }
  heads[headsOffset + index] = select(0u, 1u, isHead);
  let isLastValid = isValid && (index + 1u == ROW_COUNT || isInvalidKey(${keyAt('index + 1u')}));
  if (isLastValid) {
    validCount[validCountOffset] = index + 1u;
  } else if (index == 0u && !isValid) {
    validCount[validCountOffset] = 0u;
  }`
    })
  );

  // 3. Exclusive scan of heads gives each head its cell index.
  const cellIndices = u32('cell-indices', rowCount);
  nodes.push(
    ...new GPUScan({
      id: `${id}-scan`,
      input: heads,
      output: cellIndices,
      mode: 'exclusive'
    }).getCommandNodes(graph)
  );

  // 4. Cell row offsets for the first `capacity + 1` cells, the unclamped cell total, and each
  // row's table row (`NO_CELL` for invalid rows and cells past the capacity).
  const offsets = u32('offsets', capacity + 1);
  const cellTotal = u32('cell-total', 1);
  const rowCells = u32('row-cells', rowCount);
  nodes.push(
    createTableKernel({
      id: `${id}-offsets`,
      operation,
      variant: 'offsets',
      bindings: [
        read('heads', heads),
        read('cellIndices', cellIndices),
        read('validCount', validCount),
        write('offsets', offsets),
        write('cellTotal', cellTotal),
        write('rowCells', rowCells)
      ],
      invocationCount: rowCount,
      body: `let isHead = heads[headsOffset + index] != 0u;
  let cell = cellIndices[cellIndicesOffset + index] + select(0u, 1u, isHead) - 1u;
  let isValid = index < validCount[validCountOffset];
  rowCells[rowCellsOffset + index] = select(0xffffffffu, cell, isValid && cell < CAPACITY);
  if (isHead && cell <= CAPACITY) {
    offsets[offsetsOffset + cell] = index;
  }
  if (index == 0u) {
    let last = ROW_COUNT - 1u;
    let total = cellIndices[cellIndicesOffset + last] + heads[headsOffset + last];
    cellTotal[cellTotalOffset] = total;
    if (total <= CAPACITY) {
      offsets[offsetsOffset + total] = validCount[validCountOffset];
    }
  }`
    })
  );

  const offsetsDeclarations = `fn getCellCount() -> u32 {
  return min(cellTotal[cellTotalOffset], CAPACITY);
}`;

  // 5. Keys, and counts of raw rows, one invocation per table row.
  const childTable = source.kind === 'cells' ? source.table : undefined;
  nodes.push(
    createTableKernel({
      id: `${id}-cells`,
      operation,
      variant: 'cells',
      bindings: [
        read('offsets', offsets),
        read('cellTotal', cellTotal),
        ...sortedKeyBindings(),
        write('cellsOut', output.cells),
        write('countsOut', output.counts)
      ],
      invocationCount: capacity,
      declarations: `${offsetsDeclarations}
${CELL_KEY_WGSL}`,
      body: `if (index >= getCellCount()) {
    cellsOut[cellsOutOffset + 2u * index] = 0xffffffffu;
    cellsOut[cellsOutOffset + 2u * index + 1u] = 0xffffffffu;
    countsOut[countsOutOffset + index] = 0u;
    return;
  }
  let begin = offsets[offsetsOffset + index];
  let key = cellGetKey(${keyAt('begin')}, CELL_HEADER, CELL_RESOLUTION, CELL_LOW_BIT);
  cellsOut[cellsOutOffset + 2u * index] = key.y;
  cellsOut[cellsOutOffset + 2u * index + 1u] = key.x;
  // Roll-ups sum the child counts in the accumulation pass.
  countsOut[countsOutOffset + index] = ${childTable ? '0u' : `offsets[offsetsOffset + index + 1u] - begin`};`
    })
  );

  const wantsSums = Boolean(output.sums || output.sumValues);
  const wantsExtremes = Boolean(output.minimums || output.maximums);
  const rowValues = source.kind === 'rows' ? source.values : undefined;
  if ((wantsSums || wantsExtremes) && source.kind === 'rows' && !rowValues) {
    throw new Error(`${id} needs values to compute sums and extremes`);
  }
  if (wantsSums && childTable && !childTable.sums) {
    throw new Error(`${id} source table needs sums to roll up sums`);
  }
  if (
    childTable &&
    ((output.minimums && !childTable.minimums) || (output.maximums && !childTable.maximums))
  ) {
    throw new Error(`${id} source table needs the extremes it rolls up`);
  }
  // Integer accumulators: 64-bit sums as two atomic words with an explicit carry, and extremes as
  // order-preserving u32 keys. Integer atomics give the same result in any order.
  const sums = wantsSums
    ? (output.sums ?? createTransientView(graph, `${id}-sums-scratch`, 'uint32x2', capacity))
    : undefined;
  // Guard words above the 64-bit sums, and the table's sum-saturation flag.
  const guards = sums ? (props.sumAccumulator?.guards ?? u32('sum-guards', capacity)) : undefined;
  const sumFlag = sums ? u32('sum-overflow-flag', 1) : undefined;
  const minimumKeys = output.minimums ? u32('minimum-keys', capacity) : undefined;
  const maximumKeys = output.maximums ? u32('maximum-keys', capacity) : undefined;
  const sourceRow = permutation ? 'permutation[permutationOffset + index]' : 'index';
  const valueBindings = (): WGSLKernelBinding[] =>
    rowValues
      ? [
          ...(permutation ? [read('permutation', permutation)] : []),
          read('values', rowValues, 'f32')
        ]
      : [];

  // 6. Clear the accumulators.
  if (sums || minimumKeys || maximumKeys) {
    nodes.push(
      createTableKernel({
        id: `${id}-clear`,
        operation,
        variant: 'clear',
        bindings: [
          ...(sums && guards && sumFlag
            ? [write('sums', sums), write('guards', guards), write('sumFlag', sumFlag)]
            : []),
          ...(minimumKeys ? [write('minimumKeys', minimumKeys)] : []),
          ...(maximumKeys ? [write('maximumKeys', maximumKeys)] : [])
        ],
        invocationCount: capacity,
        body: `${sums ? 'sums[sumsOffset + 2u * index] = 0u;\n  sums[sumsOffset + 2u * index + 1u] = 0u;\n  guards[guardsOffset + index] = 0u;\n  if (index == 0u) { sumFlag[sumFlagOffset] = 0u; }' : ''}
  ${minimumKeys ? 'minimumKeys[minimumKeysOffset + index] = 0xffffffffu;' : ''}
  ${maximumKeys ? 'maximumKeys[maximumKeysOffset + index] = 0u;' : ''}`
      })
    );
  }

  // 7. Accumulate sums (and child counts), then extremes, one invocation per source row.
  if (sums || childTable) {
    // A raw row whose scaled value saturates in `cellScaleValue` raises the sum flag.
    const contribution = rowValues
      ? `cellExtendI64(cellScaleValue(rowValue))`
      : 'cellExtendI64(vec2u(childSums[childSumsOffset + 2u * index + 1u], childSums[childSumsOffset + 2u * index]))';
    const rowSaturation = rowValues
      ? `let rowValue = values[valuesOffset + ${sourceRow}];
      if (abs(round(rowValue * SUM_SCALE)) > MAXIMUM_SCALED_VALUE) {
        atomicOr(&sumFlag[sumFlagOffset], 1u);
      }`
      : '';
    nodes.push(
      createTableKernel({
        id: `${id}-accumulate-sums`,
        operation,
        variant: 'accumulate-sums',
        bindings: [
          read('rowCells', rowCells),
          ...valueBindings(),
          ...(childTable && sums ? [read('childSums', childTable.sums!)] : []),
          ...(childTable
            ? [
                read('childCounts', childTable.counts),
                {
                  name: 'countsOut',
                  view: output.counts,
                  type: 'atomic<u32>',
                  access: 'read_write'
                } as const
              ]
            : []),
          ...(sums && guards
            ? [
                {name: 'sums', view: sums, type: 'atomic<u32>', access: 'read_write'} as const,
                {name: 'guards', view: guards, type: 'atomic<u32>', access: 'read_write'} as const
              ]
            : []),
          ...(sums && sumFlag && rowValues
            ? [{name: 'sumFlag', view: sumFlag, type: 'atomic<u32>', access: 'read_write'} as const]
            : [])
        ],
        workgroupSize: ACCUMULATE_TILE_SIZE,
        invocationCount: rowCount,
        guardIndex: false,
        declarations: `${getFixedPointWGSL(props.sumScale)}
${CELL_SUM_GUARD_WGSL}
const TILE_SIZE: u32 = ${ACCUMULATE_TILE_SIZE}u;
const NO_CELL: u32 = 0xffffffffu;
var<workgroup> tileCells: array<u32, ${ACCUMULATE_TILE_SIZE}>;
${sums ? `var<workgroup> tileGuard: array<u32, ${ACCUMULATE_TILE_SIZE}>;\nvar<workgroup> tileHigh: array<u32, ${ACCUMULATE_TILE_SIZE}>;\nvar<workgroup> tileLow: array<u32, ${ACCUMULATE_TILE_SIZE}>;` : ''}
${childTable ? `var<workgroup> tileCounts: array<u32, ${ACCUMULATE_TILE_SIZE}>;` : ''}`,
        // Rows are in cell order, so a cell's rows are contiguous. A segmented inclusive scan over
        // the tile folds each run of equal cells in shared memory, and only the last row of a run
        // issues the global atomics: a hot cell costs one atomic per tile instead of one per row.
        // Integer sums are associative modulo 2^96, so the result is unchanged. Each word's
        // atomic add returns the previous word, so every carry (low to high, high to guard) is
        // exact in any order. Barriers are in uniform control flow (constant-bound loop, no early
        // return).
        body: `let lane = localInvocationIndex;
  var cell = NO_CELL;
  ${sums ? 'var value = vec3u(0u, 0u, 0u);' : ''}
  ${childTable ? 'var childCount = 0u;' : ''}
  if (index < INVOCATION_COUNT) {
    cell = rowCells[rowCellsOffset + index];
    if (cell != NO_CELL) {
      ${sums ? `${rowSaturation}\n      value = ${contribution};` : ''}
      ${childTable ? 'childCount = childCounts[childCountsOffset + index];' : ''}
    }
  }
  tileCells[lane] = cell;
  ${sums ? 'tileGuard[lane] = value.x;\n  tileHigh[lane] = value.y;\n  tileLow[lane] = value.z;' : ''}
  ${childTable ? 'tileCounts[lane] = childCount;' : ''}
  workgroupBarrier();
  for (var stride = 1u; stride < TILE_SIZE; stride = stride << 1u) {
    ${sums ? 'var addValue = vec3u(0u, 0u, 0u);' : ''}
    ${childTable ? 'var addCount = 0u;' : ''}
    if (lane >= stride && tileCells[lane - stride] == cell) {
      ${sums ? 'addValue = vec3u(tileGuard[lane - stride], tileHigh[lane - stride], tileLow[lane - stride]);' : ''}
      ${childTable ? 'addCount = tileCounts[lane - stride];' : ''}
    }
    workgroupBarrier();
    ${sums ? 'value = cellAddI96(value, addValue);\n    tileGuard[lane] = value.x;\n    tileHigh[lane] = value.y;\n    tileLow[lane] = value.z;' : ''}
    ${childTable ? 'childCount += addCount;\n    tileCounts[lane] = childCount;' : ''}
    workgroupBarrier();
  }
  let isRunEnd = lane == TILE_SIZE - 1u || tileCells[lane + 1u] != cell;
  if (cell == NO_CELL || !isRunEnd) {
    return;
  }
  ${childTable ? 'atomicAdd(&countsOut[countsOutOffset + cell], childCount);' : ''}
  ${
    sums
      ? `let previousLow = atomicAdd(&sums[sumsOffset + 2u * cell], value.z);
  let high = value.y + select(0u, 1u, previousLow + value.z < previousLow);
  var guardCarry = value.x + select(0u, 1u, high < value.y);
  if (high != 0u) {
    let previousHigh = atomicAdd(&sums[sumsOffset + 2u * cell + 1u], high);
    guardCarry += select(0u, 1u, previousHigh + high < previousHigh);
  }
  if (guardCarry != 0u) {
    atomicAdd(&guards[guardsOffset + cell], guardCarry);
  }`
      : ''
  }`
      })
    );
  }
  if (props.sumAccumulator && rowValues) {
    const rowOverflow = props.sumAccumulator.rowOverflow;
    nodes.push(
      createTableKernel({
        id: `${id}-row-overflow-clear`,
        operation,
        variant: 'row-overflow-clear',
        invocationCount: capacity,
        bindings: [write('rowOverflow', rowOverflow)],
        body: 'rowOverflow[rowOverflowOffset + index] = 0u;'
      }),
      createTableKernel({
        id: `${id}-row-overflow`,
        operation,
        variant: 'row-overflow',
        invocationCount: rowCount,
        bindings: [
          read('rowCells', rowCells),
          ...valueBindings(),
          {name: 'rowOverflow', view: rowOverflow, type: 'atomic<u32>', access: 'read_write'}
        ],
        declarations: getFixedPointWGSL(props.sumScale),
        body: `let cell = rowCells[rowCellsOffset + index];
  if (cell == 0xffffffffu) { return; }
  let value = values[valuesOffset + ${sourceRow}];
  if (abs(round(value * SUM_SCALE)) > MAXIMUM_SCALED_VALUE) {
    atomicOr(&rowOverflow[rowOverflowOffset + cell], 1u);
  }`
      })
    );
  }
  // 7b. Saturate sums outside the signed 64-bit range instead of leaving them wrapped.
  if (sums && guards && sumFlag && !props.sumAccumulator) {
    nodes.push(
      createTableKernel({
        id: `${id}-saturate-sums`,
        operation,
        variant: 'saturate-sums',
        bindings: [
          read('guards', guards),
          write('sums', sums),
          {name: 'sumFlag', view: sumFlag, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: capacity,
        declarations: CELL_SUM_GUARD_WGSL,
        body: `let guardWord = guards[guardsOffset + index];
  if (cellIsI64Overflow(guardWord, sums[sumsOffset + 2u * index + 1u])) {
    let saturated = cellGetSaturatedI64(guardWord);
    sums[sumsOffset + 2u * index] = saturated.y;
    sums[sumsOffset + 2u * index + 1u] = saturated.x;
    atomicOr(&sumFlag[sumFlagOffset], 1u);
  }`
      })
    );
  }
  if (minimumKeys || maximumKeys) {
    const minimumSource = rowValues
      ? `values[valuesOffset + ${sourceRow}]`
      : 'childMinimums[childMinimumsOffset + index]';
    const maximumSource = rowValues
      ? `values[valuesOffset + ${sourceRow}]`
      : 'childMaximums[childMaximumsOffset + index]';
    nodes.push(
      createTableKernel({
        id: `${id}-accumulate-extremes`,
        operation,
        variant: 'accumulate-extremes',
        bindings: [
          read('rowCells', rowCells),
          ...valueBindings(),
          ...(childTable && minimumKeys
            ? [read('childMinimums', childTable.minimums!, 'f32')]
            : []),
          ...(childTable && maximumKeys
            ? [read('childMaximums', childTable.maximums!, 'f32')]
            : []),
          ...(minimumKeys
            ? [
                {
                  name: 'minimumKeys',
                  view: minimumKeys,
                  type: 'atomic<u32>',
                  access: 'read_write'
                } as const
              ]
            : []),
          ...(maximumKeys
            ? [
                {
                  name: 'maximumKeys',
                  view: maximumKeys,
                  type: 'atomic<u32>',
                  access: 'read_write'
                } as const
              ]
            : [])
        ],
        workgroupSize: ACCUMULATE_TILE_SIZE,
        invocationCount: rowCount,
        guardIndex: false,
        declarations: `${ORDERED_KEY_WGSL}
const TILE_SIZE: u32 = ${ACCUMULATE_TILE_SIZE}u;
const NO_CELL: u32 = 0xffffffffu;
var<workgroup> tileCells: array<u32, ${ACCUMULATE_TILE_SIZE}>;
${minimumKeys ? `var<workgroup> tileMinimums: array<u32, ${ACCUMULATE_TILE_SIZE}>;` : ''}
${maximumKeys ? `var<workgroup> tileMaximums: array<u32, ${ACCUMULATE_TILE_SIZE}>;` : ''}`,
        // Same run-folding scheme as the sum pass, with min / max in place of the 64-bit add.
        body: `let lane = localInvocationIndex;
  var cell = NO_CELL;
  ${minimumKeys ? 'var minimumKey = 0xffffffffu;' : ''}
  ${maximumKeys ? 'var maximumKey = 0u;' : ''}
  if (index < INVOCATION_COUNT) {
    cell = rowCells[rowCellsOffset + index];
    if (cell != NO_CELL) {
      ${minimumKeys ? `minimumKey = getOrderedKey(${minimumSource});` : ''}
      ${maximumKeys ? `maximumKey = getOrderedKey(${maximumSource});` : ''}
    }
  }
  tileCells[lane] = cell;
  ${minimumKeys ? 'tileMinimums[lane] = minimumKey;' : ''}
  ${maximumKeys ? 'tileMaximums[lane] = maximumKey;' : ''}
  workgroupBarrier();
  for (var stride = 1u; stride < TILE_SIZE; stride = stride << 1u) {
    ${minimumKeys ? 'var otherMinimum = 0xffffffffu;' : ''}
    ${maximumKeys ? 'var otherMaximum = 0u;' : ''}
    if (lane >= stride && tileCells[lane - stride] == cell) {
      ${minimumKeys ? 'otherMinimum = tileMinimums[lane - stride];' : ''}
      ${maximumKeys ? 'otherMaximum = tileMaximums[lane - stride];' : ''}
    }
    workgroupBarrier();
    ${minimumKeys ? 'minimumKey = min(minimumKey, otherMinimum);\n    tileMinimums[lane] = minimumKey;' : ''}
    ${maximumKeys ? 'maximumKey = max(maximumKey, otherMaximum);\n    tileMaximums[lane] = maximumKey;' : ''}
    workgroupBarrier();
  }
  let isRunEnd = lane == TILE_SIZE - 1u || tileCells[lane + 1u] != cell;
  if (cell == NO_CELL || !isRunEnd) {
    return;
  }
  ${minimumKeys ? 'atomicMin(&minimumKeys[minimumKeysOffset + cell], minimumKey);' : ''}
  ${maximumKeys ? 'atomicMax(&maximumKeys[maximumKeysOffset + cell], maximumKey);' : ''}`
      })
    );
  }

  // 8. Finish: f32 sums and decoded extremes (NaN for empty rows).
  if (output.sumValues || minimumKeys || maximumKeys) {
    nodes.push(
      createTableKernel({
        id: `${id}-finish`,
        operation,
        variant: 'finish',
        bindings: [
          read('cellTotal', cellTotal),
          ...(output.sumValues && sums ? [read('sums', sums)] : []),
          ...(minimumKeys ? [read('minimumKeys', minimumKeys)] : []),
          ...(maximumKeys ? [read('maximumKeys', maximumKeys)] : []),
          ...(output.sumValues ? [write('sumValuesOut', output.sumValues, 'f32')] : []),
          ...(output.minimums && minimumKeys ? [write('minimumsOut', output.minimums, 'f32')] : []),
          ...(output.maximums && maximumKeys ? [write('maximumsOut', output.maximums, 'f32')] : [])
        ],
        invocationCount: capacity,
        declarations: `${offsetsDeclarations}
${ORDERED_KEY_WGSL}
${getFixedPointWGSL(props.sumScale)}
${SCALED_QUOTIENT_WGSL}`,
        body: `let isOccupied = index < getCellCount();
  let nan = getNaN();
  ${
    output.sumValues
      ? `let total = vec2u(sums[sumsOffset + 2u * index + 1u], sums[sumsOffset + 2u * index]);
  // Exponent-scaled quotient: a division by SUM_SCALE may compile as a multiplication by the
  // reciprocal, which is subnormal (and flushed to zero) for scales above 2^126.
  sumValuesOut[sumValuesOutOffset + index] = getScaledQuotient(cellI64ToF32(total), SUM_SCALE, 1.0, 0);`
      : ''
  }
  ${
    output.minimums && minimumKeys
      ? 'minimumsOut[minimumsOutOffset + index] = select(nan, decodeOrderedKey(minimumKeys[minimumKeysOffset + index]), isOccupied);'
      : ''
  }
  ${
    output.maximums && maximumKeys
      ? 'maximumsOut[maximumsOutOffset + index] = select(nan, decodeOrderedKey(maximumKeys[maximumKeysOffset + index]), isOccupied);'
      : ''
  }`
      })
    );
  }

  // 9. Publish with runtime capacity; each level keeps the same pipeline for this binding shape.
  const candidateSources = props.candidateSources ?? [];
  const extraCounts = props.extraCounts ?? [];
  nodes.push(
    createTableKernel({
      id: `${id}-publish`,
      operation,
      variant: 'publish',
      invocationCount: 1,
      bindings: [
        read('cellTotal', cellTotal),
        write('countOut', output.status.count),
        write('overflowOut', output.status.overflow),
        ...(output.status.requiredCount ? [write('totalOut', output.status.requiredCount)] : []),
        ...extraCounts.map((view, index) => write(`extraOut${index}`, view))
      ],
      body: `let total = cellTotal[cellTotalOffset];
  let count = min(total, CAPACITY);
  countOut[countOutOffset] = count;
  overflowOut[overflowOutOffset] = select(0u, 1u, total > CAPACITY);
  ${output.status.requiredCount ? 'totalOut[totalOutOffset] = total;' : ''}
  ${extraCounts.map((_, index) => `extraOut${index}[extraOut${index}Offset] = count;`).join('\n  ')}`
    })
  );
  if (output.status.sumOverflow) {
    const sumOverflowSources = props.sumOverflowSources ?? [];
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-sum-status`,
        operation,
        variant: 'sum-status',
        invocationCount: 1,
        bindings: [
          ...(sumFlag ? [read('sumFlag', sumFlag)] : []),
          ...sumOverflowSources.map((view, index) => read(`sumOverflowIn${index}`, view)),
          write('sumOverflowOut', output.status.sumOverflow)
        ],
        body: `sumOverflowOut[sumOverflowOutOffset] = select(0u, 1u, ${sumFlag ? 'sumFlag[sumFlagOffset] != 0u' : 'false'}${sumOverflowSources.map((_, index) => ` || sumOverflowIn${index}[sumOverflowIn${index}Offset] != 0u`).join('')});`
      })
    );
  }
  if (output.status.candidateOverflow)
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-candidate-status`,
        operation,
        variant: 'candidate-status',
        invocationCount: 1,
        bindings: [
          ...candidateSources.map((view, index) => read(`candidateIn${index}`, view)),
          write('candidateOverflowOut', output.status.candidateOverflow)
        ],
        body: `candidateOverflowOut[candidateOverflowOutOffset] = select(0u, 1u, false${candidateSources.map((_, index) => ` || candidateIn${index}[candidateIn${index}Offset] != 0u`).join('')});`
      })
    );
  return nodes;
}
