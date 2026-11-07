// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {MAXIMUM_ORIGINAL_SORT_ROWS} from './original-table-sort';

export type TableOperationRoute =
  | {route: 'gpu'; reason: string}
  | {route: 'cpu'; reason: string}
  | {route: 'refused'; reason: string};

/** Arrow type families, as classified by the table provider, that have prepared GPU scalar order. */
export type TableColumnOrderKind =
  | 'float32'
  | 'int32'
  | 'uint32'
  | 'dictionary-string'
  | 'wide-int64'
  | 'wide-uint64'
  | 'float64'
  | 'other-ordered'
  | 'unordered';

/**
 * Sort routing. Float32, Int32 and Uint32 columns that exist as prepared GPU scalars sort on the GPU
 * at any row count. Dictionary strings (rank-remapped codes) and 64-bit/Float64 columns (two hi/lo
 * word passes) sort exactly on the bounded CPU path up to the bound and on the GPU above it. Other
 * ordered types stay CPU-bounded; above the bound they are refused with the reason.
 */
export function chooseTableSortRoute(props: {
  orderKind: TableColumnOrderKind;
  hasPreparedGPUColumn: boolean;
  populationRows: number;
  /** Wide columns need their retained Arrow words to build GPU keys. */
  hasRetainedWideData?: boolean;
}): TableOperationRoute {
  const {orderKind} = props;
  if (orderKind === 'unordered') return {route: 'refused', reason: 'This type has no total order.'};
  if (
    props.hasPreparedGPUColumn &&
    (orderKind === 'float32' || orderKind === 'int32' || orderKind === 'uint32')
  )
    return {route: 'gpu', reason: 'Prepared GPU scalar permutation sort; unbounded row count.'};
  if (props.populationRows <= MAXIMUM_ORIGINAL_SORT_ROWS)
    return {route: 'cpu', reason: 'Exact bounded asynchronous CPU sort of retained Arrow values.'};
  if (orderKind === 'dictionary-string' && props.hasPreparedGPUColumn)
    return {
      route: 'gpu',
      reason:
        'GPU sort of dictionary codes remapped through a sorted-dictionary rank table (code order equals lexical order); unbounded row count.'
    };
  if (
    (orderKind === 'wide-int64' || orderKind === 'wide-uint64' || orderKind === 'float64') &&
    props.hasRetainedWideData !== false
  )
    return {
      route: 'gpu',
      reason:
        orderKind === 'float64'
          ? 'Two-pass stable GPU sort on order-preserving hi/lo words of the Float64 bits (NaN last).'
          : 'Two-pass stable GPU sort on hi/lo 32-bit words of the 64-bit value.'
    };
  return {
    route: 'refused',
    reason: `Exact ordering of this type runs on the CPU for at most ${MAXIMUM_ORIGINAL_SORT_ROWS.toLocaleString()} rows (${props.populationRows.toLocaleString()} requested)${orderKind === 'dictionary-string' ? '; this string field has no prepared dictionary for the GPU rank sort' : orderKind === 'other-ordered' ? '' : '; its retained 64-bit words are unavailable for the GPU sort'}. Narrow the selection or sort a Float32, Int32 or Uint32 field, which sorts on the GPU.`
  };
}

/** Statistics routing: Float32 columns up to 2^24 rows reduce on the GPU, others on the bounded CPU path. */
export function chooseColumnStatisticsRoute(props: {
  gpuSupported: boolean;
  gpuReason?: string;
  populationRows: number;
  /** Dictionary strings, Int64, timestamp and Float64: GPU profile above the exact CPU bound. */
  wideSupported?: boolean;
  wideReason?: string;
}): TableOperationRoute {
  if (props.gpuSupported)
    return {route: 'gpu', reason: 'Fused GPU column profile; only the summary returns.'};
  if (props.populationRows <= MAXIMUM_ORIGINAL_SORT_ROWS)
    return {
      route: 'cpu',
      reason: props.gpuReason ?? 'Bounded CPU statistics over the retained Arrow column.'
    };
  if (props.wideSupported)
    return {route: 'gpu', reason: props.wideReason ?? 'Approximate GPU column profile.'};
  return {
    route: 'refused',
    reason: `${props.wideReason ?? props.gpuReason ?? 'No GPU statistics for this column.'} The CPU path covers at most ${MAXIMUM_ORIGINAL_SORT_ROWS.toLocaleString()} rows (${props.populationRows.toLocaleString()} requested); narrow the selection.`
  };
}
