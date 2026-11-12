// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUGather,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createStableLexicographicIndexSortNodes} from '../../analysis-core/kernels/stable-lexicographic-index-sort';
import {createSegmentSumNode} from '../../analysis-core/kernels/sorted-segment-sums';
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUCardinalityPort, GPUOverflowStatusPort} from '../../analysis-core/contracts/status';
import {HEALPIX_MAXIMUM_ORDER} from './healpix-index-wgsl';
import type {GPUHealpixIndexScheme} from './gpu-healpix-point-to-cell';

const OPERATION = 'GPUHealpixCellAggregation';

/** Properties for {@link GPUHealpixCellAggregation}. */
export type GPUHealpixCellAggregationProps = {
  /** Prefix for generated node IDs. Defaults to `'healpix-cell-aggregation'`. */
  id?: string;
  /** HEALPix order of the keys, 0 to 29; it sets how many key bits the sort reads. */
  order: number;
  /** Scheme of the keys, as written by `GPUHealpixPointToCell`. Defaults to `'nested'`. */
  scheme?: GPUHealpixIndexScheme;
  /** One 64-bit key per row, little-endian `(low, high)` words (`GPUHealpixPointToCell.output.cells`). */
  cells: GraphDataView<'uint32x2'>;
  /**
   * Optional per-row selection: nonzero keeps the row. Pass `GPUHealpixPointToCell`
   * `output.validity` to leave out rows that received no cell; with the `'nested'` scheme the zero
   * key is a real cell, so rows are otherwise all kept.
   */
  mask?: GraphDataView<'uint32'>;
  /** Optional values to sum and take extremes of. A non-finite value leaves its row out. */
  values?: GraphDataView<'float32'>;
  /** Number of distinct cells the outputs can hold. */
  cellCapacity: number;
  /** Caller-owned outputs. */
  output: GPUHealpixCellAggregationOutput;
};

/** Caller-owned outputs of {@link GPUHealpixCellAggregation}. */
export type GPUHealpixCellAggregationOutput = {
  /** Distinct cell keys in ascending key order (`cellCapacity` entries); unused entries are zero. */
  cells: GraphDataView<'uint32x2'>;
  /** Rows per cell (`cellCapacity` entries); 0 for unused entries. */
  counts: GraphDataView<'uint32'>;
  /** Optional per-cell sum of `values`, in row order within the cell (`cellCapacity` entries). */
  sums?: GraphDataView<'float32'>;
  /** Optional per-cell minimum of `values`. */
  minimums?: GraphDataView<'float32'>;
  /** Optional per-cell maximum of `values`. */
  maximums?: GraphDataView<'float32'>;
  /**
   * Status scalars (one uint32 row each): `count` is the number of cells written (clamped to
   * `cellCapacity`), `requiredCount` the number of distinct cells, `overflow` is 1 when cells were
   * dropped, and `invalidCount` the rows left out by the mask or by a non-finite value.
   */
  status: GPUCardinalityPort & GPUOverflowStatusPort & {invalidCount?: GraphDataView<'uint32'>};
};

/**
 * Aggregates points that `GPUHealpixPointToCell` keyed into HEALPix cells: the distinct cells in
 * key order with their row counts and, for a `values` column, per-cell sums, minima and maxima.
 *
 * The rows are stably sorted by `(left out, high word, low word)` with the library's lexicographic
 * radix sort (only the key bits the order needs), run boundaries are marked and numbered with an
 * inclusive scan, and per-cell reductions are the sorted segmented sums of the other aggregation
 * contributors. There are no atomics: the sums add in row order inside each cell, so the result is
 * bitwise reproducible. Cells keep their nested (or UNIQ) keys, so ascending key order is also
 * the order of the cells along the HEALPix curve.
 *
 * Why not `GPUCellAggregation`: its tables are keyed by the Quadbin and H3 bit layouts (header and
 * resolution bits around the path), which a HEALPix key does not have. This contributor shares
 * the sort, scan and segment-sum primitives and the `GPUPointToCell` key words instead.
 *
 * Limits: the sums are float32 reductions (about 7 digits). Row counts are exact below 2^32.
 */
export class GPUHealpixCellAggregation implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUHealpixCellAggregationProps;

  constructor(props: GPUHealpixCellAggregationProps) {
    const id = props.id ?? 'healpix-cell-aggregation';
    this.id = id;
    this.props = props;
    const {output} = props;
    validateCapacity(id, 'order', props.order, 0, HEALPIX_MAXIMUM_ORDER);
    validateCapacity(id, 'cellCapacity', props.cellCapacity);
    validateSinglePackedView(props.cells, ['uint32x2'], `${id} cells`);
    const rows = props.cells.length;
    validateRequirement(id, rows >= 1, 'needs at least one row');
    for (const [name, view] of [
      ['mask', props.mask],
      ['values', props.values]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, [view.format], `${id} ${name}`);
        validateRequirement(id, view.length === rows, `${name} length must equal the row count`);
      }
    }
    validateSinglePackedView(output.cells, ['uint32x2'], `${id} output.cells`);
    validateSinglePackedView(output.counts, ['uint32'], `${id} output.counts`);
    validateRequirement(
      id,
      output.cells.length === props.cellCapacity && output.counts.length === props.cellCapacity,
      'output.cells and output.counts must have cellCapacity rows'
    );
    for (const name of ['sums', 'minimums', 'maximums'] as const) {
      const view = output[name];
      if (view) {
        validateRequirement(id, Boolean(props.values), `output.${name} needs values`);
        validateSinglePackedView(view, ['float32'], `${id} output.${name}`);
        validateRequirement(
          id,
          view.length === props.cellCapacity,
          `output.${name} must have cellCapacity rows`
        );
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.cells,
        output.counts,
        output.sums,
        output.minimums,
        output.maximums,
        ...Object.values(output.status)
      ],
      [props.cells, props.mask, props.values]
    );
  }

  /** Returns the key split, sort, run numbering, table and reduction nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.cells,
      props.mask,
      props.values,
      output.cells,
      output.counts,
      output.sums,
      output.minimums,
      output.maximums,
      ...Object.values(output.status)
    ]);
    const rows = props.cells.length;
    const capacity = props.cellCapacity;
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'output.cells',
      capacity * 8,
      'reduce cellCapacity'
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'cells',
      rows * 8,
      'aggregate the rows in several calls'
    );
    const keyBits = 2 * props.order + 4 + (props.scheme === 'uniq' ? 1 : 0);
    const highBits = Math.max(1, Math.min(32, keyBits - 32));
    const view = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);
    const lowKeys = view('low-keys', 'uint32', rows);
    const highKeys = view('high-keys', 'uint32', rows);
    const leftOut = view('left-out', 'uint32', rows);
    const rowIds = view('row-ids', 'uint32', rows);
    const starts = view('run-starts-flag', 'uint32', rows);
    const inclusive = view('run-rank', 'uint32', rows);
    const runStart = view('run-start', 'uint32', capacity + 1);
    const totals = view('totals', 'uint32', 2);
    const offsets = view('offsets', 'uint32', capacity + 1);
    const sorted = createStableLexicographicIndexSortNodes<Parameters>(graph, {
      id: `${id}-sort`,
      operation: OPERATION,
      indices: rowIds,
      keys: [
        {view: leftOut, keyBits: 1},
        ...(keyBits > 32 ? [{view: highKeys, keyBits: highBits}] : []),
        {view: lowKeys, keyBits: Math.min(32, keyBits)}
      ]
    });
    const declarations = `const ROWS: u32 = ${rows}u;
const CAPACITY: u32 = ${capacity}u;`;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-split`,
        operation: OPERATION,
        variant: 'split-keys',
        bindings: [
          {name: 'cells', view: props.cells, type: 'u32', access: 'read'},
          ...(props.mask
            ? [{name: 'rowMask', view: props.mask, type: 'u32', access: 'read'} as const]
            : []),
          ...(props.values
            ? [{name: 'values', view: props.values, type: 'f32', access: 'read'} as const]
            : []),
          {name: 'lowKeys', view: lowKeys, type: 'u32', access: 'read_write'},
          {name: 'highKeys', view: highKeys, type: 'u32', access: 'read_write'},
          {name: 'leftOut', view: leftOut, type: 'u32', access: 'read_write'},
          {name: 'rowIds', view: rowIds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations,
        body: `var keep = true;
  ${props.mask ? 'keep = rowMask[rowMaskOffset + index] != 0u;' : ''}
  ${props.values ? 'keep = keep && (bitcast<u32>(values[valuesOffset + index]) & 0x7fffffffu) < 0x7f800000u;' : ''}
  lowKeys[lowKeysOffset + index] = cells[cellsOffset + 2u * index];
  highKeys[highKeysOffset + index] = cells[cellsOffset + 2u * index + 1u];
  leftOut[leftOutOffset + index] = select(1u, 0u, keep);
  rowIds[rowIdsOffset + index] = index;`
      }),
      ...sorted.nodes,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-run-starts`,
        operation: OPERATION,
        variant: 'run-starts',
        bindings: [
          {name: 'sortedIndices', view: sorted.sortedIndices, type: 'u32', access: 'read'},
          {name: 'lowKeys', view: lowKeys, type: 'u32', access: 'read'},
          {name: 'highKeys', view: highKeys, type: 'u32', access: 'read'},
          {name: 'leftOut', view: leftOut, type: 'u32', access: 'read'},
          {name: 'starts', view: starts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations,
        body: `let row = sortedIndices[sortedIndicesOffset + index];
  var isStart = leftOut[leftOutOffset + row] == 0u;
  if (isStart && index > 0u) {
    let previous = sortedIndices[sortedIndicesOffset + index - 1u];
    isStart = leftOut[leftOutOffset + previous] != 0u ||
      lowKeys[lowKeysOffset + previous] != lowKeys[lowKeysOffset + row] ||
      highKeys[highKeysOffset + previous] != highKeys[highKeysOffset + row];
  }
  starts[startsOffset + index] = select(0u, 1u, isStart);`
      }),
      ...new GPUScan({
        id: `${id}-scan`,
        input: starts,
        output: inclusive,
        mode: 'inclusive'
      }).getCommandNodes(graph),
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-totals`,
        operation: OPERATION,
        view: totals,
        type: 'u32',
        value: '0u'
      }),
      // The last kept sorted row publishes the kept row count and the number of distinct cells;
      // every run start records its sorted position (the start of cell `rank - 1`).
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-run-table`,
        operation: OPERATION,
        variant: 'run-table',
        bindings: [
          {name: 'sortedIndices', view: sorted.sortedIndices, type: 'u32', access: 'read'},
          {name: 'leftOut', view: leftOut, type: 'u32', access: 'read'},
          {name: 'starts', view: starts, type: 'u32', access: 'read'},
          {name: 'inclusive', view: inclusive, type: 'u32', access: 'read'},
          {name: 'runStart', view: runStart, type: 'u32', access: 'read_write'},
          {name: 'totals', view: totals, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations,
        body: `let rank = inclusive[inclusiveOffset + index];
  if (starts[startsOffset + index] != 0u && rank - 1u <= CAPACITY) {
    runStart[runStartOffset + rank - 1u] = index;
  }
  let isKept = leftOut[leftOutOffset + sortedIndices[sortedIndicesOffset + index]] == 0u;
  var isLastKept = isKept;
  if (isKept && index + 1u < ROWS) {
    isLastKept = leftOut[leftOutOffset + sortedIndices[sortedIndicesOffset + index + 1u]] != 0u;
  }
  if (isLastKept) {
    totals[totalsOffset] = index + 1u;
    totals[totalsOffset + 1u] = rank;
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-table`,
        operation: OPERATION,
        variant: 'cell-table',
        bindings: [
          {name: 'sortedIndices', view: sorted.sortedIndices, type: 'u32', access: 'read'},
          {name: 'lowKeys', view: lowKeys, type: 'u32', access: 'read'},
          {name: 'highKeys', view: highKeys, type: 'u32', access: 'read'},
          {name: 'runStart', view: runStart, type: 'u32', access: 'read'},
          {name: 'totals', view: totals, type: 'u32', access: 'read'},
          {name: 'offsets', view: offsets, type: 'u32', access: 'read_write'},
          {name: 'cellsOut', view: output.cells, type: 'u32', access: 'read_write'},
          {name: 'counts', view: output.counts, type: 'u32', access: 'read_write'}
        ],
        // One more than the capacity: the last offset closes the final cell.
        invocationCount: capacity + 1,
        declarations,
        body: `let keptRows = totals[totalsOffset];
  let cellTotal = totals[totalsOffset + 1u];
  let isCell = index < cellTotal;
  let begin = select(keptRows, runStart[runStartOffset + index], isCell);
  offsets[offsetsOffset + index] = begin;
  if (index < CAPACITY) {
    var key = vec2u(0u);
    var end = keptRows;
    if (isCell) {
      let row = sortedIndices[sortedIndicesOffset + begin];
      key = vec2u(lowKeys[lowKeysOffset + row], highKeys[highKeysOffset + row]);
      end = select(keptRows, runStart[runStartOffset + index + 1u], index + 1u < cellTotal);
    }
    cellsOut[cellsOutOffset + 2u * index] = key.x;
    cellsOut[cellsOutOffset + 2u * index + 1u] = key.y;
    counts[countsOffset + index] = select(0u, end - begin, isCell);
  }`
      })
    ];
    if (props.values) {
      const gathered = view('sorted-values', 'float32', rows);
      nodes.push(
        // `sortedIndices` is a permutation of the rows, so no index is out of range.
        ...new GPUGather({
          id: `${id}-gather-values`,
          source: props.values,
          indices: sorted.sortedIndices,
          output: gathered
        }).getCommandNodes(graph)
      );
      for (const [name, reduction] of [
        ['sums', 'sum'],
        ['minimums', 'min'],
        ['maximums', 'max']
      ] as const) {
        const target = output[name];
        if (target) {
          nodes.push(
            createSegmentSumNode<Parameters>(graph, {
              id: `${id}-${name}`,
              operation: OPERATION,
              segmentCount: capacity,
              input: gathered,
              segmentOffsets: offsets,
              output: target,
              reduction
            })
          );
        }
      }
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-status`,
        operation: OPERATION,
        variant: 'status',
        bindings: [
          {name: 'totals', view: totals, type: 'u32', access: 'read'},
          ...(output.status.count
            ? [
                {
                  name: 'count',
                  view: output.status.count,
                  type: 'u32',
                  access: 'read_write'
                } as const
              ]
            : []),
          ...(output.status.requiredCount
            ? [
                {
                  name: 'requiredCount',
                  view: output.status.requiredCount,
                  type: 'u32',
                  access: 'read_write'
                } as const
              ]
            : []),
          ...(output.status.overflow
            ? [
                {
                  name: 'overflow',
                  view: output.status.overflow,
                  type: 'u32',
                  access: 'read_write'
                } as const
              ]
            : []),
          ...(output.status.invalidCount
            ? [
                {
                  name: 'invalidCount',
                  view: output.status.invalidCount,
                  type: 'u32',
                  access: 'read_write'
                } as const
              ]
            : [])
        ],
        invocationCount: 1,
        declarations,
        body: `let cellTotal = totals[totalsOffset + 1u];
  ${output.status.count ? 'count[countOffset] = min(cellTotal, CAPACITY);' : ''}
  ${output.status.requiredCount ? 'requiredCount[requiredCountOffset] = cellTotal;' : ''}
  ${output.status.overflow ? 'overflow[overflowOffset] = select(0u, 1u, cellTotal > CAPACITY);' : ''}
  ${output.status.invalidCount ? 'invalidCount[invalidCountOffset] = ROWS - totals[totalsOffset];' : ''}`
      })
    );
    return nodes;
  }
}
