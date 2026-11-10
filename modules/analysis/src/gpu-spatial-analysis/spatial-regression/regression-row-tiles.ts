// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandGraphNodeCondition,
  GPUCommandNode,
  GraphBufferUse,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Invocations per workgroup of a row-tile pass (one invocation reduces one tile of rows). */
export const REGRESSION_TILE_WORKGROUP_SIZE = 64;
/** Default tile count bound: keeps the partial buffers and the serial merge loops short. */
const MAXIMUM_DEFAULT_TILE_COUNT = 4096;
/** Default smallest tile: amortizes the per-tile partial write over enough rows. */
const MINIMUM_DEFAULT_TILE_ROWS = 64;

/**
 * Largest row count of the tiled regression passes: counts and row-count divisors are float32 in
 * the kernels, which are exact only up to `2^24`.
 */
export const REGRESSION_TILE_MAXIMUM_ROW_COUNT = 2 ** 24;

/** Rows per tile and tile count of the tiled regression passes. */
export type RegressionTileLayout = {
  /** Rows reduced by one tile invocation. */
  tileRowCount: number;
  /** Number of tiles. */
  tileCount: number;
};

/**
 * Returns the default tile layout of {@link getRowTileSumNodes}: `max(64, ceil(rows / 4096))`
 * rows per tile, the same layout as `GPUOrdinaryLeastSquares`.
 */
export function getRegressionTileLayout(rowCount: number): RegressionTileLayout {
  const tileRowCount = Math.max(
    MINIMUM_DEFAULT_TILE_ROWS,
    Math.ceil(rowCount / MAXIMUM_DEFAULT_TILE_COUNT)
  );
  return {tileRowCount, tileCount: Math.max(1, Math.ceil(rowCount / tileRowCount))};
}

/** GPU-written gate of one node (see `createWGSLDispatchGate`). */
export type RegressionTileGate<Parameters> = {
  condition: GPUCommandGraphNodeCondition<Parameters>;
  extraResources: GraphBufferUse[];
};

/** Properties of {@link getRowTileSumNodes}. */
export type RowTileSumProps<Parameters> = {
  /** Node ID prefix; the nodes are `${id}-tiles` and `${id}-merge`. */
  id: string;
  /** Operation name reported in workload estimates. */
  operation: string;
  /** Workload variant prefix. */
  variant: string;
  /** Number of rows. */
  rowCount: number;
  /** Tile layout from {@link getRegressionTileLayout}. */
  layout: RegressionTileLayout;
  /** Slots `[0, sumCount)` are summed (compensated over tiles). */
  sumCount: number;
  /** Slots `[sumCount, sumCount + maximumCount)` are maxima (of nonnegative terms, start 0). */
  maximumCount?: number;
  /** Read bindings of the tile kernel. */
  bindings: readonly WGSLKernelBinding[];
  /**
   * Module-scope WGSL that defines
   * `fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>)`, which adds the
   * row's sums into the first `SUM_COUNT` slots and folds its maxima into the others (skipped rows
   * add nothing). `TERM_COUNT`, `SUM_COUNT`, `ROW_COUNT` and `TILE_ROWS` are declared by the helper.
   */
  declarations: string;
  /** `tileCount * (sumCount + maximumCount)` float32 partials. */
  partial: GraphDataView<'float32'>;
  /** Receives the totals at `outputSlot`. */
  output: GraphDataView<'float32'>;
  /** First output slot written. */
  outputSlot?: number;
  /** Optional gates of the tile and merge nodes (shapes from {@link getRowTileSumDispatchShapes}). */
  tileGate?: RegressionTileGate<Parameters>;
  mergeGate?: RegressionTileGate<Parameters>;
};

/**
 * Returns the `[invocationCount, workgroupSize]` shapes of the tile and merge nodes of
 * {@link getRowTileSumNodes}, for `createWGSLDispatchGate` slots.
 */
export function getRowTileSumDispatchShapes(
  layout: RegressionTileLayout,
  termCount: number
): [tiles: readonly [number, number], merge: readonly [number]] {
  return [[layout.tileCount, REGRESSION_TILE_WORKGROUP_SIZE], [termCount]];
}

/**
 * Deterministic row reduction in two nodes, the pattern of `GPUOrdinaryLeastSquares`:
 * 1. One invocation per tile of consecutive rows calls `accumulateRow` for each row in row order
 *    (plain float32 within the tile, at most a few hundred rows) and writes its partials.
 * 2. One invocation per term merges the tile partials in tile order: Neumaier-compensated sums
 *    for the sum slots, maxima for the others.
 *
 * The summation order depends only on the layout, so results are bitwise reproducible. Row and
 * tile counts are pipeline overrides, so passes that differ only in row count share a module.
 */
export function getRowTileSumNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: RowTileSumProps<Parameters>
): GPUCommandNode<Parameters>[] {
  const {layout, sumCount} = props;
  const termCount = sumCount + (props.maximumCount ?? 0);
  const termDeclarations = `const TERM_COUNT: u32 = ${termCount}u;
const SUM_COUNT: u32 = ${sumCount}u;`;
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-tiles`,
      operation: props.operation,
      variant: `${props.variant}-tiles`,
      workgroupSize: REGRESSION_TILE_WORKGROUP_SIZE,
      invocationCount: layout.tileCount,
      constants: {ROW_COUNT: props.rowCount, TILE_ROWS: layout.tileRowCount},
      ...(props.tileGate ?? {}),
      bindings: [
        ...props.bindings,
        {name: 'tilePartial', view: props.partial, type: 'f32', access: 'read_write'}
      ],
      declarations: `override ROW_COUNT: u32;
override TILE_ROWS: u32;
${termDeclarations}
${props.declarations}`,
      body: `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var terms: array<f32, ${termCount}>;
  for (var row = firstRow; row < lastRow; row++) {
    accumulateRow(row, &terms);
  }
  for (var term = 0u; term < TERM_COUNT; term++) {
    tilePartial[tilePartialOffset + index * TERM_COUNT + term] = terms[term];
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-merge`,
      operation: props.operation,
      variant: `${props.variant}-merge`,
      invocationCount: termCount,
      constants: {TILE_COUNT: layout.tileCount},
      ...(props.mergeGate ?? {}),
      bindings: [
        {name: 'tilePartial', view: props.partial, type: 'f32', access: 'read'},
        {name: 'tileTotal', view: props.output, type: 'f32', access: 'read_write'}
      ],
      declarations: `override TILE_COUNT: u32;
${termDeclarations}
const OUTPUT_SLOT: u32 = ${props.outputSlot ?? 0}u;`,
      body: `var sum = 0.0;
  var compensation = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let term = tilePartial[tilePartialOffset + tile * TERM_COUNT + index];
    if (index < SUM_COUNT) {
      // Neumaier: exact to about one rounding of the total, whatever the term magnitudes.
      let total = sum + term;
      if (abs(sum) >= abs(term)) {
        compensation = compensation + ((sum - total) + term);
      } else {
        compensation = compensation + ((term - total) + sum);
      }
      sum = total;
    } else {
      sum = max(sum, term);
    }
  }
  tileTotal[tileTotalOffset + OUTPUT_SLOT + index] = sum + compensation;`
    })
  ];
}
