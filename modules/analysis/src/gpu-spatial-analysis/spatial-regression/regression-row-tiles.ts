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
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding,
  type WGSLKernelProps
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateMaximumCount,
  validateGPUStorageBindingByteLength,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getPairwiseSumWGSL} from './pairwise-sum-wgsl';

/** Invocations per workgroup of a row-tile pass (one invocation reduces one tile of rows). */
export const REGRESSION_TILE_WORKGROUP_SIZE = 64;
/** Default tile count target; chunk boundaries and explicit widths may require more tiles. */
const MAXIMUM_DEFAULT_TILE_COUNT = 4096;
/** Default smallest tile: amortizes the per-tile partial write over enough rows. */
export const REGRESSION_TILE_MINIMUM_DEFAULT_ROW_COUNT = 64;

/**
 * Largest row count of the tiled regression passes: counts and row-count divisors are float32 in
 * the kernels, which are exact only up to `2^24`.
 */
export const REGRESSION_TILE_MAXIMUM_ROW_COUNT = 2 ** 24;
/** Consecutive tile partials per merge invocation. The historical one-level order is unchanged. */
export const REGRESSION_TILE_MERGE_GROUP_SIZE = 4096;
/** Two merge levels cover every float32-exact row count, including one row per tile. */
export const REGRESSION_TILE_MAXIMUM_TILE_COUNT = REGRESSION_TILE_MERGE_GROUP_SIZE ** 2;

/** Additional float32 words needed to merge `termCount` channels above the one-level limit. */
export function getRowTileMergeScratchLength(tileCount: number, termCount: number): number {
  return tileCount > REGRESSION_TILE_MERGE_GROUP_SIZE
    ? Math.ceil(tileCount / REGRESSION_TILE_MERGE_GROUP_SIZE) * termCount
    : 0;
}

/** Layout of a partial binding reduced by {@link getRowTileMergeNodes}. @internal */
export type RowTileMergePartial = {
  name: string;
  termCount: number;
  /** Source words contain float32 bits in uint32 storage (integer metadata is reduced separately). */
  bitcastSource?: boolean;
  /** Separate count for chunked partials in a merge that also reads whole-row partials. */
  tileCount?: number;
  /** WGSL count constant of this binding; defaults to TILE_COUNT. */
  countConstant?: string;
  /** Channels stored together for each tile; defaults to all channels (tile-major). */
  blockWidth?: number;
  /** Source words between blocks; defaults to tileCount * blockWidth. */
  blockStride?: number;
  /** Maximum channels and their initial values; all other channels use pairwise sums. */
  maxima?: Readonly<Record<number, number>>;
};

/**
 * Adds a group pass before an existing final merge only above 4096 tiles. Within each consecutive
 * group of at most 4096 tiles, sums use the fixed cascade order of getPairwiseSumWGSL; the final
 * merge uses that same order over group partials. Maxima retain their caller's initial value.
 * The final node keeps its ID and publication logic. Layout blocks are densely packed in scratch.
 * Gated final merges may leave group scratch unused; the group pass writes no published outputs.
 * @internal
 */
export function getRowTileMergeNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  merge: WGSLKernelProps<Parameters>,
  plan: {
    tileCount: number;
    partials: readonly RowTileMergePartial[];
    /** Other final-merge constants whose values become the group stride. */
    strideConstants?: readonly string[];
  },
  createMergeNode: (
    graph: GPUCommandGraph<Parameters>,
    props: WGSLKernelProps<Parameters>
  ) => GPUCommandNode<Parameters> = createWGSLKernelNode
): GPUCommandNode<Parameters>[] {
  validateMaximumCount(merge.id, 'tileCount', plan.tileCount, REGRESSION_TILE_MAXIMUM_TILE_COUNT);
  if (
    plan.tileCount <= REGRESSION_TILE_MERGE_GROUP_SIZE &&
    plan.partials.every(
      partial => (partial.tileCount ?? plan.tileCount) <= REGRESSION_TILE_MERGE_GROUP_SIZE
    )
  ) {
    return [createMergeNode(graph, merge)];
  }
  const nodes: GPUCommandNode<Parameters>[] = [];
  const replacements = new Map<string, GraphDataView>();
  const counts = new Map<string, number>();
  for (const partial of plan.partials) {
    const tileCount = partial.tileCount ?? plan.tileCount;
    validateMaximumCount(merge.id, 'tileCount', tileCount, REGRESSION_TILE_MAXIMUM_TILE_COUNT);
    if (tileCount <= REGRESSION_TILE_MERGE_GROUP_SIZE) continue;
    const groupCount = Math.ceil(tileCount / REGRESSION_TILE_MERGE_GROUP_SIZE);
    counts.set(partial.countConstant ?? 'TILE_COUNT', groupCount);
    const source = merge.bindings.find(binding => binding.name === partial.name)!;
    const length = getRowTileMergeScratchLength(tileCount, partial.termCount);
    validateGPUStorageBindingByteLength(
      graph.device,
      merge.id,
      `${partial.name} merge groups`,
      length * 4,
      'increase tileRowCount'
    );
    validateRequirement(
      merge.id,
      length * 4 <= graph.device.limits.maxBufferSize,
      `${partial.name} merge groups must fit maxBufferSize (${graph.device.limits.maxBufferSize})`
    );
    const groups = createTransientView(
      graph,
      `${merge.id}-${partial.name}-groups`,
      'float32',
      length
    );
    replacements.set(partial.name, groups);
    const width = partial.blockWidth ?? partial.termCount;
    const stride = partial.blockStride ?? tileCount * width;
    const maxima = Object.entries(partial.maxima ?? {});
    const maximumBody = maxima
      .map(
        ([term, initial]) => `if (term == ${term}u) {
    var largest = ${initial === 0 ? '0.0' : String(initial)};
    for (var tile = firstTile; tile < lastTile; tile++) {
      largest = max(largest, source[sourceOffset + channelBlock * ${stride}u + tile * ${width}u + channel]);
    }
    groups[groupsOffset + destination] = largest;
    return;
  }`
      )
      .join('\n');
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${merge.id}-${partial.name}-groups`,
        operation: merge.operation,
        variant: `${merge.variant ?? 'merge'}-groups`,
        invocationCount: groupCount * partial.termCount,
        workgroupSize: REGRESSION_TILE_WORKGROUP_SIZE,
        bindings: [
          {...source, name: 'source', access: 'read'},
          {name: 'groups', view: groups, type: 'f32', access: 'read_write'}
        ],
        body: `let group = index / ${partial.termCount}u;
  let term = index % ${partial.termCount}u;
  let channelBlock = term / ${width}u;
  let channel = term % ${width}u;
  let firstTile = group * ${REGRESSION_TILE_MERGE_GROUP_SIZE}u;
  let lastTile = min(firstTile + ${REGRESSION_TILE_MERGE_GROUP_SIZE}u, ${tileCount}u);
  let destination = channelBlock * ${groupCount * width}u + group * ${width}u + channel;
  ${maximumBody}
  ${getPairwiseSumWGSL({count: 'lastTile - firstTile', term: partial.bitcastSource ? `bitcast<f32>(source[sourceOffset + channelBlock * ${stride}u + (firstTile + sumItem) * ${width}u + channel])` : `source[sourceOffset + channelBlock * ${stride}u + (firstTile + sumItem) * ${width}u + channel]`})}
  groups[groupsOffset + destination] = sum;`
      })
    );
  }
  let declarations = merge.declarations ?? '';
  for (const name of plan.strideConstants ?? []) {
    counts.set(name, Math.ceil(plan.tileCount / REGRESSION_TILE_MERGE_GROUP_SIZE));
  }
  for (const [name, count] of counts) {
    declarations = declarations.replace(
      new RegExp(`const ${name}: u32 = [0-9]+u;`, 'g'),
      `const ${name}: u32 = ${count}u;`
    );
  }
  nodes.push(
    createMergeNode(graph, {
      ...merge,
      declarations,
      ...(merge.constants
        ? {
            constants: Object.fromEntries(
              Object.entries(merge.constants).map(([name, value]) => [
                name,
                counts.get(name) ?? value
              ])
            )
          }
        : {}),
      bindings: merge.bindings.map(binding => ({
        ...binding,
        view: replacements.get(binding.name) ?? binding.view
      }))
    })
  );
  return nodes;
}

/** Rows per tile and tile count of the tiled regression passes. */
export type RegressionTileLayout = {
  /** Rows reduced by one tile invocation. */
  tileRowCount: number;
  /** Number of tiles. */
  tileCount: number;
};

/**
 * Returns the default tile layout of {@link getRowTileSumNodes}: `max(64, ceil(rows / 4096))`
 * rows per tile, the same layout as `GPUOrdinaryLeastSquares`. An explicit width is preserved.
 * This counts global rows; CSR callers must separately count tiles within each chunk.
 */
export function getRegressionTileLayout(
  rowCount: number,
  explicitTileRowCount?: number
): RegressionTileLayout {
  const tileRowCount =
    explicitTileRowCount ??
    Math.max(
      REGRESSION_TILE_MINIMUM_DEFAULT_ROW_COUNT,
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
  /** Slots `[0, sumCount)` are summed (plain within a tile, pairwise over the tiles). */
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
 * Deterministic row reduction, retaining the historical two nodes through 4096 tiles:
 * 1. One invocation per tile of consecutive rows calls `accumulateRow` for each row in row order
 *    (plain float32 within the tile, at most a few hundred rows) and writes its partials.
 * 2. One invocation per term merges the tile partials: a fixed pairwise (cascade) sum for the
 *    sum slots, whose error is about `log2(tileCount)` roundings of the partial totals, and
 *    maxima for the others. This replaces a Neumaier-compensated merge, which the shader compiler
 *    of the test GPU folded to a plain serial sum (see `getPairwiseSumWGSL`).
 *    Above 4096 tiles an additional pass first merges consecutive groups of at most 4096 tiles
 *    in that same pairwise order; the final pass merges the group partials in group order.
 *    Two levels cover REGRESSION_TILE_MAXIMUM_TILE_COUNT (2^24) tiles; each extra partial binding
 *    has ceil(tileCount / 4096) * termCount float32 words, checked against device limits.
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
    ...getRowTileMergeNodes<Parameters>(
      graph,
      {
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
        // Sum slots: fixed pairwise order over the tiles (see getPairwiseSumWGSL for why this is
        // not a compensated sum). Maximum slots: a running maximum.
        body: `if (index < SUM_COUNT) {
    ${getPairwiseSumWGSL({count: 'TILE_COUNT', term: 'tilePartial[tilePartialOffset + sumItem * TERM_COUNT + index]'})}
    tileTotal[tileTotalOffset + OUTPUT_SLOT + index] = sum;
    return;
  }
  var largest = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    largest = max(largest, tilePartial[tilePartialOffset + tile * TERM_COUNT + index]);
  }
  tileTotal[tileTotalOffset + OUTPUT_SLOT + index] = largest;`
      },
      {
        tileCount: layout.tileCount,
        partials: [
          {
            name: 'tilePartial',
            termCount,
            maxima: Object.fromEntries(
              Array.from({length: props.maximumCount ?? 0}, (_, index) => [sumCount + index, 0])
            )
          }
        ]
      }
    )
  ];
}
