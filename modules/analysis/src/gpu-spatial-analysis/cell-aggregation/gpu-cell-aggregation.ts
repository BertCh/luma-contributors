// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {
  createGraphChunkDispatchParameters,
  GRAPH_CHUNK_PARAMETER_SLOT_WORD_COUNT,
  planGraphChunkDispatch,
  validateGraphChunkDispatchBindings,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {createCellRowKernelNode, type CellRowKernelBinding} from './cell-row-kernel';
import {dggs} from '@luma.gl/shaders';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';

import {
  getGraphViewChunks,
  validateCapacity,
  validateCommandNodeBudget,
  validateGraphViewTopology,
  validateRequirement,
  validateGraphOutputsDisjointFromInputs,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  CELL_KEY_WGSL,
  getCellKeyLayout,
  QUADBIN_TILE_WGSL,
  resolveCellSpatialContext,
  type CellKeyLayout,
  type GPUCellAggregationFamily
} from './cell-keys';
import {
  getCellTableNodes,
  getCellTableViews,
  hasCellKeyHighWord,
  validateCellTable,
  snapshotCellTable,
  type GPUCellTable,
  type CellTableSumAccumulator
} from './cell-table';
import {
  GPU_CELL_DEFAULT_SUM_SCALE,
  validateSumScale
} from '../../analysis-core/kernels/cell-key-wgsl';
import {
  validateCellPublishBindingCount,
  validateCellViewBindingRanges,
  validateCellTableScratchBindingSizes,
  validateCellTableWritableViews
} from './cell-validation';
import {IS_NAN_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';

import {getTiledCellAggregationNodes} from './cell-tiled-aggregation';
import {
  validateTiledKeyUnionKernelSupport,
  validateTiledKeyUnionRowCount
} from '../../analysis-core/kernels/tiled-key-union';

import {
  getCellAggregationScratchPlan,
  MAXIMUM_CELL_PYRAMID_ROW_COUNT as MAXIMUM_CELL_ROW_COUNT
} from './cell-pyramid-planning';

const OPERATION = 'GPUCellAggregation';

/** Word order of pre-keyed 64-bit cell rows. */
export type GPUCellWordOrder = 'little-endian' | 'high-low';

/**
 * Properties for {@link GPUCellAggregation}.
 *
 * Per-frame (no recompile): the contents of every input buffer, including `mask` and `values`.
 * Topology (needs a new graph): `family`, `resolution`, `sumScale`, view lengths and chunk
 * boundaries, which optional views are present, and the output capacity.
 */
export type GPUCellAggregationProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'cell-aggregation'`. */
  id?: string;
  /** Grid family of the keys. Points (`positions`) are supported for `'quadbin'` only. */
  family: GPUCellAggregationFamily;
  /** Resolution of the output table: Quadbin 0-26, H3 0-15. */
  resolution: number;
  /**
   * Coordinate space of the input rows. `positions` are longitude/latitude degrees, so the default
   * is `{coordinateSpace: 'longitude-latitude', metric: 'none'}` and planar contexts throw. Pre-keyed
   * `cells` default to `{coordinateSpace: 'discrete-grid', metric: 'none', gridFamily: family}`;
   * a given `gridFamily` must equal `family` and a given `gridResolution` (the finest resolution of
   * the input keys) must be at least `resolution`. Nothing is measured, so metrics must be `'none'`.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Longitude/latitude degrees per row (Quadbin only). Longitude is clipped to [-180, 180] and
   * wraps 180 to column 0; latitude is clipped to ±85.051129. Rows with a NaN coordinate are
   * skipped.
   */
  positions?: GraphDataView<'float32x2'> | GraphVectorView<'float32x2'>;
  /**
   * Pre-keyed rows: one 64-bit Quadbin or H3 index per row as two `uint32` words, for example an
   * Arrow `Uint64` column. Rows finer than `resolution` are rolled up to it; invalid rows and rows
   * coarser than `resolution` are skipped.
   */
  cells?: GraphDataView<'uint32x2'> | GraphVectorView<'uint32x2'>;
  /** Word order of `cells`. Defaults to `'little-endian'` (`(low, high)`, Arrow layout). */
  wordOrder?: GPUCellWordOrder;
  /** Optional per-row value. Required for sums and extremes; rows with a non-finite value are skipped. */
  values?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  /** Optional per-row mask; zero skips the row. */
  mask?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /**
   * Fixed-point scale of `output.sums`, a positive float32 value; a power of two keeps the
   * scaling exact. Defaults to 65536 (16 fractional bits): every row is rounded to a multiple of
   * `1 / 65536` before it is added, so values below about 0.01 lose accuracy and values below
   * `2^-17` add 0. `getGPUCellSumScale(maximumMagnitude, maximumRowCount)` picks a scale from the
   * data.
   */
  sumScale?: number;
  /** Caller-owned cell table; its capacity is `output.cells.length`. */
  output: GPUCellTable;
  /** Retains the original one-binding sort as an explicit reference route. */
  scratchMode?: 'auto' | 'packed';
  /** Optional power-of-two tile width; smaller widths can exercise split scratch on small data. */
  scratchTileSize?: number;
};

/**
 * Bins rows into discrete global grid cells and aggregates them per cell into a compact, sorted,
 * capacity-bounded cell table.
 *
 * Keys are 64-bit Quadbin or H3 indexes handled as two `u32` words. Point rows get Quadbin keys
 * in WGSL with integer arithmetic only (exact tile column; fixed-point Mercator row, see
 * `QUADBIN_TILE_WGSL`), so every device and the BigInt CPU reference produce the same key; f32
 * polynomials would not, because compilers may fuse multiply-adds. Pre-keyed rows (Quadbin or H3,
 * for example CARTO tables) are validated (H3 through the gpu-dggs `dggs_h3_is_valid_cell_id`)
 * and truncated to `resolution`.
 *
 * Rows are sorted by key with stable radix sorts (one when the key path fits 31 bits: Quadbin up
 * to resolution 15, H3 up to 8), and cell boundaries come from a scan of segment heads. Each row
 * then adds into its table row with integer atomics: counts, 64-bit fixed-point sums (two words
 * with an explicit carry), and extremes as order-preserving u32 keys, so every output is exact and
 * independent of thread order. The table keeps the smallest keys when it overflows.
 *
 * Accepts chunked input: positions or cells, values and mask must have matching chunk boundaries
 * and packed chunks. Per-chunk key dispatches write global row IDs; chunked values are copied on
 * the GPU to scalar scratch for the existing fixed-point reduction. Row offsets live in
 * per-chunk parameter slots bound at element offset zero, so equal-sized chunks share pipelines. No host repacking is needed.
 * Outputs stay single packed cell tables because table rows aggregate across input chunks. The
 * packed route uses global scalar scratch. Above that binding limit, fixed-width tile sorts
 * and an exact sorted union keep every scratch binding bounded. Tail rows are invalid sentinels.
 * Counts, integer sums and extremes are identical to the packed route, including overflow totals.
 */
export class GPUCellAggregation implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  readonly scratchTileSize: number;
  readonly usesTiledScratch: boolean;
  private readonly chunkPlan: GraphChunkDispatchPlan;
  /** Validated properties. */
  readonly props: GPUCellAggregationProps;
  /** Conservative bound including per-chunk classification and value copies. */
  readonly estimatedCommandNodeCount: number;
  /** Key layout of the output resolution. */
  readonly layout: CellKeyLayout;
  /** Fixed-point scale of the sums. */
  readonly sumScale: number;
  /** Resolved spatial context: longitude/latitude for `positions`, `discrete-grid` for `cells`. */
  readonly spatialContext: GPUSpatialContext;

  private readonly extraCounts: readonly GraphDataView<'uint32'>[];
  private readonly sumAccumulator?: CellTableSumAccumulator;

  constructor(props: GPUCellAggregationProps);
  constructor(
    props: GPUCellAggregationProps,
    extraCounts: readonly GraphDataView<'uint32'>[] = [],
    sumAccumulator?: CellTableSumAccumulator
  ) {
    const fields = snapshotConstructionFields(props, [
      'id',
      'family',
      'resolution',
      'spatialContext',
      'positions',
      'cells',
      'wordOrder',
      'values',
      'mask',
      'sumScale',
      'output',
      'scratchMode',
      'scratchTileSize'
    ]);
    props = Object.freeze({
      ...fields,
      output: snapshotCellTable(fields.output),
      spatialContext: fields.spatialContext
        ? snapshotConstructionSpatialContext(fields.spatialContext)
        : undefined
    });
    this.extraCounts = [...extraCounts];
    this.sumAccumulator = sumAccumulator;
    this.id = props.id ?? 'cell-aggregation';
    this.props = props;
    const id = this.id;
    if (props.family !== 'quadbin' && props.family !== 'h3') {
      throw new Error(`${id} family must be 'quadbin' or 'h3'`);
    }
    this.layout = getCellKeyLayout(props.family, props.resolution);
    this.spatialContext = resolveCellSpatialContext(
      id,
      props.family,
      props.spatialContext,
      props.positions ? 'points' : 'cells',
      props.resolution
    );
    this.sumScale = props.sumScale ?? GPU_CELL_DEFAULT_SUM_SCALE;
    validateSumScale(id, this.sumScale);
    if (Boolean(props.positions) === Boolean(props.cells)) {
      throw new Error(`${id} needs exactly one of positions or cells`);
    }
    if (props.positions && props.family !== 'quadbin') {
      throw new Error(`${id} positions are supported for the quadbin family only`);
    }
    const rowView = (props.positions ?? props.cells)!;
    validateCapacity(id, 'input rows', rowView.length, 1, MAXIMUM_CELL_ROW_COUNT);
    for (const [name, view, formats] of [
      ['positions', props.positions, ['float32x2']],
      ['cells', props.cells, ['uint32x2']],
      ['values', props.values, ['float32']],
      ['mask', props.mask, ['uint32']]
    ] as const) {
      if (view) {
        for (const chunk of getGraphViewChunks(view as GraphDataView | GraphVectorView)) {
          validateSinglePackedView(chunk, formats, `${id} ${name}`);
        }
        validateRequirement(
          id,
          view.length === rowView.length,
          `${name} length must equal the row count`
        );
        validateGraphViewTopology(id, name, rowView, view);
      }
    }
    const chunkPlan = planGraphChunkDispatch(id, {
      rows: rowView,
      values: props.values,
      mask: props.mask
    });
    // Empty row chunks are never bound. Exclude them from the shared parameter planner's
    // binding checks too: an empty tail may start exactly at the end of its allocation.
    this.chunkPlan = {...chunkPlan, chunks: chunkPlan.chunks.filter(chunk => chunk.length > 0)};
    const limits = (getGraphViewChunks(rowView)[0].buffer.graph as GPUCommandGraph).device.limits;
    const scratchPlan = getCellAggregationScratchPlan(
      {
        rowCount: rowView.length,
        scratchMode: props.scratchMode,
        scratchTileSize: props.scratchTileSize,
        chunkCount: this.chunkPlan.chunks.length,
        splitValues: Boolean(props.values && getGraphViewChunks(props.values).length > 1)
      },
      limits,
      id
    );
    this.scratchTileSize = scratchPlan.scratchTileSize;
    this.usesTiledScratch = scratchPlan.usesTiledScratch;
    this.estimatedCommandNodeCount = scratchPlan.estimatedCommandNodeCount;
    if (this.usesTiledScratch) {
      validateTiledKeyUnionRowCount(id, scratchPlan.tileCount * this.scratchTileSize);
    }
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateCellTable(id, 'output', props.output);
    const {output} = props;
    if (!props.values && (output.sums || output.sumValues || output.minimums || output.maximums)) {
      throw new Error(`${id} needs values for sums and extremes`);
    }
    for (const [index, view] of this.extraCounts.entries()) {
      validateSinglePackedView(view, ['uint32'], `${id} extraCounts[${index}]`);
      validateCapacity(id, `extraCounts[${index}] rows`, view.length);
    }
    validateCellTableWritableViews(id, output, this.extraCounts, {
      separateColumns: this.usesTiledScratch
    });
    validateGraphOutputsDisjointFromInputs(
      id,
      [...getCellTableViews(output), ...this.extraCounts],
      [props.positions, props.cells, props.values, props.mask]
    );
  }

  /** Validates all caller views and derived scratch without mutating the graph. @internal */
  private validateBindings<Parameters>(graph: GPUCommandGraph<Parameters>): void {
    const {id, props} = this;
    validateCellPublishBindingCount(
      graph.device,
      id,
      Boolean(props.output.status.requiredCount),
      0,
      this.extraCounts.length
    );
    validateCellViewBindingRanges(graph, id, [
      ...[props.positions, props.cells, props.values, props.mask].flatMap(view =>
        view ? getGraphViewChunks(view).filter(chunk => chunk.length > 0) : []
      ),
      ...getCellTableViews(props.output),
      ...this.extraCounts
    ]);
    const rowView = (props.positions ?? props.cells)!;
    validateCellTableScratchBindingSizes(
      graph.device,
      id,
      this.usesTiledScratch ? this.scratchTileSize : rowView.length,
      props.output.cells.length
    );
    if (this.usesTiledScratch) {
      validateTiledKeyUnionKernelSupport(graph, id, this.extraCounts.length);
      validateCellTableScratchBindingSizes(
        graph.device,
        id,
        this.scratchTileSize,
        this.scratchTileSize
      );
      validateCapacity(
        id,
        'tile chunk parameter buffer byteLength',
        GRAPH_CHUNK_PARAMETER_SLOT_WORD_COUNT * Uint32Array.BYTES_PER_ELEMENT,
        0,
        graph.device.limits.maxBufferSize
      );
    }
    if (!this.usesTiledScratch) validateGraphChunkDispatchBindings(graph, id, this.chunkPlan);
  }

  /** Returns key, sort, boundary, table, reduction, and publish nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    this.validateBindings(graph);
    if (this.usesTiledScratch)
      return getTiledCellAggregationNodes(
        graph,
        this.props,
        this.id,
        this.scratchTileSize,
        this.sumScale,
        this.extraCounts
      );
    const {id, props, layout} = this;
    const rowView = (props.positions ?? props.cells)!;
    const rows = rowView.length;
    const rowChunks = getGraphViewChunks(rowView);
    const twoWords = hasCellKeyHighWord(layout);
    const keyLow = createTransientView(graph, `${id}-key-low`, 'uint32', rows);
    const keyHigh = twoWords
      ? createTransientView(graph, `${id}-key-high`, 'uint32', rows)
      : undefined;
    const rowIds = createTransientView(graph, `${id}-row-ids`, 'uint32', rows);
    const chunkParameters = createGraphChunkDispatchParameters(graph, {
      id: `${id}-chunk-parameters`,
      plan: this.chunkPlan
    });
    const nodes: GPUCommandNode<Parameters>[] = [...chunkParameters.nodes];
    const valueChunks = props.values ? getGraphViewChunks(props.values) : undefined;
    const maskChunks = props.mask ? getGraphViewChunks(props.mask) : undefined;
    const values = valueChunks
      ? valueChunks.length === 1
        ? valueChunks[0]
        : createTransientView(graph, `${id}-values`, 'float32', rows)
      : undefined;
    let parameterIndex = 0;
    for (const [chunkIndex, chunk] of rowChunks.entries()) {
      if (chunk.length === 0) {
        continue;
      }
      const parameters = chunkParameters.slots[parameterIndex++];
      const mask = maskChunks?.[chunkIndex];
      const valueChunk = valueChunks?.[chunkIndex];
      if (values && valueChunk && valueChunks!.length > 1) {
        nodes.push(
          createCellRowKernelNode<Parameters>(graph, {
            id: `${id}-values-chunk-${chunkIndex}`,
            operation: OPERATION,
            variant: 'pack-values',
            bindings: [
              {name: 'source', view: valueChunk, type: 'f32', access: 'read', components: 1},
              {
                name: 'destination',
                view: values,
                type: 'f32',
                access: 'read_write',
                components: 1,
                scope: 'total'
              }
            ],
            length: chunk.length,
            parameters,
            body: 'destination[destinationOffset + chunkParameters[chunkParametersOffset] + index] = source[sourceOffset + index];'
          })
        );
      }
      const bindings: CellRowKernelBinding[] = [
        props.positions
          ? {
              name: 'positions',
              view: chunk,
              components: 2,
              type: 'f32',
              access: 'read'
            }
          : {name: 'cells', view: chunk, type: 'u32', access: 'read', components: 2},
        ...(mask
          ? [
              {
                name: 'rowMask',
                view: mask,
                components: 1,
                type: 'u32',
                access: 'read'
              } as const
            ]
          : []),
        ...(valueChunk
          ? [
              {
                name: 'values',
                view: valueChunk,
                components: 1,
                type: 'f32',
                access: 'read'
              } as const
            ]
          : []),
        {
          name: 'keyLow',
          view: keyLow,
          type: 'u32',
          access: 'read_write',
          components: 1,
          scope: 'total'
        },
        ...(keyHigh
          ? [
              {
                name: 'keyHigh',
                view: keyHigh,
                components: 1,
                scope: 'total',
                type: 'u32',
                access: 'read_write'
              } as const
            ]
          : []),
        {
          name: 'rowIds',
          view: rowIds,
          type: 'u32',
          access: 'read_write',
          components: 1,
          scope: 'total'
        }
      ];
      const invalidKey = `cellShiftLeft(vec2u(0u, 1u), ${layout.width}u)`;
      const keyBody = props.positions
        ? `let longitude = positions[positionsOffset + 2u * index];
  let latitude = positions[positionsOffset + 2u * index + 1u];
  if (isValid && !isNanBits(longitude) && !isNanBits(latitude)) {
    let x = quadbinGetTileX(longitude, RESOLUTION);
    let y = quadbinGetTileY(latitude, RESOLUTION);
    compact = quadbinGetCompactKey(x, y);
  }`
        : `let words = vec2u(cells[cellsOffset + 2u * index], cells[cellsOffset + 2u * index + 1u]);
  let key = ${props.wordOrder === 'high-low' ? 'words' : 'words.yx'};
  ${
    props.family === 'h3'
      ? 'let isValidKey = dggs_h3_is_valid_cell_id(key) && dggs_h3_get_resolution(key) >= RESOLUTION;'
      : 'let isValidKey = cellIsValidQuadbin(key) && ((key.x >> 20u) & 0x1fu) >= RESOLUTION;'
  }
  if (isValid && isValidKey) {
    compact = cellGetCompactKey(key, ${layout.lowBit}u, ${layout.width}u);
  }`;
      nodes.push(
        createCellRowKernelNode<Parameters>(graph, {
          id: rowChunks.length === 1 ? `${id}-keys` : `${id}-keys-chunk-${chunkIndex}`,
          operation: OPERATION,
          variant: props.positions ? 'quadbin-points' : `${props.family}-cells`,
          bindings,
          length: chunk.length,
          parameters,
          declarations: `const RESOLUTION: u32 = ${layout.resolution}u;
${props.cells && props.family === 'h3' ? dggs.source : ''}
${CELL_KEY_WGSL}
${props.positions ? QUADBIN_TILE_WGSL : ''}
${IS_NAN_BITS_WGSL}`,
          body: `let row = chunkParameters[chunkParametersOffset] + index;
  var isValid = true;
  ${mask ? 'isValid = rowMask[rowMaskOffset + index] != 0u;' : ''}
  ${
    valueChunk
      ? `let value = values[valuesOffset + index];
  isValid = isValid && (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u;`
      : ''
  }
  var compact = ${invalidKey};
  ${keyBody}
  keyLow[keyLowOffset + row] = compact.y;
  ${keyHigh ? 'keyHigh[keyHighOffset + row] = compact.x;' : ''}
  rowIds[rowIdsOffset + row] = row;`
        })
      );
    }
    return [
      ...nodes,
      ...getCellTableNodes<Parameters>(graph, {
        id,
        operation: OPERATION,
        layout,
        keyLow,
        keyHigh,
        rowIds,
        sorted: false,
        source: {kind: 'rows', values},
        sumScale: this.sumScale,
        output: props.output,
        extraCounts: this.extraCounts,
        sumAccumulator: this.sumAccumulator
      })
    ];
  }
}

/** Runs the private allocation-free preflight for pyramid composition. @internal */
export function validateCellAggregationBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  contributor: GPUCellAggregation
): void {
  Reflect.apply(Reflect.get(contributor, 'validateBindings'), contributor, [graph]);
}

/** Adds composition-only destinations for a pyramid's level counts. @internal */
export function createCellAggregation(
  props: GPUCellAggregationProps & {
    extraCounts?: readonly GraphDataView<'uint32'>[];
    sumAccumulator?: CellTableSumAccumulator;
  }
): GPUCellAggregation {
  return Reflect.construct(GPUCellAggregation, [
    props,
    props.extraCounts,
    props.sumAccumulator
  ]) as GPUCellAggregation;
}
