// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  getGraphViewChunks,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGPUStorageBindingByteLength,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createCellLevelKernelNode} from './cell-level-kernel';
import {getCellKeyLayout} from './cell-keys';
import {
  getCellTableFirstRow,
  getCellTableViews,
  snapshotCellTable,
  validateCellTable,
  type GPUCellTable
} from './cell-table';
import {
  createCellAggregation,
  validateCellAggregationBindings,
  type GPUCellAggregation,
  type GPUCellAggregationProps
} from './gpu-cell-aggregation';
import {
  validateCellPublishBindingCount,
  validateCellViewBindingRanges,
  validateCellTableScratchBindingSizes,
  validateCellWritableViews,
  validateCellTableWritableViews
} from './cell-validation';
import {createCellRollup} from './gpu-cell-rollup';
import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';

import {
  GPU_CELL_PYRAMID_LIMITS,
  getCellPyramidExecutionPlan,
  CELL_PYRAMID_LEVEL_COUNT_SLOT_STRIDE as LEVEL_COUNT_SLOT_STRIDE,
  type GPUCellPyramidShape,
  type GPUCellPyramidDeviceLimits,
  type GPUCellPyramidExecutionPlan
} from './cell-pyramid-planning';

/** Caller-owned outputs of {@link GPUCellPyramid}. */
export type GPUCellPyramidOutput = {
  /** One table per level, finest first, matching `resolutions`. */
  levels: readonly GPUCellTable[];
  /**
   * Optional packed `uint32` view of `levels.length` rows receiving each level's clamped count,
   * the input of {@link GPUCellLevelSelection}.
   */
  levelCounts?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUCellPyramid}: the row inputs of {@link GPUCellAggregation} plus the
 * levels to build.
 */
export type GPUCellPyramidProps = Omit<
  GPUCellAggregationProps,
  'id' | 'resolution' | 'output' | 'extraCounts'
> & {
  /** Prefix for generated node and transient IDs. Defaults to `'cell-pyramid'`. */
  id?: string;
  /**
   * Resolution of each level, finest first and strictly decreasing. Level 0 aggregates the rows;
   * level `i` rolls up level `i - 1`.
   */
  resolutions: readonly number[];
  /** Caller-owned level tables and counts. */
  output: GPUCellPyramidOutput;
};

/**
 * Builds a zoom pyramid of cell tables: aggregates rows once at the finest level, then rolls each
 * level up into the next coarser one without re-reading the rows. Accepts chunked input rows
 * with matching chunk boundaries, as documented by {@link GPUCellAggregation}; level tables
 * remain single packed outputs.
 *
 * Coarser levels that request sums or extremes need them on every finer level; missing
 * intermediate columns become graph transients. Every level is exact (see
 * {@link GPUCellRollup}). Each level's `overflow` describes only its own capacity;
 * `candidateOverflow` propagates losses from finer levels, making its requirement a lower bound.
 * The input domain is at most MAXIMUM_CELL_PYRAMID_ROW_COUNT = 2147483648 rows.
 * Auto scratch stays packed while rowCount * 4 fits the smaller storage-binding/buffer byte limit,
 * or while rowCount <= an explicit scratchTileSize. Otherwise it uses exact tiled scratch.
 * Each tile is at most MAXIMUM_CELL_PYRAMID_TILE_ROW_COUNT = 1073741824 rows; construction
 * also enforces the aggregation command budget of 32768 nodes. Coarser level tables stay packed.
 * Use limits and getExecutionPlan for the numeric constraints on the actual shape and device.
 */
export class GPUCellPyramid implements GPUCommandNodeProducer {
  /** Numeric route constraints, including the exact automatic scratch crossover. */
  static readonly limits = GPU_CELL_PYRAMID_LIMITS;
  /** Plan packed/tiled scratch and numeric requirements without creating graph resources. */
  static getExecutionPlan(
    shape: GPUCellPyramidShape,
    deviceLimits: GPUCellPyramidDeviceLimits
  ): GPUCellPyramidExecutionPlan {
    return getCellPyramidExecutionPlan(shape, deviceLimits);
  }
  /** Construction-time numeric plan; lowering also validates the actual graph views. */
  readonly executionPlan: GPUCellPyramidExecutionPlan;
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCellPyramidProps;
  /** Resolved spatial context of the input rows, see {@link GPUCellAggregation}. */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUCellPyramidProps) {
    const fields = snapshotConstructionFields(props, [
      'id',
      'family',
      'spatialContext',
      'positions',
      'cells',
      'wordOrder',
      'values',
      'mask',
      'sumScale',
      'scratchMode',
      'scratchTileSize',
      'resolutions',
      'output'
    ]);
    const outputFields = snapshotConstructionFields(fields.output, ['levels', 'levelCounts']);
    props = Object.freeze({
      ...fields,
      resolutions: Object.freeze([...fields.resolutions]),
      output: Object.freeze({
        ...outputFields,
        levels: Object.freeze(outputFields.levels.map(snapshotCellTable))
      }),
      spatialContext: fields.spatialContext
        ? snapshotConstructionSpatialContext(fields.spatialContext)
        : undefined
    });
    this.id = props.id ?? 'cell-pyramid';
    this.props = props;
    const id = this.id;
    const {resolutions, output} = props;
    if (resolutions.length < 1) {
      throw new Error(`${id} needs at least one level`);
    }
    if (output.levels.length !== resolutions.length) {
      throw new Error(`${id} output.levels must hold one table per resolution`);
    }
    for (const [levelIndex, resolution] of resolutions.entries()) {
      getCellKeyLayout(props.family, resolution);
      validateCellTable(id, `output.levels[${levelIndex}]`, output.levels[levelIndex]);
      validateRequirement(
        id,
        levelIndex === 0 || Boolean(output.levels[levelIndex].status.candidateOverflow),
        `output.levels[${levelIndex}].status.candidateOverflow is required to report source loss`
      );
      if (levelIndex > 0 && resolution >= resolutions[levelIndex - 1]) {
        throw new Error(`${id} resolutions must strictly decrease`);
      }
    }
    if (output.levelCounts) {
      validateSinglePackedView(output.levelCounts, ['uint32'], `${id} output.levelCounts`);
      if (output.levelCounts.length !== resolutions.length) {
        throw new Error(`${id} output.levelCounts must hold one row per level`);
      }
    }
    validateRequirement(
      id,
      Boolean(props.values) ||
        !output.levels.some(
          table => table.sums || table.sumValues || table.minimums || table.maximums
        ),
      'values is required for sums and extremes at every output level'
    );
    const outputViews = [...output.levels.flatMap(getCellTableViews), output.levelCounts];
    validateCellWritableViews(
      id,
      outputViews.map((view, index) => [`output field ${index}`, view] as const),
      {separateDispatches: true}
    );
    for (const table of output.levels.slice(1))
      validateCellTableWritableViews(id, table, [], {rollup: true});
    validateGraphOutputsDisjointFromInputs(id, outputViews, [
      props.positions,
      props.cells,
      props.values,
      props.mask
    ]);
    // Validate the finest level eagerly; roll-ups validate when nodes are built.
    const aggregation = this.createAggregation(output.levels[0], []);
    this.spatialContext = aggregation.spatialContext;
    const chunks = getGraphViewChunks((props.positions ?? props.cells)!);
    this.executionPlan = GPUCellPyramid.getExecutionPlan(
      {
        rowCount: (props.positions ?? props.cells)!.length,
        inputChunkRowCounts: chunks.map(chunk => chunk.length),
        levelCapacities: output.levels.map(table => table.cells.length),
        scratchMode: props.scratchMode,
        scratchTileSize: props.scratchTileSize,
        hasValues: Boolean(props.values),
        includeLevelCounts: Boolean(output.levelCounts),
        levelColumns: output.levels.map(table => ({
          sums: Boolean(table.sums),
          sumValues: Boolean(table.sumValues),
          minimums: Boolean(table.minimums),
          maximums: Boolean(table.maximums),
          requiredCount: Boolean(table.status.requiredCount)
        }))
      },
      (chunks[0].buffer.graph as GPUCommandGraph).device.limits
    );
  }

  /** Row offset of each level's `cells` within its buffer, for {@link GPUCellLevelSelection}. */
  get levelFirstRows(): number[] {
    return this.props.output.levels.map(getCellTableFirstRow);
  }

  /**
   * Returns the aggregation nodes of level 0, one roll-up per coarser level and, with
   * `output.levelCounts`, the node that packs the level counts.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {levelCounts} = props.output;
    // Every stage is checked before declaring even the level-count or intermediate-column scratch.
    validateCellAggregationBindings(graph, this.createAggregation(props.output.levels[0], []));
    validateCellViewBindingRanges(graph, id, [
      ...props.output.levels.flatMap(getCellTableViews),
      levelCounts
    ]);
    for (const [levelIndex, table] of props.output.levels.entries()) {
      validateCellPublishBindingCount(
        graph.device,
        id,
        Boolean(table.status.requiredCount),
        0,
        Number(Boolean(levelCounts))
      );
      const sourceRows =
        levelIndex === 0
          ? (props.positions ?? props.cells)!.length
          : props.output.levels[levelIndex - 1].cells.length;
      if (levelIndex > 0)
        validateCellTableScratchBindingSizes(graph.device, id, sourceRows, table.cells.length);
    }
    if (levelCounts) {
      validateCapacity(
        id,
        'level count slots scratch byteLength',
        props.resolutions.length * LEVEL_COUNT_SLOT_STRIDE * 4,
        0,
        graph.device.limits.maxBufferSize
      );
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'level count slots scratch',
        props.resolutions.length * LEVEL_COUNT_SLOT_STRIDE * 4,
        'reduce resolutions length'
      );
    }
    // Rule 12: each level publishes its count into its own 256-byte slot of a scratch view, so the
    // slot binds at element offset 0 and every level shares one publish module (a slice of the
    // packed `levelCounts` would bake a different element offset per level). One copy node then
    // packs the slots into `levelCounts`.
    const levelCountSlots = levelCounts
      ? createTransientView(
          graph,
          `${id}-level-count-slots`,
          'uint32',
          props.resolutions.length * LEVEL_COUNT_SLOT_STRIDE
        )
      : undefined;
    const levelCountViews = props.resolutions.map((_, levelIndex) =>
      levelCountSlots
        ? [
            graph.createDataView(levelCountSlots.buffer, {
              format: 'uint32',
              length: 1,
              byteOffset:
                levelCountSlots.byteOffset +
                levelIndex * LEVEL_COUNT_SLOT_STRIDE * Uint32Array.BYTES_PER_ELEMENT
            })
          ]
        : []
    );
    const tables = this.getLevelTables(graph);
    const nodes: GPUCommandNode<Parameters>[] = [
      ...this.createAggregation(tables[0], levelCountViews[0]).getCommandNodes(graph)
    ];
    for (let levelIndex = 1; levelIndex < tables.length; levelIndex++) {
      nodes.push(
        ...createCellRollup({
          id: `${id}-level-${levelIndex}`,
          family: props.family,
          sourceResolution: props.resolutions[levelIndex - 1],
          resolution: props.resolutions[levelIndex],
          source: tables[levelIndex - 1],
          sumScale: props.sumScale,
          output: tables[levelIndex],
          extraCounts: levelCountViews[levelIndex]
        }).getCommandNodes(graph)
      );
    }
    if (levelCounts && levelCountSlots) {
      const pack = createCellLevelKernelNode<Parameters>(
        graph,
        {
          id: `${id}-level-counts`,
          operation: 'GPUCellPyramid',
          variant: 'level-counts',
          bindings: [
            {name: 'slots', view: levelCountSlots, type: 'u32', access: 'read'},
            {name: 'levelCounts', view: levelCounts, type: 'u32', access: 'read_write'}
          ],
          invocationCount: props.resolutions.length,
          body: `levelCounts[levelCountsOffset + index] = slots[slotsOffset + index * ${LEVEL_COUNT_SLOT_STRIDE}u];`
        },
        {}
      );
      nodes.push(...pack.uploads, pack.node);
    }
    return nodes;
  }

  private createAggregation(
    output: GPUCellTable,
    extraCounts: readonly GraphDataView<'uint32'>[]
  ): GPUCellAggregation {
    const {resolutions, output: _output, ...rowProps} = this.props;
    return createCellAggregation({
      ...rowProps,
      id: `${this.id}-level-0`,
      resolution: resolutions[0],
      output,
      extraCounts
    });
  }

  /** Level tables with graph-transient sums and extremes where a coarser level needs them. */
  private getLevelTables<Parameters>(graph: GPUCommandGraph<Parameters>): GPUCellTable[] {
    const tables = this.props.output.levels.map(table => ({...table}));
    let needsSums = false;
    let needsSumOverflow = false;
    let needsMinimums = false;
    let needsMaximums = false;
    for (let levelIndex = tables.length - 1; levelIndex >= 0; levelIndex--) {
      const table = tables[levelIndex];
      const capacity = table.cells.length;
      const prefix = `${this.id}-level-${levelIndex}`;
      if (needsSumOverflow && !table.status.sumOverflow) {
        table.status = {
          ...table.status,
          sumOverflow: createTransientView(graph, `${prefix}-sum-overflow`, 'uint32', 1)
        };
      }
      if (needsSums && !table.sums) {
        table.sums = createTransientView(graph, `${prefix}-sums`, 'uint32x2', capacity);
      }
      if (needsMinimums && !table.minimums) {
        table.minimums = createTransientView(graph, `${prefix}-minimums`, 'float32', capacity);
      }
      if (needsMaximums && !table.maximums) {
        table.maximums = createTransientView(graph, `${prefix}-maximums`, 'float32', capacity);
      }
      needsSumOverflow ||= Boolean(table.status.sumOverflow);
      needsSums ||= Boolean(table.sums || table.sumValues);
      needsMinimums ||= Boolean(table.minimums);
      needsMaximums ||= Boolean(table.maximums);
    }
    return tables;
  }
}
