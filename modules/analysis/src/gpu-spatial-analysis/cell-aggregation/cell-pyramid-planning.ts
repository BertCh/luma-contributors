// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {
  GPUAnalysisExecutionPlanner,
  GPUAnalysisRoute
} from '../../analysis-core/contracts/analysis-execution-planning';
import {
  GPU_MAXIMUM_COMMAND_NODE_COUNT,
  validateCapacity,
  validatePowerOfTwo,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {CELL_ROW_WORKGROUP_SIZE} from './cell-row-kernel';
import {CELL_LEVEL_PARAMETER_WORD_COUNT} from './cell-level-kernel';
import {
  GRAPH_CHUNK_PARAMETER_SLOT_WORD_COUNT,
  GRAPH_CHUNK_PARAMETER_WORD_COUNT
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {TILED_KEY_UNION_PARAMETER_BYTE_LENGTH} from '../../analysis-core/kernels/tiled-key-union';

/** Published radix sort limit, per packed input domain. */
export const MAXIMUM_CELL_PYRAMID_ROW_COUNT = 0x80000000;
/** Two words per tile row, with uint32 scalar indexing. */
export const MAXIMUM_CELL_PYRAMID_TILE_ROW_COUNT = 0x40000000;
/** Conservative local sort/table command allowance, not a count of submitted GPU dispatches. @internal */
export const CELL_TABLE_COMMAND_NODE_ALLOWANCE = 512;
/** Conservative tile command allowance, including optional reduction columns. @internal */
export const CELL_TILE_COMMAND_NODE_ALLOWANCE = 800;
/** Storage binding alignment in words for each level-count slot. @internal */
export const CELL_PYRAMID_LEVEL_COUNT_SLOT_STRIDE = 64;

/** Requested level columns; finer-level intermediates are inferred as construction does. */
export type GPUCellPyramidLevelColumns = {
  sums?: boolean;
  sumValues?: boolean;
  minimums?: boolean;
  maximums?: boolean;
  requiredCount?: boolean;
};

/** Numeric shape only: no buffers or GPU allocations are needed to plan a pyramid. */
export type GPUCellPyramidShape = {
  rowCount: number;
  /** Ordered packed input chunk lengths; defaults to one chunk. All row columns use this topology. */
  inputChunkRowCounts?: readonly number[];
  /** Capacity of each level, finest first; rollups read the preceding level's capacity. */
  levelCapacities: readonly number[];
  scratchMode?: 'auto' | 'packed';
  scratchTileSize?: number;
  /** Values present in multiple chunks adds one gather per nonempty input chunk. */
  hasValues?: boolean;
  includeLevelCounts?: boolean;
  /** One selection per level. If omitted, planning covers every value column when hasValues is true. */
  levelColumns?: readonly GPUCellPyramidLevelColumns[];
};

export type GPUCellPyramidDeviceLimits = Pick<
  Device['limits'],
  | 'maxStorageBufferBindingSize'
  | 'maxBufferSize'
  | 'maxStorageBuffersPerShaderStage'
  | 'maxUniformBufferBindingSize'
  | 'maxUniformBuffersPerShaderStage'
  | 'maxComputeInvocationsPerWorkgroup'
  | 'maxComputeWorkgroupSizeX'
>;

type NumericConstraint = GPUAnalysisRoute & {satisfied: boolean; received: number};
type BindingRequirement = {name: string; byteLength: number; scope: 'chunk' | 'binding' | 'output'};

export type GPUCellPyramidExecutionPlan = ReturnType<
  GPUAnalysisExecutionPlanner<GPUCellPyramidShape, GPUCellPyramidDeviceLimits>['getExecutionPlan']
> & {
  route: 'packed' | 'tiled';
  scratchTileSize: number;
  tileCount: number;
  /** Logical packed binding sizes. View offsets, aliases and graph ownership are checked at lowering. */
  bindingRequirements: readonly BindingRequirement[];
  /** Maximum requirements across the stages emitted by this shape. */
  stageRequirements: readonly {name: string; storageBindingCount: number; workgroupSize: number}[];
  commands: {
    aggregationUpperBound: number;
    rollupUpperBoundPerLevel: number;
    totalUpperBound: number;
  };
  validAlternative?: {
    route: 'packed' | 'tiled';
    scratchMode: 'auto' | 'packed';
    scratchTileSize: number;
    accuracyImplications: string;
  };
};

const SOURCE =
  'modules/analysis/src/gpu-spatial-analysis/cell-aggregation/cell-pyramid-planning.ts';
const ACCURACY = 'Exact integer cell keys and fixed-point sums on both scratch layouts.';
const PLANNER = {
  package: '@luma-contributors/analysis/gpu-spatial-analysis',
  exportName: 'GPUCellPyramid'
} as const;

/** Machine-readable constraints in the catalog route contract. */
export const GPU_CELL_PYRAMID_LIMITS: readonly GPUAnalysisRoute[] = Object.freeze([
  {
    name: 'input rows',
    kind: 'hard-limit',
    quantity: 'rowCount',
    unit: 'rows',
    appliesWhen: 'all routes',
    maximum: MAXIMUM_CELL_PYRAMID_ROW_COUNT,
    scope: 'domain',
    alternative: 'split the input into independent pyramids',
    accuracyImplications: ACCURACY,
    planner: PLANNER,
    source: SOURCE
  },
  {
    name: 'auto scratch crossover',
    kind: 'auto-crossover',
    quantity: 'rowCount * 4',
    unit: 'bytes',
    appliesWhen: 'scratchMode is auto and scratchTileSize is absent; packed at equality',
    byteExpression: 'rowCount * 4 <= min(maxStorageBufferBindingSize, maxBufferSize)',
    scope: 'binding',
    alternative: 'tiled scratch, or explicitly select a smaller power-of-two scratchTileSize',
    accuracyImplications: ACCURACY,
    planner: PLANNER,
    source: SOURCE
  },
  {
    name: 'tile rows',
    kind: 'hard-limit',
    quantity: 'scratchTileSize',
    unit: 'rows',
    appliesWhen: 'power-of-two scratch width on either route',
    maximum: MAXIMUM_CELL_PYRAMID_TILE_ROW_COUNT,
    scope: 'chunk',
    alternative: 'reduce scratchTileSize',
    accuracyImplications: ACCURACY,
    planner: PLANNER,
    source: SOURCE
  },
  {
    name: 'aggregation commands',
    kind: 'hard-limit',
    quantity: 'estimatedCommandNodeCount',
    unit: 'commands',
    appliesWhen: 'finest-level aggregation',
    maximum: GPU_MAXIMUM_COMMAND_NODE_COUNT,
    scope: 'domain',
    alternative:
      'increase scratchTileSize if its bindings fit, or split the input into independent pyramids',
    accuracyImplications: ACCURACY,
    planner: PLANNER,
    source: SOURCE
  }
]);

/** The exact scratch-selection and command-budget calculation used by aggregation construction. @internal */
export function getCellAggregationScratchPlan(
  shape: {
    rowCount: number;
    scratchMode?: 'auto' | 'packed';
    scratchTileSize?: number;
    chunkCount: number;
    splitValues: boolean;
  },
  limits: Pick<GPUCellPyramidDeviceLimits, 'maxStorageBufferBindingSize' | 'maxBufferSize'>,
  id: string
) {
  validateRequirement(
    id,
    shape.scratchMode === undefined ||
      shape.scratchMode === 'auto' ||
      shape.scratchMode === 'packed',
    "scratchMode must be 'auto' or 'packed'"
  );
  const scratchByteLimit = Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
  const scratchTileSize =
    shape.scratchTileSize ??
    2 **
      Math.floor(
        Math.log2(Math.max(1, Math.min(MAXIMUM_CELL_PYRAMID_TILE_ROW_COUNT, scratchByteLimit / 8)))
      );
  validateCapacity(id, 'scratchTileSize', scratchTileSize, 1, MAXIMUM_CELL_PYRAMID_TILE_ROW_COUNT);
  validatePowerOfTwo(id, 'scratchTileSize', scratchTileSize);
  const usesTiledScratch =
    shape.scratchMode !== 'packed' &&
    (shape.scratchTileSize !== undefined
      ? shape.rowCount > scratchTileSize
      : shape.rowCount * 4 > scratchByteLimit);
  const tileCount = Math.ceil(shape.rowCount / scratchTileSize);
  const estimatedCommandNodeCount = usesTiledScratch
    ? CELL_TILE_COMMAND_NODE_ALLOWANCE * tileCount +
      6 * tileCount * tileCount +
      8 * shape.chunkCount
    : CELL_TABLE_COMMAND_NODE_ALLOWANCE + 1 + shape.chunkCount * (shape.splitValues ? 2 : 1);
  return {scratchTileSize, usesTiledScratch, tileCount, estimatedCommandNodeCount};
}

/**
 * Plans numeric size/device constraints for packed input chunks and packed level tables.
 * It does not inspect view offsets, aliasing, family/resolution validity or requested-column
 * relationships; construction/lowering still validates those declarations. Commands are
 * conservative upper bounds (including optional columns), not an exact dispatch inventory.
 */
export function getCellPyramidExecutionPlan(
  shape: GPUCellPyramidShape,
  deviceLimits: GPUCellPyramidDeviceLimits
): GPUCellPyramidExecutionPlan {
  const id = 'GPUCellPyramid.getExecutionPlan';
  validateCapacity(id, 'rowCount', shape.rowCount, 1, MAXIMUM_CELL_PYRAMID_ROW_COUNT);
  validateCapacity(id, 'levelCapacities.length', shape.levelCapacities.length, 1);
  const chunks = shape.inputChunkRowCounts ?? [shape.rowCount];
  for (const length of chunks)
    validateCapacity(id, 'inputChunkRowCounts entry', length, 0, MAXIMUM_CELL_PYRAMID_ROW_COUNT);
  validateRequirement(
    id,
    chunks.reduce((sum, length) => sum + length, 0) === shape.rowCount,
    `inputChunkRowCounts must sum to rowCount ${shape.rowCount}`
  );
  for (const capacity of shape.levelCapacities)
    validateCapacity(id, 'levelCapacities entry', capacity, 1, 0xffffffff);
  if (shape.levelColumns)
    validateRequirement(
      id,
      shape.levelColumns.length === shape.levelCapacities.length,
      `levelColumns must contain ${shape.levelCapacities.length} selections; received ${shape.levelColumns.length}`
    );
  const columns = shape.levelCapacities.map(
    (_, index) =>
      shape.levelColumns?.[index] ?? {
        sums: Boolean(shape.hasValues),
        sumValues: Boolean(shape.hasValues),
        minimums: Boolean(shape.hasValues),
        maximums: Boolean(shape.hasValues),
        requiredCount: true
      }
  );
  validateRequirement(
    id,
    Boolean(shape.hasValues) ||
      !columns.some(
        column => column.sums || column.sumValues || column.minimums || column.maximums
      ),
    'hasValues must be true when levelColumns request sums or extremes'
  );
  const effectiveColumns = columns.map(column => ({...column}));
  let needsSums = false;
  let needsMinimums = false;
  let needsMaximums = false;
  for (let index = effectiveColumns.length - 1; index >= 0; index--) {
    const column = effectiveColumns[index];
    column.sums ||= needsSums;
    column.minimums ||= needsMinimums;
    column.maximums ||= needsMaximums;
    needsSums ||= Boolean(column.sums || column.sumValues);
    needsMinimums ||= Boolean(column.minimums);
    needsMaximums ||= Boolean(column.maximums);
  }
  const chunkCount = chunks.filter(length => length > 0).length;
  const build = (
    scratchMode: 'auto' | 'packed' | undefined,
    scratchTileSize: number | undefined
  ): GPUCellPyramidExecutionPlan => {
    const scratch = getCellAggregationScratchPlan(
      {
        rowCount: shape.rowCount,
        scratchMode,
        scratchTileSize,
        chunkCount,
        splitValues: Boolean(shape.hasValues && chunks.length > 1)
      },
      deviceLimits,
      id
    );
    const route = scratch.usesTiledScratch ? 'tiled' : 'packed';
    const bindingRequirements: BindingRequirement[] = [];
    const constraints: NumericConstraint[] = [];
    const constrain = (
      name: string,
      received: number,
      maximum: number,
      scope: GPUAnalysisRoute['scope'],
      unit: GPUAnalysisRoute['unit']
    ) => {
      constraints.push({
        name,
        kind: 'hard-limit',
        quantity: name,
        unit,
        appliesWhen: route,
        maximum,
        scope,
        alternative: 'use a valid alternative plan or reduce the named size',
        accuracyImplications: ACCURACY,
        planner: PLANNER,
        source: SOURCE,
        received,
        satisfied: Number.isFinite(maximum) && received <= maximum
      });
    };
    const binding = (
      name: string,
      byteLength: number,
      scope: BindingRequirement['scope'] = 'binding'
    ) => {
      bindingRequirements.push({name, byteLength, scope});
      constrain(
        name,
        byteLength,
        Math.min(deviceLimits.maxStorageBufferBindingSize, deviceLimits.maxBufferSize),
        scope,
        'bytes'
      );
    };
    chunks.forEach((length, index) => {
      if (length) binding(`input chunk ${index}`, length * 8, 'chunk');
    });
    const scratchRows = scratch.usesTiledScratch ? scratch.scratchTileSize : shape.rowCount;
    binding('aggregation row scratch', scratchRows * 4);
    binding('cell level parameters', CELL_LEVEL_PARAMETER_WORD_COUNT * 4);
    if (scratch.usesTiledScratch) {
      binding('tile cell accumulator scratch', scratch.scratchTileSize * 8);
      binding('tile cell offsets scratch', (scratch.scratchTileSize + 1) * 4);
      constrain(
        'padded tile rows',
        scratch.tileCount * scratch.scratchTileSize,
        0xffffffff,
        'domain',
        'rows'
      );
      constrain(
        'tile union storage bindings',
        7,
        deviceLimits.maxStorageBuffersPerShaderStage,
        'dispatch',
        'elements'
      );
      constrain(
        'tile union uniform bytes',
        TILED_KEY_UNION_PARAMETER_BYTE_LENGTH,
        Math.min(deviceLimits.maxUniformBufferBindingSize, deviceLimits.maxBufferSize),
        'binding',
        'bytes'
      );
      constrain(
        'tile union uniform bindings',
        1,
        deviceLimits.maxUniformBuffersPerShaderStage,
        'dispatch',
        'elements'
      );
      constrain(
        'tile union workgroup lanes',
        CELL_ROW_WORKGROUP_SIZE,
        Math.min(
          deviceLimits.maxComputeInvocationsPerWorkgroup,
          deviceLimits.maxComputeWorkgroupSizeX
        ),
        'dispatch',
        'elements'
      );
    } else {
      binding('chunk parameter slot', GRAPH_CHUNK_PARAMETER_WORD_COUNT * 4);
      constrain(
        'chunk parameter allocation',
        chunkCount * GRAPH_CHUNK_PARAMETER_SLOT_WORD_COUNT * 4,
        deviceLimits.maxBufferSize,
        'domain',
        'bytes'
      );
    }
    shape.levelCapacities.forEach((capacity, index) => {
      binding(`level ${index} cell keys`, capacity * 8, 'output');
      binding(`level ${index} cell accumulator scratch`, capacity * 8);
      binding(`level ${index} cell offsets scratch`, (capacity + 1) * 4);
      if (index > 0) binding(`level ${index} row scratch`, shape.levelCapacities[index - 1] * 4);
    });
    if (shape.includeLevelCounts)
      binding(
        'level count slots',
        shape.levelCapacities.length * CELL_PYRAMID_LEVEL_COUNT_SLOT_STRIDE * 4
      );
    // Every table's offsets kernel has six storage columns plus cellParameters.
    // The optional float finish co-binds its inputs and outputs; three requested
    // float columns need eight bindings. Tiled level zero finishes columns separately.
    const stageRequirements = [
      {
        name: 'table offsets and cells',
        storageBindingCount: 7,
        workgroupSize: CELL_ROW_WORKGROUP_SIZE
      }
    ];
    effectiveColumns.forEach((column, index) => {
      const floatCount =
        Number(Boolean(column.sumValues && !(index === 0 && scratch.usesTiledScratch))) +
        Number(Boolean(column.minimums)) +
        Number(Boolean(column.maximums));
      if (floatCount > 0)
        stageRequirements.push({
          name: `level ${index} float finish`,
          storageBindingCount: 2 + 2 * floatCount,
          workgroupSize: CELL_ROW_WORKGROUP_SIZE
        });
      stageRequirements.push({
        name: `level ${index} publish`,
        storageBindingCount:
          4 +
          Number(Boolean(columns[index].requiredCount)) +
          Number(Boolean(shape.includeLevelCounts)),
        workgroupSize: CELL_ROW_WORKGROUP_SIZE
      });
    });
    constrain(
      'stage storage bindings',
      Math.max(...stageRequirements.map(stage => stage.storageBindingCount)),
      deviceLimits.maxStorageBuffersPerShaderStage,
      'dispatch',
      'elements'
    );
    constrain(
      'stage workgroup lanes',
      Math.max(...stageRequirements.map(stage => stage.workgroupSize)),
      Math.min(
        deviceLimits.maxComputeInvocationsPerWorkgroup,
        deviceLimits.maxComputeWorkgroupSizeX
      ),
      'dispatch',
      'elements'
    );
    constrain(
      'aggregation commands',
      scratch.estimatedCommandNodeCount,
      GPU_MAXIMUM_COMMAND_NODE_COUNT,
      'domain',
      'commands'
    );
    return {
      route,
      feasible: constraints.every(constraint => constraint.satisfied),
      constraints,
      accuracyImplications: [ACCURACY],
      scratchTileSize: scratch.scratchTileSize,
      tileCount: scratch.usesTiledScratch ? scratch.tileCount : 1,
      bindingRequirements,
      stageRequirements,
      commands: {
        aggregationUpperBound: scratch.estimatedCommandNodeCount,
        rollupUpperBoundPerLevel: CELL_TABLE_COMMAND_NODE_ALLOWANCE,
        totalUpperBound:
          scratch.estimatedCommandNodeCount +
          (shape.levelCapacities.length - 1) * CELL_TABLE_COMMAND_NODE_ALLOWANCE +
          Number(Boolean(shape.includeLevelCounts))
      }
    };
  };
  const plan = build(shape.scratchMode, shape.scratchTileSize);
  // Derive an independent width from the device, never from a rejected forced
  // request (which may specify width 1). Smaller powers of two cover binding
  // constraints such as the extra cell-offset row without allocating a workload.
  const deviceTileSize = getCellAggregationScratchPlan(
    {
      rowCount: shape.rowCount,
      chunkCount,
      splitValues: Boolean(shape.hasValues && chunks.length > 1)
    },
    deviceLimits,
    id
  ).scratchTileSize;
  const alternatives: {scratchMode: 'auto' | 'packed'; scratchTileSize: number}[] = [];
  if (plan.route === 'tiled')
    alternatives.push({scratchMode: 'packed', scratchTileSize: plan.scratchTileSize});
  for (let width = deviceTileSize; width >= 1; width /= 2)
    alternatives.push({scratchMode: 'auto', scratchTileSize: width});
  for (const candidate of alternatives) {
    const alternative = build(candidate.scratchMode, candidate.scratchTileSize);
    const different =
      alternative.route !== plan.route ||
      (!plan.feasible && candidate.scratchTileSize !== plan.scratchTileSize);
    if (alternative.feasible && different) {
      plan.validAlternative = {
        route: alternative.route,
        ...candidate,
        accuracyImplications: ACCURACY
      };
      break;
    }
  }
  return plan;
}
