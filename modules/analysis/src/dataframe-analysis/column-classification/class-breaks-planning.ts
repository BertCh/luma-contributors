// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {
  GPUAnalysisExecutionPlanner,
  GPUAnalysisRoute
} from '../../analysis-core/contracts/analysis-execution-planning';
import type {GPUClassBreaksMethod} from './class-breaks-parameters';
import {GPU_MAXIMUM_COMMAND_NODE_COUNT} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  DEFAULT_NATURAL_BREAKS_BIN_COUNT,
  validateClassBreaksGraphOptions,
  validateClassBreaksInputAssertions,
  resolveClassBreaksMethods
} from './class-breaks-input';
import type {GPUClassBreaksProps} from './gpu-class-breaks';
import {
  CLASS_BREAKS_STATE_LENGTH,
  getClassBreaksMomentTileCount,
  MOMENT_STRIDE,
  NATURAL_BREAKS_CELL_COUNT,
  NATURAL_BREAKS_ELEMENT_STRIDE
} from './class-breaks-kernels';
import {getNaturalBreaksDispatchWordCount} from './natural-breaks-nodes';
import {
  NATURAL_BREAKS_MAXIMUM_EXACT_ROWS,
  NATURAL_BREAKS_MAXIMUM_MATERIALIZED_ROWS,
  NATURAL_BREAKS_MAXIMUM_MONOTONE_ROWS
} from './natural-breaks-limits';

export type GPUClassBreaksShape = Pick<
  GPUClassBreaksProps,
  | 'maximumClassCount'
  | 'methods'
  | 'naturalBreaksMode'
  | 'naturalBreaksBinCount'
  | 'naturalBreaksExactStorage'
> & {
  /** Total live rows, never the backing allocation capacity. */
  rowCount: number;
  /** Presence means a vector, even for zero or one chunk; lengths must sum to rowCount. */
  chunkRowCounts?: readonly number[];
  valuesFormat?: 'float32' | 'uint32' | 'sint32';
  mask?: boolean;
  /**
   * Optional physical overrides merged into inferred bindings: values.0/mask.0 (chunk indices),
   * parameters and output.breaks/classCount/classCounts. Byte lengths include alignment prefixes;
   * backing allocations need capacity, not four-byte length alignment. Required bindings remain
   * checked when only part of the inventory is supplied; overrides cannot reduce required rows.
   */
  bindings?: readonly {name: string; byteLength: number; bufferByteLength: number}[];
};
export type GPUClassBreaksDeviceLimits = Pick<
  Device['limits'],
  | 'maxBufferSize'
  | 'maxStorageBufferBindingSize'
  | 'maxStorageBuffersPerShaderStage'
  | 'maxUniformBufferBindingSize'
  | 'maxUniformBuffersPerShaderStage'
  | 'maxComputeInvocationsPerWorkgroup'
  | 'maxComputeWorkgroupSizeX'
  | 'maxComputeWorkgroupsPerDimension'
>;

const SOURCE = 'modules/analysis/src/dataframe-analysis/column-classification/gpu-class-breaks.ts';
const NATURAL_BREAKS_ACCURACY =
  'Exact exhaustively optimizes the normalized float32 recurrence; CPU exact uses float64. Approximate histogram/refinement and quantized monotone pruning can change boundaries.';
const createRouteConstraint = (
  name: string,
  maximum: number,
  appliesWhen: string,
  alternative?: string
): GPUAnalysisRoute =>
  Object.freeze({
    name,
    kind: 'hard-limit',
    quantity: 'values rows',
    unit: 'rows',
    maximum,
    appliesWhen,
    scope: 'domain',
    alternative,
    accuracyImplications: NATURAL_BREAKS_ACCURACY,
    source: SOURCE
  });

/** Conditional route limits, in addition to device-dependent binding/allocation constraints. */
export const CLASS_BREAKS_LIMITS: readonly GPUAnalysisRoute[] = Object.freeze([
  createRouteConstraint('rows', 2 ** 24, 'all methods; total rows over all chunks'),
  createRouteConstraint(
    'materialized-rows',
    NATURAL_BREAKS_MAXIMUM_MATERIALIZED_ROWS,
    'natural-breaks enabled, exact mode, explicit materialized storage',
    'exact recompute'
  ),
  createRouteConstraint(
    'exact-rows',
    NATURAL_BREAKS_MAXIMUM_EXACT_ROWS,
    'natural-breaks enabled, exact mode, auto or recompute storage',
    'approximate or monotone (different mathematical guarantee)'
  ),
  createRouteConstraint(
    'monotone-rows',
    NATURAL_BREAKS_MAXIMUM_MONOTONE_ROWS,
    'natural-breaks enabled, monotone mode',
    'histogram/refinement approximation'
  ),
  Object.freeze({
    name: 'storage-bindings',
    kind: 'allocation',
    quantity: 'each aligned input/output or scratch binding',
    unit: 'bytes',
    byteExpression: 'byteLength <= min(device.maxBufferSize, device.maxStorageBufferBindingSize)',
    appliesWhen:
      'all methods; chunked inputs per chunk, materialized moment/conversion scratch over total rows',
    scope: 'binding',
    accuracyImplications:
      'Smaller chunks do not reduce whole-column scratch. Omit moment/sort methods to avoid their materialization.',
    source: SOURCE
  }),
  Object.freeze({
    name: 'commands',
    kind: 'hard-limit',
    quantity: 'estimated command nodes',
    unit: 'commands',
    maximum: GPU_MAXIMUM_COMMAND_NODE_COUNT,
    appliesWhen: 'all compiled methods, including inactive per-frame methods',
    scope: 'domain',
    alternative: 'omit unused methods or reduce nonempty chunk count',
    accuracyImplications: 'Omitting a method makes it unavailable to the per-frame parameters.',
    source: SOURCE
  })
]);

/** Shared by the planner and graph preflight, before any scratch is declared. @internal */
export function getClassBreaksScratchWordCounts(
  shape: GPUClassBreaksShape,
  methods: readonly string[]
) {
  const count = shape.rowCount;
  const classCapacity = shape.maximumClassCount;
  const quantileCount = Math.max(classCapacity - 1, methods.includes('box-plot') ? 3 : 1);
  const binCapacity = shape.naturalBreaksBinCount ?? DEFAULT_NATURAL_BREAKS_BIN_COUNT;
  const chunkCount = shape.chunkRowCounts?.length ?? 1;
  const needsMoments = methods.includes('standard-deviation') || methods.includes('head-tail');
  const usesQuantile = methods.includes('quantile') || methods.includes('box-plot');
  const usesNatural = methods.includes('natural-breaks');
  const exact = usesNatural && shape.naturalBreaksMode === 'exact';
  const approximate = usesNatural && (shape.naturalBreaksMode ?? 'approximate') === 'approximate';
  const monotone = usesNatural && shape.naturalBreaksMode === 'monotone';
  const materialized =
    exact &&
    (shape.naturalBreaksExactStorage === 'materialized' ||
      (shape.naturalBreaksExactStorage !== 'recompute' &&
        count <= NATURAL_BREAKS_MAXIMUM_MATERIALIZED_ROWS));
  return {
    state: CLASS_BREAKS_STATE_LENGTH,
    'packed value conversion/moment input':
      needsMoments || (shape.valuesFormat ?? 'float32') !== 'float32' ? count : 0,
    'packed mask conversion/moment input':
      shape.mask && (needsMoments || (shape.valuesFormat ?? 'float32') !== 'float32') ? count : 0,
    'natural refinement dispatch': approximate ? getNaturalBreaksDispatchWordCount(chunkCount) : 0,
    extremes: 4,
    edges: classCapacity + 1,
    'natural coordinate scale': approximate ? classCapacity + 1 : 0,
    'quantile histogram': usesQuantile ? 2 * quantileCount * 256 : 0,
    'quantile state': usesQuantile ? 8 * quantileCount + 2 : 0,
    'quantile parameters': usesQuantile ? 4 + quantileCount : 0,
    'head-tail state': methods.includes('head-tail') ? 4 + Math.max(classCapacity - 1, 1) : 0,
    moments: needsMoments ? 2 * MOMENT_STRIDE : 0,
    'head-tail dispatch': methods.includes('head-tail') ? 6 : 0,
    'moment partials': needsMoments ? getClassBreaksMomentTileCount(count) * MOMENT_STRIDE : 0,
    'natural histogram': approximate ? 5 * (binCapacity + 1) : 0,
    'natural DP': approximate ? classCapacity * binCapacity : 0,
    'natural refinement plan': approximate ? 4 * (classCapacity + 1) : 0,
    'natural refinement cells': approximate
      ? Math.max(classCapacity - 1, 1) *
        (NATURAL_BREAKS_CELL_COUNT + 2) *
        NATURAL_BREAKS_ELEMENT_STRIDE
      : 0,
    'sorted row scratch': methods.includes('maximum-breaks') || exact || monotone ? count : 0,
    'sorted moments/prefix': monotone ? count * 5 : 0,
    'exact partitions': exact ? (classCapacity + 1) * (count + 1) : 0,
    'exact scores': exact ? (materialized ? classCapacity + 1 : 2) * (count + 1) : 0,
    'exact costs': materialized ? count * count : 0,
    'monotone partitions': monotone ? (classCapacity + 1) * (count + 1) : 0,
    'monotone scores': monotone ? 2 * (count + 1) : 0,
    'monotone tasks': monotone ? count * 8 : 0,
    'monotone task dispatch': monotone ? 3 : 0
  };
}

/** One evaluated route constraint of the class-breaks plan. */
export type ClassBreaksRouteConstraint = GPUAnalysisRoute & {satisfied: boolean; received: number};

/** Route admission only; callers must also supply valid, disjoint views belonging to their graph. */
export function getClassBreaksExecutionPlan(
  shape: GPUClassBreaksShape,
  deviceLimits?: GPUClassBreaksDeviceLimits
) {
  const constraints: ClassBreaksRouteConstraint[] = [];
  const addConstraint = (
    descriptor: GPUAnalysisRoute,
    received: number,
    satisfied = Number.isFinite(received) &&
      received >= (descriptor.minimum ?? 0) &&
      received <= (descriptor.maximum ?? Infinity)
  ) => constraints.push({...descriptor, received, satisfied});
  let methods: readonly GPUClassBreaksMethod[] = [];
  try {
    validateClassBreaksInputAssertions('GPUClassBreaks.getExecutionPlan', shape);
    validateClassBreaksGraphOptions('GPUClassBreaks.getExecutionPlan', shape);
    methods = resolveClassBreaksMethods(
      'GPUClassBreaks.getExecutionPlan',
      shape,
      shape.chunkRowCounts !== undefined
    );
  } catch (error) {
    addConstraint(
      {
        ...createRouteConstraint('options', 0, String(error)),
        quantity: 'invalid option configuration'
      },
      1,
      false
    );
  }
  addConstraint(
    {
      ...createRouteConstraint('class-capacity', 64, 'all GPU class-breaks methods'),
      quantity: 'maximumClassCount',
      unit: 'elements',
      minimum: 1
    },
    shape.maximumClassCount,
    Number.isInteger(shape.maximumClassCount) &&
      shape.maximumClassCount >= 1 &&
      shape.maximumClassCount <= 64
  );
  const chunks = shape.chunkRowCounts ?? [shape.rowCount];
  addConstraint(
    {...CLASS_BREAKS_LIMITS[0], minimum: 1},
    shape.rowCount,
    Number.isInteger(shape.rowCount) && shape.rowCount >= 1 && shape.rowCount <= 2 ** 24
  );
  addConstraint(
    {
      ...createRouteConstraint(
        'chunk-topology',
        shape.rowCount,
        'chunk lengths are nonnegative integers and sum to rowCount'
      ),
      quantity: 'chunk rows'
    },
    chunks.reduce((sum, count) => sum + count, 0),
    chunks.every(count => Number.isInteger(count) && count >= 0) &&
      chunks.reduce((sum, count) => sum + count, 0) === shape.rowCount
  );
  const usesNatural = methods.includes('natural-breaks');
  const mode = shape.naturalBreaksMode ?? 'approximate';
  const storage =
    mode === 'exact' && usesNatural
      ? shape.naturalBreaksExactStorage === 'materialized' ||
        (shape.naturalBreaksExactStorage !== 'recompute' &&
          shape.rowCount <= NATURAL_BREAKS_MAXIMUM_MATERIALIZED_ROWS)
        ? 'materialized'
        : 'recompute'
      : undefined;
  if (usesNatural && mode !== 'approximate')
    addConstraint(
      CLASS_BREAKS_LIMITS[
        mode === 'monotone' ? 3 : shape.naturalBreaksExactStorage === 'materialized' ? 1 : 2
      ],
      shape.rowCount
    );
  const chunkCount = chunks.filter(count => count > 0).length;
  const estimatedCommandNodeCount =
    32 +
    8 * chunkCount +
    4 * shape.maximumClassCount +
    (usesNatural
      ? 128 * shape.maximumClassCount + 38 * chunkCount + (mode === 'approximate' ? 3 : 0)
      : 0) +
    (methods.includes('maximum-breaks') ? 2049 : 0) +
    (usesNatural && mode !== 'approximate' ? 1024 + 4 * shape.maximumClassCount : 0);
  addConstraint(CLASS_BREAKS_LIMITS[5], estimatedCommandNodeCount);
  const scratchWordCounts = getClassBreaksScratchWordCounts(shape, methods);
  if (deviceLimits) {
    const bindingMaximum = Math.min(
      deviceLimits.maxBufferSize,
      deviceLimits.maxStorageBufferBindingSize
    );
    const addBindingConstraint = (name: string, bytes: number, maximum = bindingMaximum) =>
      addConstraint(
        {...CLASS_BREAKS_LIMITS[4], name, maximum},
        bytes,
        Number.isInteger(bytes) && bytes >= 4 && bytes % 4 === 0 && bytes <= maximum
      );
    for (const [name, words] of Object.entries(scratchWordCounts))
      addBindingConstraint(`scratch.${name}`, Math.max(words, 1) * 4);
    const inferredBindings = [
      ...chunks.map((count, index) => ({
        name: `values.${index}`,
        byteLength: Math.max(count, 1) * 4,
        bufferByteLength: Math.max(count, 1) * 4
      })),
      ...(shape.mask
        ? chunks.map((count, index) => ({
            name: `mask.${index}`,
            byteLength: Math.max(count, 1) * 4,
            bufferByteLength: Math.max(count, 1) * 4
          }))
        : []),
      {
        name: 'parameters',
        byteLength: (shape.maximumClassCount + 9) * 4,
        bufferByteLength: (shape.maximumClassCount + 9) * 4
      },
      {
        name: 'output.breaks',
        byteLength: (shape.maximumClassCount + 1) * 4,
        bufferByteLength: (shape.maximumClassCount + 1) * 4
      },
      {name: 'output.classCount', byteLength: 4, bufferByteLength: 4},
      {
        name: 'output.classCounts',
        byteLength: shape.maximumClassCount * 4,
        bufferByteLength: shape.maximumClassCount * 4
      }
    ];
    const bindings = new Map(inferredBindings.map(view => [view.name, view]));
    for (const override of shape.bindings ?? []) {
      const name =
        override.name === 'values' || override.name === 'mask'
          ? `${override.name}.0`
          : ['breaks', 'classCount', 'classCounts'].includes(override.name)
            ? `output.${override.name}`
            : override.name;
      const inferred = bindings.get(name);
      if (inferred)
        addConstraint(
          {...CLASS_BREAKS_LIMITS[4], name: `${name}.capacity`, minimum: inferred.byteLength},
          override.byteLength,
          Number.isInteger(override.byteLength) && override.byteLength >= inferred.byteLength
        );
      bindings.set(name, {...override, name});
    }
    for (const view of bindings.values()) {
      addBindingConstraint(view.name, view.byteLength);
      // The binding range is aligned, but its backing allocation may have any byte length.
      addConstraint(
        {
          ...CLASS_BREAKS_LIMITS[4],
          name: `${view.name}.buffer`,
          minimum: view.byteLength,
          maximum: deviceLimits.maxBufferSize
        },
        view.bufferByteLength,
        Number.isInteger(view.bufferByteLength) &&
          view.bufferByteLength >= view.byteLength &&
          view.bufferByteLength <= deviceLimits.maxBufferSize
      );
    }
    const dispatchWorkgroups: Record<string, number> = {};
    for (const [index, rows] of chunks.entries())
      dispatchWorkgroups[`rows.${index}`] = Math.max(1, Math.ceil(rows / 256));
    if (methods.includes('standard-deviation') || methods.includes('head-tail'))
      dispatchWorkgroups['moment-tiles'] = getClassBreaksMomentTileCount(shape.rowCount);
    if (methods.includes('quantile') || methods.includes('box-plot'))
      dispatchWorkgroups['quantile-histogram-clear'] =
        2 * Math.max(shape.maximumClassCount - 1, methods.includes('box-plot') ? 3 : 1);
    if (usesNatural && mode === 'approximate') {
      dispatchWorkgroups['natural-histogram-clear'] = Math.ceil(
        (5 * (shape.naturalBreaksBinCount ?? DEFAULT_NATURAL_BREAKS_BIN_COUNT)) / 256
      );
      dispatchWorkgroups['natural-refinement-clear'] = Math.ceil(
        scratchWordCounts['natural refinement cells'] / 256
      );
    }
    if (usesNatural && mode === 'exact') {
      dispatchWorkgroups['exact-initialize'] = Math.ceil(
        ((shape.maximumClassCount + 1) * (shape.rowCount + 1)) / 256
      );
      dispatchWorkgroups['exact-layer'] = Math.ceil(shape.rowCount / 64);
    }
    if (usesNatural && mode === 'monotone') dispatchWorkgroups['monotone-search'] = shape.rowCount;
    const dimensionMaximum = deviceLimits.maxComputeWorkgroupsPerDimension;
    for (const [name, received] of Object.entries(dispatchWorkgroups))
      addConstraint(
        {
          name: `dispatch.${name}`,
          kind: 'hard-limit',
          quantity: 'workgroups',
          unit: 'elements',
          maximum: dimensionMaximum ** 3,
          appliesWhen: 'enabled kernel route; bounded three-dimensional dispatch',
          scope: 'dispatch',
          alternative: 'reduce rows per chunk, total rows or omit this method',
          accuracyImplications: 'Changing the method may change the mathematical guarantee.',
          source: SOURCE
        },
        received,
        Number.isSafeInteger(dimensionMaximum) &&
          dimensionMaximum >= 1 &&
          Number.isFinite(received) &&
          received <= dimensionMaximum ** 3
      );
    for (const [name, received, maximum] of [
      ['storage bindings', 8, deviceLimits.maxStorageBuffersPerShaderStage],
      ['uniform bytes', 256, deviceLimits.maxUniformBufferBindingSize],
      ['uniform buffer bytes', 256, deviceLimits.maxBufferSize],
      ['uniform bindings', 1, deviceLimits.maxUniformBuffersPerShaderStage],
      [
        'workgroup size',
        256,
        Math.min(
          deviceLimits.maxComputeInvocationsPerWorkgroup,
          deviceLimits.maxComputeWorkgroupSizeX
        )
      ]
    ] as const)
      addConstraint(
        {
          name,
          kind: 'hard-limit',
          quantity: name,
          unit: 'elements',
          maximum,
          appliesWhen: 'all classification kernels',
          scope: 'dispatch',
          accuracyImplications: '',
          source: SOURCE
        },
        received
      );
  }
  return Object.freeze({
    route: usesNatural ? `${mode}${storage ? `-${storage}` : ''}` : 'non-natural',
    feasible: constraints.every(constraint => constraint.satisfied),
    constraints: Object.freeze(constraints.map(constraint => Object.freeze(constraint))),
    accuracyImplications: Object.freeze([NATURAL_BREAKS_ACCURACY]),
    methods,
    naturalBreaksMode: mode,
    naturalBreaksExactStorage: storage,
    estimatedCommandNodeCount,
    scratchWordCounts: Object.freeze(scratchWordCounts)
  });
}

// Keep the common catalog contract checked without imposing a runtime catalog dependency.
const plannerContract: GPUAnalysisExecutionPlanner<
  GPUClassBreaksShape,
  GPUClassBreaksDeviceLimits
> = {getExecutionPlan: getClassBreaksExecutionPlan};
void plannerContract;
