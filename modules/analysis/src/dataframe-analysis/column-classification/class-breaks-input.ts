// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

import {
  GPU_CLASS_BREAKS_METHODS,
  GPU_CLASS_BREAKS_BOX_PLOT_CLASS_COUNT,
  type GPUClassBreaksMethod
} from './class-breaks-parameters';

/** Shared default coarse histogram capacity for graph admission. @internal */
export const DEFAULT_NATURAL_BREAKS_BIN_COUNT = 2048;

/** Resolve the compatible default methods; explicit selections retain strict admission. @internal */
export function resolveClassBreaksMethods(
  id: string,
  props: {
    maximumClassCount: number;
    methods?: readonly GPUClassBreaksMethod[];
    naturalBreaksMode?: 'approximate' | 'exact' | 'monotone';
    naturalBreaksBinCount?: number;
  },
  chunked = false,
  usesBinnedNaturalBreaks = true
): readonly GPUClassBreaksMethod[] {
  const compatible = (method: GPUClassBreaksMethod) =>
    (method !== 'box-plot' || props.maximumClassCount >= GPU_CLASS_BREAKS_BOX_PLOT_CLASS_COUNT) &&
    (!chunked ||
      (method !== 'maximum-breaks' &&
        (method !== 'natural-breaks' ||
          (props.naturalBreaksMode ?? 'approximate') === 'approximate'))) &&
    (!usesBinnedNaturalBreaks ||
      method !== 'natural-breaks' ||
      props.naturalBreaksMode === 'exact' ||
      props.naturalBreaksMode === 'monotone' ||
      props.maximumClassCount <= (props.naturalBreaksBinCount ?? DEFAULT_NATURAL_BREAKS_BIN_COUNT));
  const methods = [...new Set(props.methods ?? GPU_CLASS_BREAKS_METHODS.filter(compatible))];
  validateRequirement(id, methods.length > 0, 'methods needs at least one method');
  for (const method of methods) {
    validateRequirement(
      id,
      GPU_CLASS_BREAKS_METHODS.includes(method),
      `methods unknown method ${String(method)}; accepted values: ${GPU_CLASS_BREAKS_METHODS.join(', ')}`
    );
    validateRequirement(
      id,
      method !== 'box-plot' || props.maximumClassCount >= GPU_CLASS_BREAKS_BOX_PLOT_CLASS_COUNT,
      `box-plot needs maximumClassCount of at least 6; received ${props.maximumClassCount}`
    );
    validateRequirement(
      id,
      !chunked ||
        (method !== 'maximum-breaks' &&
          (method !== 'natural-breaks' ||
            (props.naturalBreaksMode ?? 'approximate') === 'approximate')),
      `methods ${method} requires one packed view; use approximate natural-breaks or omit this method for chunked input`
    );
  }
  return Object.freeze(methods);
}

/** Optional assertions on the resolved class-breaks input. @internal */
export type ClassBreaksInputAssertions = {
  rowCount?: number;
  valuesFormat?: 'float32' | 'uint32' | 'sint32';
};

/** Validate option dependencies independently of the selected method and device limits. @internal */
export function validateClassBreaksNaturalOptions(
  id: string,
  props: {
    naturalBreaksMode?: 'approximate' | 'exact' | 'monotone';
    naturalBreaksExactStorage?: 'auto' | 'materialized' | 'recompute';
  }
): void {
  validateRequirement(
    id,
    props.naturalBreaksMode === undefined ||
      ['approximate', 'exact', 'monotone'].includes(props.naturalBreaksMode),
    `naturalBreaksMode must be approximate, exact or monotone; received ${String(props.naturalBreaksMode)}`
  );
  validateRequirement(
    id,
    props.naturalBreaksExactStorage === undefined ||
      ['auto', 'materialized', 'recompute'].includes(props.naturalBreaksExactStorage),
    `naturalBreaksExactStorage must be auto, materialized or recompute; received ${String(props.naturalBreaksExactStorage)}`
  );
  validateRequirement(
    id,
    props.naturalBreaksExactStorage === undefined || props.naturalBreaksMode === 'exact',
    'naturalBreaksExactStorage requires exact mode'
  );
}

/** Validates declarations before either operation form resolves graph values. @internal */
export function validateClassBreaksInputAssertions(
  id: string,
  props: ClassBreaksInputAssertions
): void {
  if (props.rowCount !== undefined) validateCapacity(id, 'rowCount', props.rowCount, 0);
  validateRequirement(
    id,
    props.valuesFormat === undefined ||
      ['float32', 'uint32', 'sint32'].includes(props.valuesFormat),
    'valuesFormat must be float32, uint32 or sint32'
  );
}

/** Checks the same actual input format and optional assertions on both forms. @internal */
export function validateClassBreaksInput(
  id: string,
  props: ClassBreaksInputAssertions,
  input: {format: string; length: number},
  allowEmpty = false
): void {
  validateRequirement(
    id,
    allowEmpty || input.length > 0,
    `values.length received ${input.length}; GPU class breaks requires at least one row; skip the GPU operation for empty input or use CPUClassBreaksOperation`
  );
  validateCapacity(id, 'values.length', input.length, 0);
  validateRequirement(
    id,
    ['float32', 'uint32', 'sint32'].includes(input.format),
    'values format must be float32, uint32 or sint32'
  );
  validateRequirement(
    id,
    props.rowCount === undefined || input.length === props.rowCount,
    `values length ${input.length} must equal rowCount ${props.rowCount}`
  );
  validateRequirement(
    id,
    props.valuesFormat === undefined || input.format === props.valuesFormat,
    `values format ${input.format} must equal valuesFormat ${props.valuesFormat}`
  );
}

/** Shared graph-form preflight; the pure reference intentionally keeps exact Fisher-Jenks. @internal */
export function validateClassBreaksGraphOptions(
  id: string,
  props: Parameters<typeof resolveClassBreaksMethods>[1] &
    Parameters<typeof validateClassBreaksNaturalOptions>[1],
  backend: 'cpu' | 'gpu' = 'gpu'
): readonly GPUClassBreaksMethod[] {
  validateClassBreaksNaturalOptions(id, props);
  validateCapacity(id, 'maximumClassCount', props.maximumClassCount, 1, 0xfffffffe);
  if (props.naturalBreaksBinCount !== undefined)
    validateCapacity(
      id,
      'naturalBreaksBinCount',
      props.naturalBreaksBinCount,
      2,
      backend === 'gpu' ? DEFAULT_NATURAL_BREAKS_BIN_COUNT : 0xffffffff
    );
  // CPU executes exact Fisher-Jenks on packed rows. GPU bin/layout restrictions must not
  // silently remove its default natural method. Explicit approximate declarations still
  // receive the shared graph relationship check against the same resolved bin default.
  const methods = resolveClassBreaksMethods(id, props, false, backend === 'gpu');
  const explicitlyApproximate =
    props.naturalBreaksMode === 'approximate' || props.methods?.includes('natural-breaks');
  validateRequirement(
    id,
    (backend === 'cpu' && !explicitlyApproximate) ||
      !methods.includes('natural-breaks') ||
      (props.naturalBreaksMode ?? 'approximate') !== 'approximate' ||
      props.maximumClassCount <= (props.naturalBreaksBinCount ?? DEFAULT_NATURAL_BREAKS_BIN_COUNT),
    `maximumClassCount ${props.maximumClassCount} must not exceed naturalBreaksBinCount ${props.naturalBreaksBinCount ?? DEFAULT_NATURAL_BREAKS_BIN_COUNT} in approximate natural-breaks mode`
  );
  return methods;
}

/** Records the natural-breaks route even though the packed per-frame method may select another method. @internal */
export function getClassBreaksProvenance(
  props: {
    naturalBreaksMode?: 'approximate' | 'exact' | 'monotone';
    methods?: readonly GPUClassBreaksMethod[];
    maximumClassCount: number;
  },
  backend: 'cpu' | 'gpu'
) {
  const naturalBreaksMode = props.naturalBreaksMode ?? 'approximate';
  const naturalBreaksAlgorithm =
    backend === 'cpu'
      ? 'exact-fisher-jenks-float64'
      : naturalBreaksMode === 'exact'
        ? 'exact-fisher-jenks-normalized-float32'
        : naturalBreaksMode === 'monotone'
          ? 'monotone-quantized-dynamic-programming'
          : 'histogram-refinement';
  return Object.freeze({
    naturalBreaksMode,
    naturalBreaksAlgorithm,
    methods: Object.freeze([
      ...resolveClassBreaksMethods('class-breaks-provenance', props, false, backend !== 'cpu')
    ])
  });
}
