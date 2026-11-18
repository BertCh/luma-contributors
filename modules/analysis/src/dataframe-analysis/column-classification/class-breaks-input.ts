// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

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
    'naturalBreaksMode must be approximate, exact or monotone'
  );
  validateRequirement(
    id,
    props.naturalBreaksExactStorage === undefined ||
      ['auto', 'materialized', 'recompute'].includes(props.naturalBreaksExactStorage),
    'naturalBreaksExactStorage must be auto, materialized or recompute'
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
  if (props.rowCount !== undefined) validateCapacity(id, 'rowCount', props.rowCount, 1);
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
  input: {format: string; length: number}
): void {
  validateCapacity(id, 'values.length', input.length, 1);
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
