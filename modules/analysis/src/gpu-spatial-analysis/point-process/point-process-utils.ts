// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Validate the packed numerical views used by bounded point-process kernels. @internal */
export function validatePointProcessView(
  view: GraphDataView,
  format: 'uint32' | 'uint32x2' | 'float32' | 'float32x2',
  length: number,
  id: string = 'point-process'
): void {
  validateSinglePackedView(view, [format], id);
  // A point-process field must cover exactly the declared rows.
  validateRequirement(id, view.length === length, `views must hold exactly ${length} rows`);
}

/** Validate positive finite scalars embedded in numerical kernels. @internal */
export function validatePointProcessPositive(
  value: number,
  id: string = 'point-process',
  name: string = 'scalar'
): void {
  // Positive finite constants must also remain finite and nonzero in shader arithmetic.
  validateRequirement(
    id,
    Number.isFinite(Math.fround(value)) && Math.fround(value) > 0,
    `${name} must be a positive finite number in float32`
  );
}
