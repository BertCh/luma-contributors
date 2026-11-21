// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Supported cell arithmetic requests. Double-single GPU arithmetic uses two float32 values. */
export type CellPrecision = 'float32' | 'double-single';

/** Resolve and validate before choosing a kernel, including in named and pure CPU forms. */
export function resolveCellPrecision(
  id: string,
  precision: CellPrecision | undefined,
  supportsDoubleSingle = true
): 'float32' | 'double-single' {
  validateRequirement(
    id,
    precision === undefined || precision === 'float32' || precision === 'double-single',
    `precision received ${String(precision)}; use 'float32' or 'double-single'`
  );
  const resolved = precision ?? 'float32';
  validateRequirement(
    id,
    resolved === 'float32' || supportsDoubleSingle,
    `precision ${String(precision)} requires family 'h3' for point indexing; use 'float32' for this family`
  );
  return resolved;
}
