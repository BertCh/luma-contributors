// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Declared source-column index space and diagonal policy for an uploaded CSR matrix. */
export type SpatialWeightsValidationOptions = {
  /** Source-column count; neighbor rows must be below it. Defaults to the target-row count. */
  columnCount?: number;
  /** Defaults to forbid. require-unit requires one weight=1 diagonal slot per target row. */
  diagonal?: 'forbid' | 'allow' | 'require-unit';
};

/** Shared construction-time defaults and validation for every validation form. @internal */
export function resolveSpatialWeightsValidationOptions(
  id: string,
  rowCount: number,
  options: SpatialWeightsValidationOptions
): Required<SpatialWeightsValidationOptions> {
  const columnCount = options.columnCount ?? rowCount;
  const diagonal = options.diagonal ?? 'forbid';
  validateCapacity(id, 'columnCount', columnCount, 0);
  validateRequirement(
    id,
    ['forbid', 'allow', 'require-unit'].includes(diagonal),
    `diagonal must be one of forbid, allow, require-unit; received ${String(diagonal)}`
  );
  validateRequirement(
    id,
    diagonal !== 'require-unit' || columnCount >= rowCount,
    `columnCount must be >= rowCount (${rowCount}) for diagonal require-unit; received ${columnCount}; use diagonal allow for a matrix without every target diagonal`
  );
  return Object.freeze({columnCount, diagonal});
}
