// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Checks a setting before packing; optionally preserves zero as an explicit control. @internal */
export function validateGeometryFloat32Setting(
  id: string,
  name: string,
  value: number,
  nonnegative = false,
  preservePositive = false
): void {
  const converted = Math.fround(value);
  validateRequirement(
    id,
    Number.isFinite(value) &&
      Number.isFinite(converted) &&
      (!nonnegative || value >= 0) &&
      (!preservePositive || value === 0 || converted > 0),
    `${name} must be finite${nonnegative ? ' and non-negative' : ''} after float32 conversion${preservePositive ? ', with positive values remaining positive' : ''}; received ${value}`
  );
}
