// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {validatePackedView} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Validate the packed numerical views used by bounded point-process kernels. @internal */
export function validatePointProcessView(
  view: GraphDataView,
  format: 'uint32' | 'uint32x2' | 'float32' | 'float32x2',
  length: number
): void {
  validatePackedView(view, [format], 'point-process');
  // A point-process field must cover exactly the declared rows.
  assert(view.length === length);
}

/** Validate positive finite scalars embedded in numerical kernels. @internal */
export function validatePointProcessPositive(value: number): void {
  // Positive finite constants must also remain finite and nonzero in shader arithmetic.
  assert(Number.isFinite(Math.fround(value)) && Math.fround(value) > 0);
}
