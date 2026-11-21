// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

type PointProcessFormat = 'uint32' | 'uint32x2' | 'uint32x4' | 'float32' | 'float32x2';

/** Validate point-process bindings with an explicit identity, field path and expected rows. @internal */
export function validatePointProcessView(
  id: string,
  path: string,
  view: GraphDataView,
  formats: readonly PointProcessFormat[],
  expectedRows: number
): void {
  validateRequirement(
    'point-process',
    typeof id === 'string',
    'Use validatePointProcessView(id, path, view, formats, expectedRows)'
  );
  validateSinglePackedView(view, formats, {id, contributor: 'point-process'}, path);
  validateRequirement(
    id,
    view.length === expectedRows,
    `${path} must have exactly ${expectedRows} rows. Received ${view.length}`,
    {code: 'VIEW_LENGTH', path, received: view.length, expected: {rows: expectedRows}}
  );
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
