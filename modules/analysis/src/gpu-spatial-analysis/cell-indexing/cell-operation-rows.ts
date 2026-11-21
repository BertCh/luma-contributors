// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Check resolved inputs even when a named indexing operation needs no dispatch. */
export function validatePointOperationRows(
  id: string,
  rowCount: number,
  positions: GraphDataView<'float32x2'>,
  mask?: GraphDataView<'uint32'>
): void {
  validateRequirement(
    id,
    positions.length === rowCount,
    `positions length must equal rowCount ${rowCount}; received ${positions.length}`
  );
  if (mask !== undefined) {
    validateRequirement(
      id,
      mask.length === rowCount,
      `mask length must equal rowCount ${rowCount}; received ${mask.length}`
    );
  }
}
