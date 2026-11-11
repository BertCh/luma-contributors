// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Display adapters for zone-by-zone flow matrices: GPU kernels that read an operation's output and
// reshape it into per-zone rasters (one origin's row, the column sums) or a combined population. The
// matrices stay on the GPU; nothing is read back.

import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
import type {RunBuffer, RunBuilder} from '../../../define-run';

/** Row `selection[0]` of a `zoneCount` by `zoneCount` matrix: the flows leaving the selected zone. */
export function addMatrixRow(
  builder: RunBuilder,
  options: {
    id: string;
    matrix: RunBuffer<'float32'>;
    zoneCount: number;
    selection: RunBuffer<'uint32'>;
  }
): RunBuffer<'float32'> {
  const row = builder.allocate(`${options.id}-row`, 'float32', options.zoneCount);
  builder.graph.add(
    createWGSLKernelNode(builder.graph, {
      id: `${options.id}-row`,
      operation: 'display-matrix-row',
      invocationCount: options.zoneCount,
      bindings: [
        {name: 'matrix', view: options.matrix.view, type: 'f32', access: 'read'},
        {name: 'selection', view: options.selection.view, type: 'u32', access: 'read'},
        {name: 'row', view: row.view, type: 'f32', access: 'read_write'}
      ],
      body: `row[rowOffset + index] = matrix[matrixOffset + selection[selectionOffset] * ${options.zoneCount}u + index];`
    })
  );
  return row;
}

/**
 * Share of each zone's outflow that a flow matrix places: the row sum divided by `outflows`.
 * Zones with no outflow get 1. A radiation model places `1 - m_i / M` of it, and its finite-size
 * normalization makes every row exactly 1.
 */
export function addRowShares(
  builder: RunBuilder,
  options: {
    id: string;
    matrix: RunBuffer<'float32'>;
    outflows: RunBuffer<'float32'>;
    zoneCount: number;
  }
): RunBuffer<'float32'> {
  const shares = builder.allocate(`${options.id}-row-shares`, 'float32', options.zoneCount);
  builder.graph.add(
    createWGSLKernelNode(builder.graph, {
      id: `${options.id}-row-shares`,
      operation: 'display-row-shares',
      invocationCount: options.zoneCount,
      bindings: [
        {name: 'matrix', view: options.matrix.view, type: 'f32', access: 'read'},
        {name: 'outflows', view: options.outflows.view, type: 'f32', access: 'read'},
        {name: 'shares', view: shares.view, type: 'f32', access: 'read_write'}
      ],
      body: `
  var total = 0.0;
  for (var destination = 0u; destination < ${options.zoneCount}u; destination = destination + 1u) {
    total = total + matrix[matrixOffset + index * ${options.zoneCount}u + destination];
  }
  let outflow = outflows[outflowsOffset + index];
  shares[sharesOffset + index] = select(1.0, total / outflow, outflow > 0.0);`
    })
  );
  return shares;
}
