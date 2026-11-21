// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUGather,
  createTransientView,
  GPUTranspose,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {SPATIAL_REGRESSION_MAXIMUM_PREDICTOR_COUNT} from './spatial-regression-solve';

/** Joins preserve left observation order; completeness certifies that missing keys are genuine. */
export type OrdinaryLeastSquaresJoinedTable = {
  matched: GraphDataView<'uint32'>;
  status: {
    complete: GraphDataView<'uint32'>;
    candidateOverflow?: GraphDataView<'uint32'>;
  };
};

/**
 * Adds a training mask in the join's left observation order. Incomplete tables exclude every row,
 * so OLS reports too few rows and suppresses the fit instead of silently fitting a retained subset.
 * Keep the returned status alongside the fit: candidateOverflow retains the upstream group loss.
 */
export function createOrdinaryLeastSquaresJoinedTableMask<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  joinedTable: OrdinaryLeastSquaresJoinedTable,
  mask?: GraphDataView<'uint32'>,
  id = 'ordinary-least-squares-joined-table'
): {mask: GraphDataView<'uint32'>; status: OrdinaryLeastSquaresJoinedTable['status']} {
  const matched = joinedTable.matched;
  const complete = joinedTable.status.complete;
  const candidateOverflow = joinedTable.status.candidateOverflow;
  validateSinglePackedView(matched, ['uint32'], `${id} joinedTable.matched`);
  validateCapacity(id, 'joinedTable.matched.length', matched.length);
  validateSinglePackedView(complete, ['uint32'], `${id} joinedTable.status.complete`);
  for (const [name, view] of [
    ['complete', complete],
    ['candidateOverflow', candidateOverflow]
  ] as const) {
    if (!view) continue;
    if (name === 'candidateOverflow')
      validateSinglePackedView(view, ['uint32'], `${id} joinedTable.status.${name}`);
    validateRequirement(
      id,
      view.length === 1,
      `joinedTable.status.${name} length must equal 1; received ${view.length}`
    );
  }
  if (mask) {
    validateSinglePackedView(mask, ['uint32'], `${id} mask`);
    validateRequirement(
      id,
      mask.length === matched.length,
      `mask length must equal joinedTable.matched.length (${matched.length}); received ${mask.length}`
    );
  }
  for (const view of [matched, complete, candidateOverflow, mask]) {
    if (view)
      validateRequirement(
        id,
        view.buffer.graph === graph,
        'joinedTable and mask views must belong to the supplied graph'
      );
  }
  const trainingMask = createTransientView(graph, `${id}-training-mask`, 'uint32', matched.length);
  graph.add(
    createWGSLKernelNode(graph, {
      id: `${id}-training-mask`,
      operation: 'OrdinaryLeastSquaresJoinedTableMask',
      variant: 'left-order',
      invocationCount: matched.length,
      bindings: [
        {name: 'matched', view: matched, type: 'u32', access: 'read'},
        {name: 'complete', view: complete, type: 'u32', access: 'read'},
        ...(candidateOverflow
          ? [
              {
                name: 'candidateOverflow',
                view: candidateOverflow,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : []),
        ...(mask
          ? [{name: 'mask', view: mask, type: 'u32' as const, access: 'read' as const}]
          : []),
        {name: 'trainingMask', view: trainingMask, type: 'u32', access: 'read_write'}
      ],
      body: `let usable = complete[completeOffset] != 0u && matched[matchedOffset + index] != 0u${candidateOverflow ? ' && candidateOverflow[candidateOverflowOffset] == 0u' : ''}${mask ? ' && mask[maskOffset + index] != 0u' : ''};
  trainingMask[trainingMaskOffset + index] = select(0u, 1u, usable);`
    })
  );
  return {
    mask: trainingMask,
    status: Object.freeze({complete, ...(candidateOverflow ? {candidateOverflow} : {})})
  };
}

/**
 * Adapts float32 predictor columns in coefficient order to OLS's row-major design. A single column
 * retains its identity. Multiple columns use graph gathers and the upstream GPUTranspose.
 * Columns must be nonempty packed views of the same length. Their contents remain dynamic.
 */
export function createOrdinaryLeastSquaresDesign<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  columns: readonly GraphDataView<'float32'>[],
  id = 'ordinary-least-squares-design'
): GraphDataView<'float32'> {
  const captured = [...columns];
  validateCapacity(
    id,
    'columns.length',
    captured.length,
    1,
    SPATIAL_REGRESSION_MAXIMUM_PREDICTOR_COUNT - 1
  );
  const rowCount = captured[0].length;
  validateCapacity(id, 'columns[0].length', rowCount);
  validateCapacity(id, 'rowCount * columns.length', rowCount * captured.length);
  for (const [index, column] of captured.entries()) {
    validateSinglePackedView(column, ['float32'], `${id} columns[${index}]`);
    validateRequirement(
      id,
      column.length === rowCount,
      `columns[${index}].length must equal ${rowCount}; received ${column.length}`
    );
    validateRequirement(
      id,
      column.buffer.graph === graph,
      `columns[${index}] must belong to the supplied graph`
    );
  }
  if (captured.length === 1) return captured[0];
  const columnMajor = createTransientView(
    graph,
    `${id}-columns`,
    'float32',
    rowCount * captured.length
  );
  const predictors = createTransientView(
    graph,
    `${id}-rows`,
    'float32',
    rowCount * captured.length
  );
  const rows = createTransientView(graph, `${id}-row-indices`, 'uint32', rowCount);
  graph.add(
    createWGSLKernelNode(graph, {
      id: `${id}-row-indices`,
      operation: 'OrdinaryLeastSquaresDesign',
      variant: 'observation-rows',
      invocationCount: rowCount,
      bindings: [{name: 'rows', view: rows, type: 'u32', access: 'read_write'}],
      body: 'rows[rowsOffset + index] = index;'
    })
  );
  for (const [index, column] of captured.entries()) {
    graph.add(
      new GPUGather({
        id: `${id}-column-${index}`,
        source: column,
        indices: rows,
        output: graph.createDataView(columnMajor.buffer, {
          format: 'float32',
          length: rowCount,
          byteOffset: index * rowCount * 4
        })
      })
    );
  }
  graph.add(
    new GPUTranspose({
      id: `${id}-transpose`,
      input: columnMajor,
      output: predictors,
      rows: captured.length,
      columns: rowCount
    })
  );
  return predictors;
}
