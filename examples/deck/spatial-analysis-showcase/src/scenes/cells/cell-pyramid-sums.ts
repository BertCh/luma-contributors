// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUCellTable} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {addKernelPass} from '../../engine/mode-kernels';

/** Scene selection: certify sum columns independently of counts and clear inactive tails. */
export function addShowcaseCellTableColumnsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  table: Pick<GPUCellTable, 'counts' | 'sumValues' | 'status'>,
  output: {values: GraphDataView<'float32'>; mask: GraphDataView<'uint32'>}
): void {
  const sumValues = table.sumValues;
  const lossViews = [
    table.status.overflow,
    table.status.candidateOverflow,
    ...(sumValues ? [table.status.sumOverflow] : [])
  ].filter((view): view is GraphDataView<'uint32'> => Boolean(view));
  addKernelPass(graph, {
    id: `${id}-cell-table-columns`,
    invocationCount: table.counts.length,
    bindings: [
      {name: 'counts', view: table.counts, type: 'u32', access: 'read'},
      {name: 'rowCount', view: table.status.count, type: 'u32', access: 'read'},
      ...(sumValues
        ? [{name: 'sumValues', view: sumValues, type: 'f32', access: 'read'} as const]
        : []),
      ...lossViews.map((view, index) => ({
        name: `loss${index}`,
        view,
        type: 'u32' as const,
        access: 'read' as const
      })),
      {name: 'columnValues', view: output.values, type: 'f32', access: 'read_write'},
      {name: 'columnMask', view: output.mask, type: 'u32', access: 'read_write'}
    ],
    body: `let complete = ${lossViews.map((_, index) => `loss${index}[loss${index}Offset] == 0u`).join(' && ') || 'true'};
  let occupied = counts[countsOffset + index];
  let isActive = complete && index < rowCount[rowCountOffset] && occupied > 0u;
  var value = 0.0;
  if (isActive) { value = ${sumValues ? 'sumValues[sumValuesOffset + index]' : 'f32(occupied)'}; }
  columnValues[columnValuesOffset + index] = value;
  columnMask[columnMaskOffset + index] = select(0u, 1u, isActive);`
  });
}

/** Preserve the table's counts while hiding uncertified sum styling on the scene layer. */
export function addShowcaseCertifiedSumCountsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  counts: GraphDataView<'uint32'>,
  status: GPUCellTable['status'],
  output: GraphDataView<'uint32'>
): void {
  const losses = [status.overflow, status.candidateOverflow, status.sumOverflow].filter(
    (view): view is GraphDataView<'uint32'> => Boolean(view)
  );
  addKernelPass(graph, {
    id,
    invocationCount: counts.length,
    bindings: [
      {name: 'counts', view: counts, type: 'u32', access: 'read'},
      {name: 'rowCount', view: status.count, type: 'u32', access: 'read'},
      ...losses.map((view, index) => ({
        name: `loss${index}`,
        view,
        type: 'u32' as const,
        access: 'read' as const
      })),
      {name: 'certifiedCounts', view: output, type: 'u32', access: 'read_write'}
    ],
    body: `let complete = ${losses.map((_, index) => `loss${index}[loss${index}Offset] == 0u`).join(' && ') || 'true'};
  certifiedCounts[certifiedCountsOffset + index] = select(0u, counts[countsOffset + index], complete && index < rowCount[rowCountOffset]);`
  });
}
