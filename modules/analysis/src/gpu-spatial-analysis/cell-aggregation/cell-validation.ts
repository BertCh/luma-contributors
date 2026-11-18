// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  CELL_LEVEL_PARAMETER_WORD_COUNT,
  CELL_LEVEL_MAXIMUM_BINDING_COUNT
} from './cell-level-kernel';
import type {Device} from '@luma.gl/core';
import {type GPUCommandGraph, type GraphDataView, GraphVectorView} from '@luma.gl/graph/gpu-core';
import {
  getGraphViewChunks,
  validateGPUStorageBindingByteLength,
  validateCapacity,
  validateMaximumCount
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUCellTable} from './cell-table';
import {
  validateCellDispatchOutputs,
  validateCellReadWriteDispatch
} from '../cell-indexing/cell-dispatch-validation';

/** Validate logical separation globally and aligned ranges within each table publisher. @internal */
export function validateCellTableWritableViews(
  id: string,
  table: GPUCellTable,
  extraCounts: readonly GraphDataView<'uint32'>[] = [],
  options: {separateColumns?: boolean; rollup?: boolean} = {}
): void {
  const columns = {
    cells: table.cells,
    counts: table.counts,
    sums: table.sums,
    sumValues: table.sumValues,
    minimums: table.minimums,
    maximums: table.maximums
  };
  const finalStatus = {
    count: table.status.count,
    overflow: table.status.overflow,
    requiredCount: table.status.requiredCount
  };
  const candidateOverflow = table.status.candidateOverflow;
  const counts = Object.fromEntries(
    extraCounts.map((view, index) => [`extraCounts[${index}]`, view])
  );
  const finish = {sumValues: table.sumValues, minimums: table.minimums, maximums: table.maximums};
  validateCellDispatchOutputs(id, {...columns, ...finalStatus, candidateOverflow, ...counts}, [
    ...(options.separateColumns ? [] : [{cells: table.cells, counts: table.counts}, finish]),
    ...(options.rollup && table.sums ? [{counts: table.counts, sums: table.sums}] : []),
    {...finalStatus, ...counts},
    {candidateOverflow}
  ]);
  if (!options.separateColumns && table.sumValues && table.sums)
    validateCellReadWriteDispatch(id, {sums: table.sums}, finish);
}

/** Rejects writable cell-column aliases, including aliases between separate dispatches. @internal */
export function validateCellWritableViews(
  id: string,
  outputs: readonly (readonly [string, GraphDataView | GraphVectorView | undefined])[],
  options: {separateDispatches?: boolean} = {}
): void {
  if (options.separateDispatches) {
    const chunks = outputs.flatMap(([, output]) =>
      (output ? getGraphViewChunks(output) : []).map(view =>
        (view.buffer.graph as GPUCommandGraph).createDataView(view.buffer, {
          format: 'uint32',
          length: (view.length * view.rowByteLength) / 4,
          byteOffset: view.byteOffset
        })
      )
    );
    const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
    validateDisjointWritableViews(id, {
      'output fields': new GraphVectorView({
        id: `${id}-output-fields`,
        name: 'output fields',
        format: 'uint32',
        length,
        valueLength: length,
        stride: 1,
        byteStride: 4,
        rowByteLength: 4,
        data: chunks
      })
    });
    return;
  }
  validateDisjointWritableViews(id, Object.fromEntries(outputs));
}

/** Checks graph membership and actual aligned binding ranges without declaring resources. @internal */
export function validateCellViewBindingRanges<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  views: readonly (GraphDataView | GraphVectorView | undefined)[]
): void {
  validateGraphChunkBindingRanges(
    graph,
    id,
    Object.fromEntries(views.map((view, index) => [`input/output field ${index}`, view]))
  );
  for (const [index, view] of views.entries()) {
    for (const chunk of view ? getGraphViewChunks(view) : []) {
      validateCapacity(
        id,
        `input/output field ${index} buffer byteLength`,
        chunk.buffer.byteLength,
        0,
        graph.device.limits.maxBufferSize
      );
    }
  }
}

/** Checks the widest bindings of the cell sort/scan/reduction scratch before declaration. @internal */
export function validateCellTableScratchBindingSizes(
  device: Device,
  id: string,
  rowCount: number,
  capacity: number
): void {
  const scratchBindings = [
    ['row scratch', Math.max(rowCount, 1) * 4],
    ['cell accumulator scratch', Math.max(capacity, 1) * 8],
    ['cell offsets scratch', (capacity + 1) * 4],
    ['cell level parameters', CELL_LEVEL_PARAMETER_WORD_COUNT * Uint32Array.BYTES_PER_ELEMENT]
  ] as const;
  // Packed scalar sort/scan scratch is bounded by row scratch: published radix histograms
  // use <= one word per row, and scan summaries are smaller. Fixed-point accumulators use
  // two words per cell. Every stage parameter buffer has the fixed layout checked here.
  for (const [name, byteLength] of scratchBindings) {
    validateGPUStorageBindingByteLength(
      device,
      id,
      name,
      byteLength,
      'reduce the total row count or output capacity, or use a device with larger limits'
    );
    validateCapacity(id, `${name} byteLength`, byteLength, 0, device.limits.maxBufferSize);
  }
}

/** The publish kernel binds parameters, total, count, overflow and optional status views. @internal */
export function validateCellPublishBindingCount(
  device: Device,
  id: string,
  requiredCount: boolean,
  overflowSourceCount: number,
  extraCountCount: number
): void {
  validateMaximumCount(
    id,
    'publish storage bindings',
    4 + Number(requiredCount) + overflowSourceCount + extraCountCount,
    Math.min(device.limits.maxStorageBuffersPerShaderStage, CELL_LEVEL_MAXIMUM_BINDING_COUNT + 1)
  );
}
