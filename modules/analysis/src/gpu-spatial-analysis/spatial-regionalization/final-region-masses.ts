// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createLoadBalancedSegmentSumNodes} from '../../analysis-core/kernels/load-balanced-for-each';
import {createSortedSegmentOffsetsNode} from '../../analysis-core/kernels/sorted-segment-offsets';
import {getSortKeyBits} from '../../analysis-core/kernels/sorted-segment-sums';
import {validateGPUStorageBindingByteLength} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Sort, segmented mass sum and final count, including published primitive overhead. @internal */
export const FINAL_REGION_MASS_NODE_ALLOWANCE = 128;

/** @internal Validate the widest scratch bindings before the caller allocates graph nodes. */
export function validateFinalRegionMassScratch<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  rows: number
): void {
  validateGPUStorageBindingByteLength(
    graph.device,
    id,
    'final mass label offsets',
    (rows + 1) * 4,
    'reduce rowCount'
  );
}

/** Count nonempty final regions below their extensive floor, not their construction phase. @internal */
export function createFinalRegionMassNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  labels: GraphDataView<'uint32'>,
  extensive: GraphDataView<'float32'>,
  threshold: GraphDataView<'float32'>
) {
  const rows = labels.length;
  const rowIds = createTransientView(graph, `${id}-mass-row-ids`, 'uint32', rows);
  const sortedLabels = createTransientView(graph, `${id}-mass-labels`, 'uint32', rows);
  const sortedRows = createTransientView(graph, `${id}-mass-rows`, 'uint32', rows);
  const offsets = createTransientView(graph, `${id}-mass-offsets`, 'uint32', rows + 1);
  const masses = createTransientView(graph, `${id}-mass-sums`, 'float32', rows);
  const count = createTransientView(graph, `${id}-final-infeasible`, 'uint32', 1);
  const nodes = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-mass-initialize`,
      operation: 'GPUMaxPRegions',
      invocationCount: rows,
      bindings: [
        {name: 'rowIds', view: rowIds, type: 'u32', access: 'read_write'},
        {name: 'count', view: count, type: 'u32', access: 'read_write'}
      ],
      body: 'rowIds[rowIdsOffset + index] = index; if (index == 0u) { count[countOffset] = 0u; }'
    }),
    ...new GPUSort({
      id: `${id}-mass-sort`,
      keys: labels,
      values: rowIds,
      outputKeys: sortedLabels,
      outputValues: sortedRows,
      keyBits: getSortKeyBits(rows)
    }).getCommandNodes(graph),
    createSortedSegmentOffsetsNode<Parameters>(graph, {
      id: `${id}-mass-offsets`,
      operation: 'GPUMaxPRegions',
      segmentCount: rows,
      sortedKeys: sortedLabels,
      segmentOffsets: offsets
    }),
    ...createLoadBalancedSegmentSumNodes<Parameters>(graph, {
      id: `${id}-mass-sums`,
      operation: 'GPUMaxPRegions',
      offsets,
      itemCapacity: rows,
      accumulatorType: 'f32',
      output: masses,
      bindings: [
        {name: 'extensive', view: extensive, type: 'f32', access: 'read'},
        {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'}
      ],
      declarations: `fn getLoadBalancedContribution(region: u32, item: u32, lane: u32) -> f32 {
  return extensive[extensiveOffset + sortedRows[sortedRowsOffset + item]];
}`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-mass-count`,
      operation: 'GPUMaxPRegions',
      invocationCount: rows,
      bindings: [
        {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
        {name: 'masses', view: masses, type: 'f32', access: 'read'},
        {name: 'threshold', view: threshold, type: 'f32', access: 'read'},
        {name: 'count', view: count, type: 'atomic<u32>', access: 'read_write'}
      ],
      body: `if (offsets[offsetsOffset + index] < offsets[offsetsOffset + index + 1u] &&
      masses[massesOffset + index] < threshold[thresholdOffset]) { atomicAdd(&count[countOffset], 1u); }`
    })
  ];
  return {nodes, count};
}
