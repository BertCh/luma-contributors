// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUTranspose,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement,
  validateSinglePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUNetworkCostMatrixResult} from '../../analysis-core/contracts/network-cost-matrix-result';
import {GPULocationAllocation, type GPULocationAllocationProps} from './gpu-location-allocation';

/** Storage axes are explicit even for square matrices; trips describe the original directed costs. */
export type GPUNetworkLocationAllocationAdapterProps = Omit<
  GPULocationAllocationProps,
  'costs' | 'sparseCosts' | 'facilityCount' | 'costsComplete' | 'costLimit'
> & {
  matrix: GPUNetworkCostMatrixResult;
  facilityAxis: 'rows' | 'columns';
  trips: 'demand-to-facility' | 'facility-to-demand';
  /** Optional assignment in the matrix's network-node domain; 0xffffffff for unassigned demand. */
  assignmentNodes?: GraphDataView<'uint32'>;
};

/**
 * Adds allocation on a labelled network matrix, transposing storage with public GPUTranspose when
 * facilities occupy rows. Transposition preserves trip direction. Ordinary assignments remain
 * facility-axis rows; assignmentNodes maps them through the target list (or one start per row).
 * Convergence and source completeness are combined on-device, and the horizon is always forwarded.
 */
export function addNetworkLocationAllocationAdapter<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUNetworkLocationAllocationAdapterProps
): {contributor: GPULocationAllocation; facilityNodes?: GraphDataView<'uint32'>} {
  const id = props.id ?? 'network-location-allocation-adapter';
  const matrix = {...props.matrix};
  validateCapacity(id, 'matrix.rowCount', matrix.rowCount);
  validateCapacity(id, 'matrix.columnCount', matrix.columnCount);
  validateRequirement(
    id,
    ['rows', 'columns'].includes(props.facilityAxis),
    'facilityAxis must be rows or columns'
  );
  validateRequirement(
    id,
    ['start-to-target', 'target-to-start'].includes(matrix.direction),
    'matrix.direction must be start-to-target or target-to-start'
  );
  validateRequirement(
    id,
    ['demand-to-facility', 'facility-to-demand'].includes(props.trips),
    'trips must be demand-to-facility or facility-to-demand'
  );
  const facilitiesInColumns = props.facilityAxis === 'columns';
  const demandToFacility = facilitiesInColumns
    ? matrix.direction === 'start-to-target'
    : matrix.direction === 'target-to-start';
  validateRequirement(
    id,
    demandToFacility === (props.trips === 'demand-to-facility'),
    'trips must match matrix.direction and facilityAxis; transposing storage does not reverse trips'
  );
  validateSinglePackedView(matrix.values, ['float32'], `${id} matrix.values`);
  validateRequirement(
    id,
    matrix.values.length === matrix.rowCount * matrix.columnCount,
    `matrix.values length must equal rowCount * columnCount; received ${matrix.values.length}`
  );
  validateRequirement(
    id,
    Boolean(matrix.converged),
    'matrix.converged is required; request the matrix convergence output'
  );
  for (const [name, view] of [
    ['matrix.converged', matrix.converged],
    ['matrix.sourceComplete', matrix.sourceComplete]
  ] as const) {
    if (!view) continue;
    validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
    validateRequirement(
      id,
      view.length === 1,
      `${name} length must equal 1; received ${view.length}`
    );
  }
  const facilityCount = facilitiesInColumns ? matrix.columnCount : matrix.rowCount;
  const demandCount = facilitiesInColumns ? matrix.rowCount : matrix.columnCount;
  validateRequirement(
    id,
    facilitiesInColumns ||
      (!matrix.startRows &&
        matrix.startsPerRow === 1 &&
        matrix.startNodes.length === matrix.rowCount),
    'facility rows require one start node per row without startRows; use a target-list matrix with facilityAxis columns for grouped starts'
  );
  const facilityNodes = facilitiesInColumns ? matrix.targetNodes : matrix.startNodes;
  if (facilityNodes) {
    validateSinglePackedView(facilityNodes, ['uint32'], `${id} facilityNodes`);
    validateRequirement(
      id,
      facilityNodes.length === facilityCount,
      `facilityNodes length must equal facilityCount ${facilityCount}; received ${facilityNodes.length}`
    );
  }
  if (props.assignmentNodes) {
    validateSinglePackedView(props.assignmentNodes, ['uint32'], `${id} assignmentNodes`);
    validateRequirement(
      id,
      props.assignmentNodes.length === demandCount,
      `assignmentNodes length must equal demandCount ${demandCount}; received ${props.assignmentNodes.length}`
    );
  }
  validateGraphViewsBelongToGraph(id, graph, [
    matrix.values,
    matrix.converged,
    matrix.sourceComplete,
    matrix.costLimit,
    facilityNodes,
    props.assignmentNodes
  ]);
  validateGraphOutputsDisjointFromInputs(
    id,
    [...Object.values(props.output), props.assignmentNodes],
    [matrix.values, matrix.converged, matrix.sourceComplete, matrix.costLimit, facilityNodes]
  );
  validateGraphOutputsDisjointFromInputs(id, [props.assignmentNodes], Object.values(props.output));
  // Validate allocation dimensions/options before registering any adapter scratch. Validation
  // depends on the dense dimensions, not its axis order; the final binding is replaced below.
  const validated = new GPULocationAllocation({
    ...props,
    id,
    costs: matrix.values,
    facilityCount,
    costsComplete: matrix.converged,
    costLimit: matrix.costLimit
  });
  let costs = matrix.values;
  if (!facilitiesInColumns)
    costs = createTransientView(graph, `${id}-demand-major`, 'float32', matrix.values.length);
  let costsComplete = matrix.converged!;
  if (matrix.sourceComplete)
    costsComplete = createTransientView(graph, `${id}-costs-complete`, 'uint32', 1);
  const contributor =
    costs === matrix.values && costsComplete === matrix.converged
      ? validated
      : new GPULocationAllocation({
          ...validated.props,
          id,
          costs,
          facilityCount,
          costsComplete,
          costLimit: matrix.costLimit
        });
  if (!facilitiesInColumns)
    graph.add(
      new GPUTranspose({
        id: `${id}-transpose`,
        input: matrix.values,
        output: costs,
        rows: matrix.rowCount,
        columns: matrix.columnCount
      })
    );
  if (matrix.sourceComplete)
    graph.add(
      createWGSLKernelNode(graph, {
        id: `${id}-combine-costs-complete`,
        operation: 'NetworkLocationAllocationAdapter',
        invocationCount: 2,
        bindings: [
          {name: 'search', view: matrix.converged!, type: 'u32', access: 'read'},
          {name: 'source', view: matrix.sourceComplete, type: 'u32', access: 'read'},
          {name: 'complete', view: costsComplete, type: 'u32', access: 'read_write'}
        ],
        body: 'if (index == 0u) { complete[completeOffset] = u32(search[searchOffset] != 0u && source[sourceOffset] != 0u); }'
      })
    );
  graph.add(contributor);
  if (props.assignmentNodes)
    graph.add(
      createWGSLKernelNode(graph, {
        id: `${id}-assignment-nodes`,
        operation: 'NetworkLocationAllocationAdapter',
        invocationCount: demandCount,
        bindings: [
          {name: 'assignments', view: props.output.assignments, type: 'u32', access: 'read'},
          ...(facilityNodes
            ? [
                {
                  name: 'facilityNodes',
                  view: facilityNodes,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'nodes', view: props.assignmentNodes, type: 'u32', access: 'read_write'}
        ],
        body: `let facility = assignments[assignmentsOffset + index];
  var node = 0xffffffffu;
  if (facility < ${facilityCount}u) { node = ${facilityNodes ? 'facilityNodes[facilityNodesOffset + facility]' : 'facility'}; }
  nodes[nodesOffset + index] = node;`
      })
    );
  return {contributor, facilityNodes};
}
