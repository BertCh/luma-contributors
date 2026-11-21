// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPUNetworkCostWeights} from '@luma-contributors/analysis/gpu-network';
import {
  validateRequirement,
  validateSinglePackedView,
  validateViewLength,
  validateGraphViewsBelongToGraph
} from '@luma-contributors/analysis/analysis-core';

/** Uploaded inputs and the facility-by-node cost matrix produced by snapping and a cost matrix. */
export type NetworkCostWeightsExampleInput = {
  graph: GPUCommandGraph;
  matrixCosts: GraphDataView<'float32'>;
  parameters: GraphDataView<'float32'>;
  facilityCount: number;
  nodeCount: number;
} & (
  | {matrixComplete?: undefined; matrixCostLimit?: undefined}
  | {
      /** The matrix completeness certificate, required with a bounded horizon. */
      matrixComplete: GraphDataView<'uint32'>;
      matrixCostLimit?: GraphDataView<'float32'>;
    }
);

/** Builds the contributor shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(_device: Device, input: NetworkCostWeightsExampleInput) {
  const id = 'network-cost-weights-example';
  const hasMatrixComplete = input.matrixComplete !== undefined;
  const hasMatrixCostLimit = input.matrixCostLimit !== undefined;
  for (const [name, view, format] of [
    ['matrixComplete', input.matrixComplete, 'uint32'],
    ['matrixCostLimit', input.matrixCostLimit, 'float32']
  ] as const) {
    if (view !== undefined) {
      validateSinglePackedView(view, [format], {id, contributor: 'GPUNetworkCostWeights'}, name);
      validateViewLength(id, name, view, 1);
      validateGraphViewsBelongToGraph(id, input.graph, [view]);
    }
  }
  validateRequirement(
    id,
    !hasMatrixCostLimit || hasMatrixComplete,
    'matrixCostLimit requires matrixComplete; received matrixCostLimit without matrixComplete'
  );
  const {graph, matrixCosts, parameters, facilityCount, nodeCount} = input;
  const slotCount = facilityCount * nodeCount;
  const {matrixComplete, matrixCostLimit} = input;
  const complete = hasMatrixComplete
    ? createTransientView(graph, 'weightComplete', 'uint32', 1)
    : undefined;
  // #region example
  const costWeights = new GPUNetworkCostWeights({
    costs: matrixCosts,
    costsComplete: matrixComplete,
    costLimit: matrixCostLimit,
    rowCount: facilityCount,
    orientation: 'rows',
    parameters,
    output: {
      offsets: createTransientView(graph, 'weightOffsets', 'uint32', facilityCount + 1),
      neighbors: createTransientView(graph, 'weightNeighbors', 'uint32', slotCount),
      weights: createTransientView(graph, 'weightValues', 'float32', slotCount),
      distances: createTransientView(graph, 'weightDistances', 'float32', slotCount),
      status: {complete, overflow: createTransientView(graph, 'weightOverflow', 'uint32', 1)}
    }
  });
  graph.add(costWeights);
  const {offsets, neighbors, weights} = costWeights.props.output;
  // #endregion example
  const {distances, status} = costWeights.props.output;
  return {costWeights, offsets, neighbors, weights, distances: distances!, status: status!};
}
