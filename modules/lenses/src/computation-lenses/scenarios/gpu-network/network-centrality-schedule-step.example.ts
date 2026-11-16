// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPUNetworkCentralityScheduleStep,
  type GPUNetworkTopology
} from '@luma-contributors/analysis/gpu-network';

/** Uploaded inputs: the roads, their exact transpose and the sampled pivots, reset flag and persistent status. */
export type NetworkCentralityScheduleStepExampleInput = {
  graph: GPUCommandGraph;
  roads: GPUNetworkTopology;
  reverseRoads: GPUNetworkTopology;
  pivotNodes: GraphDataView<'uint32'>;
  reset: GraphDataView<'uint32'>;
  converged: GraphDataView<'uint32'>;
};

/** Builds the contributor shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(
  _device: Device,
  input: NetworkCentralityScheduleStepExampleInput
) {
  const {graph, roads, reverseRoads, pivotNodes, reset, converged} = input;
  const pivotCount = pivotNodes.length;
  // #region example
  const centrality = new GPUNetworkCentralityScheduleStep({
    topology: roads,
    reverseTopology: reverseRoads,
    pivotNodes,
    reset,
    laneCount: pivotCount,
    maximumIterations: 32,
    output: {
      closeness: createTransientView(graph, 'closeness', 'float32', pivotCount),
      reachedCounts: createTransientView(graph, 'reachedCounts', 'uint32', pivotCount),
      status: {converged}
    }
  });
  graph.add(centrality);
  const {closeness, reachedCounts} = centrality.props.output;
  // #endregion example
  return {
    centrality,
    closeness: closeness!,
    reachedCounts: reachedCounts!,
    converged: centrality.props.output.status!.converged!
  };
}
