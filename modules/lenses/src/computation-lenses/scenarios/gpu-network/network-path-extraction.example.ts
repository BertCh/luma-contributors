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
  GPUNetworkPathExtraction,
  GPUNetworkReachability,
  type GPUNetworkTopology
} from '@luma-contributors/analysis/gpu-network';

/** Uploaded inputs: the road network, one start intersection and one target intersection. */
export type NetworkPathExtractionExampleInput = {
  graph: GPUCommandGraph;
  roads: GPUNetworkTopology;
  startNodes: GraphDataView<'uint32'>;
  targets: GraphDataView<'uint32'>;
};

/** Builds the contributor shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(_device: Device, input: NetworkPathExtractionExampleInput) {
  const {graph, roads, startNodes, targets} = input;
  const nodeCount = roads.offsets.length - 1;
  const converged = createTransientView(graph, 'converged', 'uint32', 1);
  const costs = createTransientView(graph, 'costs', 'float32', nodeCount);
  // #region example
  const predecessors = createTransientView(graph, 'predecessors', 'uint32', nodeCount);
  const reachability = new GPUNetworkReachability({
    topology: roads,
    startNodes,
    output: {costs, predecessors, status: {converged}}
  });
  const paths = new GPUNetworkPathExtraction({
    predecessors,
    costs,
    targets,
    predecessorsComplete: converged,
    output: {
      ids: createTransientView(graph, 'ids', 'uint32', nodeCount),
      nodeRows: createTransientView(graph, 'nodeRows', 'uint32', nodeCount),
      lineOffsets: createTransientView(graph, 'lineOffsets', 'uint32', 2),
      pathFound: createTransientView(graph, 'pathFound', 'uint32', 1),
      status: {
        count: createTransientView(graph, 'count', 'uint32', 1),
        overflow: createTransientView(graph, 'overflow', 'uint32', 1),
        candidateOverflow: createTransientView(graph, 'candidateOverflow', 'uint32', 1)
      }
    }
  });
  graph.add([reachability, paths]);
  const {ids, nodeRows, lineOffsets, pathFound} = paths.props.output;
  // #endregion example
  return {
    paths,
    costs,
    predecessors,
    converged,
    ids,
    nodeRows: nodeRows!,
    lineOffsets: lineOffsets!,
    pathFound: pathFound!
  };
}
