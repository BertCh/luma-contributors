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
  GPUNetworkCoarsening,
  type GPUNetworkTopology
} from '@luma-contributors/analysis/gpu-network';

/** Uploaded inputs: the roads (with node positions) and one dense zone label per intersection. */
export type NetworkCoarseningExampleInput = {
  graph: GPUCommandGraph;
  roads: GPUNetworkTopology;
  zones: GraphDataView<'uint32'>;
  groupCapacity: number;
};

/** Builds the contributor shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(_device: Device, input: NetworkCoarseningExampleInput) {
  const {graph, roads, zones, groupCapacity} = input;
  const superedgeCapacity = groupCapacity * groupCapacity;
  // #region example
  const coarsening = new GPUNetworkCoarsening({
    topology: roads,
    labels: zones,
    directed: true,
    groupCapacity,
    output: {
      groupNodeCount: createTransientView(graph, 'groupNodeCount', 'uint32', groupCapacity),
      groupCentroid: createTransientView(graph, 'groupCentroid', 'float32x2', groupCapacity),
      edges: {
        ids: createTransientView(graph, 'sourceGroups', 'uint32', superedgeCapacity),
        status: {
          count: createTransientView(graph, 'superedgeCount', 'uint32', 1),
          overflow: createTransientView(graph, 'overflow', 'uint32', 1),
          candidateOverflow: createTransientView(graph, 'candidateOverflow', 'uint32', 1)
        }
      },
      edgeTargetGroups: createTransientView(graph, 'targetGroups', 'uint32', superedgeCapacity),
      edgeCounts: createTransientView(graph, 'edgeCounts', 'uint32', superedgeCapacity)
    }
  });
  graph.add(coarsening);
  const {groupNodeCount, groupCentroid, edges, edgeTargetGroups, edgeCounts} =
    coarsening.props.output;
  // #endregion example
  return {
    coarsening,
    groupNodeCount,
    groupCentroid: groupCentroid!,
    sourceGroups: edges.ids,
    targetGroups: edgeTargetGroups,
    edgeCounts,
    superedgeCount: edges.status.count!,
    overflow: edges.status.overflow,
    candidateOverflow: edges.status.candidateOverflow!
  };
}

/** Refuse capture of a quotient graph that lost live nodes/edges or final superedge rows. */
export function validateNetworkCoarseningCapture(result: {
  overflow: number;
  candidateOverflow: number;
  groupNodeCount: ArrayLike<number>;
}): void {
  if (result.overflow !== 0)
    throw new Error('Network coarsening overflowed its superedge capacity');
  if (result.candidateOverflow !== 0)
    throw new Error('Network coarsening excluded labels or edges (candidateOverflow)');
  if (Array.from(result.groupNodeCount).some(nodes => nodes === 0))
    throw new Error('Network coarsening: a zone is empty');
}
