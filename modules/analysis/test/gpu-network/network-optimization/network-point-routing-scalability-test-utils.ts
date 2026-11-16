// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  GPUNetworkPointRouting,
  type GPUNetworkPointRoutingProps
} from '../../../src/gpu-network/network-optimization/gpu-network-point-routing';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

/** Declaration-only routing graph for limits, kernel parsing and command-budget checks. */
export function setupPointRouting(
  options: {
    edgeCount?: number;
    turnCount?: number;
    queryCount?: number;
    maximumRelaxations?: number;
    mode?: GPUNetworkPointRoutingProps['mode'];
    device?: Device;
  } = {}
) {
  const edgeCount = options.edgeCount ?? 300;
  const turnCount = options.turnCount ?? edgeCount * 4;
  const queryCount = options.queryCount ?? 2;
  const graph = new GPUCommandGraph(options.device ?? createNullWebGPUDevice());
  const props: GPUNetworkPointRoutingProps = {
    costs: createTransientView(graph, 'edge-costs', 'float32', edgeCount),
    turns: createTransientView(graph, 'turns', 'uint32x2', turnCount),
    queries: createTransientView(graph, 'queries', 'uint32x2', queryCount),
    fractions: createTransientView(graph, 'fractions', 'float32x2', queryCount),
    mode: options.mode,
    maximumRelaxations: options.maximumRelaxations,
    output: {
      costs: createTransientView(graph, 'costs', 'float32', queryCount),
      status: createTransientView(graph, 'status', 'uint32', queryCount),
      predecessorEdges: createTransientView(graph, 'predecessors', 'uint32', edgeCount * queryCount)
    }
  };
  return {graph, props, contributor: new GPUNetworkPointRouting(props)};
}

/** Four fixture points each contribute one point-edge pair to the global snapping candidate table. */
export const NETWORK_POINT_ROUTING_SNAPPING_CANDIDATE_CAPACITY = 4;
