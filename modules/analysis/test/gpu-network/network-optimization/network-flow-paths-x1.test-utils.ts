// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {NetworkFlowOperationProps} from '../../../src/gpu-network/network-optimization/gpu-network-flow-operation';

export function getEmptyFlowOperationProps(
  solver: 'reference' | 'parallel'
): NetworkFlowOperationProps {
  return {
    id: 'empty-flow',
    solver,
    nodeCount: 2,
    edgeCount: 0,
    pathCapacity: 1,
    edges: 'edges',
    capacities: 'capacities',
    parameters: 'parameters',
    output: {
      flow: 'flow',
      cut: 'cut',
      summary: 'summary',
      status: {
        invalidCount: 'invalid',
        converged: 'converged',
        iterationCount: 'iterations',
        iterationLimitReached: 'limited'
      },
      paths: {
        edgeRows: 'pathEdges',
        lengths: 'pathLengths',
        volumes: 'pathVolumes',
        status: {
          count: 'pathCount',
          requiredCount: 'pathRequired',
          overflow: 'pathOverflow',
          invalidCount: 'pathInvalid',
          converged: 'pathConverged',
          iterationLimitReached: 'pathLimited'
        }
      }
    }
  };
}

/** Hand-worked oracle: no route can serve the unit request; exhaustion is a complete solve. */
export function getEmptyFlowExpectedRows(): Record<string, Uint32Array> {
  return {
    flow: new Uint32Array(0),
    cut: Uint32Array.of(1, 0),
    summary: Uint32Array.of(1, 1, 0, 1, 0, 0, 0, 1),
    invalid: Uint32Array.of(0),
    converged: Uint32Array.of(1),
    iterations: Uint32Array.of(0),
    limited: Uint32Array.of(0),
    pathEdges: Uint32Array.of(0xffffffff),
    pathLengths: Uint32Array.of(0),
    pathVolumes: Uint32Array.of(0),
    pathCount: Uint32Array.of(0),
    pathRequired: Uint32Array.of(0),
    pathOverflow: Uint32Array.of(0),
    pathInvalid: Uint32Array.of(0),
    pathConverged: Uint32Array.of(1),
    pathLimited: Uint32Array.of(0)
  };
}
