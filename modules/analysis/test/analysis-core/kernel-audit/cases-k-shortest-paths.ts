// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUKShortestPaths} from '../../../src/gpu-network/k-shortest-paths/index';
import type {KernelAuditCase} from './kernel-audit-context';

export const K_SHORTEST_PATHS_AUDIT_CASES: readonly KernelAuditCase[] = [
  {
    nodeCount: 10000,
    maximumPathLength: 16,
    routeCount: 4,
    candidateCapacity: 256,
    maximumIterations: 2
  },
  {
    nodeCount: 2,
    maximumPathLength: 1024,
    routeCount: 32,
    candidateCapacity: 4096,
    maximumIterations: 1
  },
  {
    nodeCount: 300,
    maximumPathLength: 300,
    routeCount: 1,
    candidateCapacity: 1,
    maximumIterations: 1024
  }
].map(options => ({
  exportName: 'GPUKShortestPaths',
  name: `${options.nodeCount} vertices, ${options.maximumPathLength} path slots, k ${options.routeCount}, ${options.candidateCapacity} candidates, ${options.maximumIterations} shortest rounds`,
  build: ({graph, view}) =>
    graph.add(
      new GPUKShortestPaths({
        ...options,
        topology: {
          offsets: view('uint32', options.nodeCount + 1),
          neighbors: view('uint32', options.nodeCount * 2),
          weights: view('float32', options.nodeCount * 2)
        },
        sources: view('uint32', 1),
        targets: view('uint32', 1),
        output: {
          vertices: view('uint32', options.routeCount * options.maximumPathLength),
          edges: view('uint32', options.routeCount * options.maximumPathLength),
          lengths: view('uint32', options.routeCount),
          costs: view('float32', options.routeCount),
          counts: view('uint32', 1),
          status: {
            converged: view('uint32', 1),
            iterationLimitReached: view('uint32', 1),
            candidateOverflow: view('uint32', 1),
            iterationCount: view('uint32', 1)
          }
        }
      })
    )
}));
