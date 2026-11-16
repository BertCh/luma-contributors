// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUNetworkCentralityScheduleStep} from '../../../src/gpu-network/network-centrality/index';
import type {KernelAuditCase} from '../../analysis-core/kernel-audit/kernel-audit-context';

/** Ready for inclusion by the shared kernel-audit registry after the parallel pass. */
export const NETWORK_CENTRALITY_SCHEDULE_KERNEL_AUDIT_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUNetworkCentralityScheduleStep',
    name: 'maximum rounds and local iterations, all outputs, sampled pivots and radius',
    build({graph, view}) {
      const nodeCount = 1024;
      const edgeCount = 4096;
      const pivotCount = 2;
      const topology = () => ({
        offsets: view('uint32', nodeCount + 1),
        neighbors: view('uint32', edgeCount),
        weights: view('float32', edgeCount)
      });
      graph.add(
        new GPUNetworkCentralityScheduleStep({
          topology: topology(),
          reverseTopology: topology(),
          pivotNodes: view('uint32', pivotCount),
          reset: view('uint32', 1),
          costLimit: view('float32', 1),
          maximumIterations: 1024,
          localIterations: 64,
          laneCount: pivotCount,
          sampling: 'sampled',
          betweennessScale: 512,
          output: {
            betweenness: view('float32', nodeCount),
            edgeBetweenness: view('float32', edgeCount),
            closeness: view('float32', pivotCount),
            harmonicCloseness: view('float32', pivotCount),
            reachedCounts: view('uint32', pivotCount),
            status: {
              converged: view('uint32', 1),
              zeroWeightCycle: view('uint32', 1),
              iterationLimitReached: view('uint32', 1),
              approximation: view('uint32', 1)
            }
          }
        })
      );
    }
  }
];
