// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUMapMatchedEdgeVolumes} from '../../../src/gpu-network/flow-aggregation/index';
import {
  GPUNetworkPotentialPathArea,
  GPUNetworkTravelTimePercentiles
} from '../../../src/gpu-network/network-time-geography/index';
import type {KernelAuditCase} from './kernel-audit-context';

/** Kernel-audit cases of the edge volumes (MOB-05) and the time-geography contributors (MOB-09). */
export const MOB05_MOB09_AUDIT_CASES: readonly KernelAuditCase[] = [
  ...[false, true].map(
    (transitions): KernelAuditCase => ({
      exportName: 'GPUMapMatchedEdgeVolumes',
      name: `4096 tracks of 64 entries, 60000 edges, transitions=${transitions}`,
      build: ({graph, view}) => {
        const edgeCount = transitions ? 60000 : 1_000_000;
        const flowCapacity = 256;
        graph.add(
          new GPUMapMatchedEdgeVolumes({
            traversedEdges: {
              edges: view('uint32', 4096 * 64),
              counts: view('uint32', 4096),
              completeness: 'known-complete'
            },
            trackCount: 4096,
            traversedEdgeCapacity: 64,
            edgeCount,
            transitionPairCapacity: transitions ? 65536 : undefined,
            output: {
              edgeVolumes: view('uint32', edgeCount),
              status: {candidateOverflow: view('uint32', 1)},
              transitions: transitions
                ? {
                    ids: view('uint32', flowCapacity),
                    flowCounts: view('uint32', flowCapacity),
                    zoneOutCounts: view('uint32', edgeCount),
                    zoneInCounts: view('uint32', edgeCount),
                    status: {
                      count: view('uint32', 1),
                      overflow: view('uint32', 1),
                      requiredCount: view('uint32', 1)
                    }
                  }
                : undefined
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUNetworkPotentialPathArea',
    name: 'one million nodes and edges',
    build: ({graph, view}) =>
      graph.add(
        new GPUNetworkPotentialPathArea({
          topology: {
            offsets: view('uint32', 1_000_001),
            neighbors: view('uint32', 4_000_000),
            weights: view('float32', 4_000_000)
          },
          forwardCosts: view('float32', 1_000_000),
          reverseCosts: view('float32', 1_000_000),
          budget: view('float32', 1),
          output: {
            edgeInPrism: view('uint32', 4_000_000),
            edgeSlack: view('float32', 4_000_000),
            nodeInPrism: view('uint32', 1_000_000),
            summary: view('uint32', 2)
          }
        })
      )
  },
  {
    exportName: 'GPUNetworkTravelTimePercentiles',
    name: '60 departures, 256 states, 4096 transitions, 16 percentiles',
    build: ({graph, view}) =>
      graph.add(
        new GPUNetworkTravelTimePercentiles({
          stateVertices: view('uint32', 256),
          transitions: view('uint32x2', 4096),
          costs: view('float32x2', 4096),
          initialState: view('uint32', 1),
          vertexCount: 256,
          departureCount: 60,
          percentiles: Array.from({length: 16}, (_, index) => (index + 1) * 6),
          window: view('float32', 2),
          output: {
            travelTimes: view('float32', 256 * 16),
            reachedCounts: view('uint32', 256),
            status: view('uint32', 120)
          }
        })
      )
  }
];
