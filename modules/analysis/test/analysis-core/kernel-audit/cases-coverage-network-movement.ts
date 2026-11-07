// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUNetworkFlow,
  GPUNetworkLouvain,
  GPUNetworkPointRouting,
  GPUNetworkTimeRouting,
  GPUNetworkFacilityOptimization,
  GPUNetworkRationalAccess
} from '../../../src/gpu-network/network-optimization/index';
import {GPUTrajectoryKalman} from '../../../src/gpu-spatial-analysis/trajectory-estimation/index';
import {GPUTrajectoryRestriction} from '../../../src/gpu-spatial-analysis/trajectory-restriction/index';
import {GPUTrackSimilaritySearch} from '../../../src/gpu-spatial-analysis/track-similarity/index';
import type {KernelAuditCase} from './kernel-audit-context';

export const COVERAGE_NETWORK_MOVEMENT_CASES: readonly KernelAuditCase[] = [
  {
    exportName: 'GPUTrajectoryRestriction',
    name: 'pairwise temporal distance and bounded intervals',
    build: ({graph, view}) =>
      graph.add(
        new GPUTrajectoryRestriction({
          positions: view('float32x2', 1024),
          times: view('float32', 1024),
          offsets: view('uint32', 129),
          parameters: view('float32', 7),
          pairwiseTracks: [
            [0, 1],
            [2, 3]
          ],
          intervalsPerPair: 65536,
          geometryBoundary: 'open',
          timeBoundary: 'right-closed',
          output: {
            intervals: view('float32x2', 131072),
            positions: view('float32x4', 131072),
            status: view('uint32', 131072),
            relationships: view('uint32x4', 2)
          }
        })
      )
  },
  {
    exportName: 'GPUNetworkRationalAccess',
    name: 'fractional demand and absolute shifts',
    build: ({graph, view}) =>
      graph.add(
        new GPUNetworkRationalAccess({
          demandMode: 'fractional',
          stepMode: 'absolute',
          costs: view('float32', 2048),
          demand: view('float32', 64),
          supply: view('float32', 32),
          parameters: view('float32', 6),
          output: {
            assignments: view('float32', 2048),
            scores: view('float32', 64),
            unservedDemand: view('float32', 64),
            status: view('uint32', 4)
          }
        })
      )
  },
  {
    exportName: 'GPUNetworkFlow',
    name: 'exact integer objective words',
    build: ({graph, view}) =>
      graph.add(
        new GPUNetworkFlow({
          nodeCount: 128,
          edges: view('uint32x2', 1024),
          capacities: view('uint32', 1024),
          costs: view('uint32', 1024),
          parameters: view('uint32', 3),
          output: {
            flow: view('uint32', 1024),
            cut: view('uint32', 128),
            summary: view('uint32', 8),
            exactCost: view('uint32x2', 1)
          }
        })
      )
  },
  {
    exportName: 'GPUNetworkLouvain',
    name: 'directed seeded modularity',
    build: ({graph, view}) =>
      graph.add(
        new GPUNetworkLouvain({
          nodeCount: 32,
          directed: true,
          seed: 0xffffffff,
          weights: view('float32', 1024),
          parameters: view('float32', 2),
          output: {
            communities: view('uint32', 32),
            objectiveTrace: view('float32', 129),
            summary: view('uint32', 4)
          }
        })
      )
  },
  {
    exportName: 'GPUNetworkTimeRouting',
    name: '256 samples per transition, FIFO waiting',
    build: ({graph, view}) =>
      graph.add(
        new GPUNetworkTimeRouting({
          stateVertices: view('uint32', 256),
          transitions: view('uint32x2', 4096),
          costs: view('float32x2', 4096 * 256),
          costModel: 'piecewise-arrival',
          samplesPerTransition: 256,
          waitingPolicy: 'wait-until-first-sample',
          endpoints: view('uint32', 2),
          departure: view('float32', 1),
          output: {
            arrivals: view('float32', 256),
            predecessorTransitions: view('uint32', 256),
            summary: view('uint32', 4)
          }
        })
      )
  },
  {
    exportName: 'GPUTrajectoryKalman',
    name: 'full correlated covariance and pivoted RTS',
    build: ({graph, view}) =>
      graph.add(
        new GPUTrajectoryKalman({
          covarianceMode: 'full',
          positions: view('float32x2', 1024),
          times: view('float32', 1024),
          offsets: view('uint32', 129),
          parameters: view('float32', 10),
          output: {
            filteredState: view('float32x4', 1024),
            filteredCovariance: view('float32x4', 4096),
            smoothedState: view('float32x4', 1024),
            smoothedCovariance: view('float32x4', 4096),
            status: view('uint32', 1024)
          }
        })
      )
  },
  {
    exportName: 'GPUTrajectoryRestriction',
    name: '128 polygon vertices,256 slots,open/half-open,merged compaction',
    build: ({graph, view}) =>
      graph.add(
        new GPUTrajectoryRestriction({
          positions: view('float32x2', 1024),
          times: view('float32', 1024),
          offsets: view('uint32', 129),
          parameters: view('float32', 263),
          polygonRingVertexCounts: [64, 64],
          intervalsPerSegment: 256,
          geometryBoundary: 'open',
          timeBoundary: 'left-closed',
          mergeIntervals: true,
          output: {
            intervals: view('float32x2', 262144),
            positions: view('float32x4', 262144),
            status: view('uint32', 262144),
            relationships: view('uint32x4', 128),
            compactIntervals: view('float32x2', 1024),
            compactPositions: view('float32x4', 1024),
            compactSources: view('uint32x2', 1024),
            compactStatus: view('uint32', 1024),
            compactSummary: view('uint32', 3)
          }
        })
      )
  },
  {
    exportName: 'GPUTrackSimilaritySearch',
    name: 'bounded survivor compaction and exact top1024',
    build: ({graph, view}) =>
      graph.add(
        new GPUTrackSimilaritySearch({
          linesA: {
            kind: 'lines',
            positions: view('float32x2', 1024),
            lineOffsets: view('uint32', 129)
          },
          pairA: view('uint32', 4096),
          pairB: view('uint32', 4096),
          threshold: view('float32', 1),
          metric: 'frechet',
          survivorCapacity: 1024,
          topK: 1024,
          output: {
            distance: view('float32', 4096),
            lowerBound: view('float32', 4096),
            matched: view('uint32', 4096),
            status: view('uint32', 4096),
            survivorSummary: view('uint32', 3),
            topKIndices: view('uint32', 131072),
            topKSummary: view('uint32x2', 128)
          }
        })
      )
  },
  ...[false, true].map(
    (costs): KernelAuditCase => ({
      exportName: 'GPUNetworkFlow',
      name: `128 nodes,1024 edges,65536 augmentations,costs=${costs}`,
      build: ({graph, view}) =>
        graph.add(
          new GPUNetworkFlow({
            nodeCount: 128,
            maximumAugmentations: 65536,
            edges: view('uint32x2', 1024),
            capacities: view('uint32', 1024),
            costs: costs ? view('uint32', 1024) : undefined,
            parameters: view('uint32', 3),
            output: {
              flow: view('uint32', 1024),
              cut: view('uint32', 128),
              summary: view('uint32', 8)
            }
          })
        )
    })
  ),
  {
    exportName: 'GPUNetworkLouvain',
    name: '32 vertices,32 levels,64 iterations',
    build: ({graph, view}) =>
      graph.add(
        new GPUNetworkLouvain({
          nodeCount: 32,
          maximumLevels: 32,
          maximumIterations: 64,
          weights: view('float32', 1024),
          parameters: view('float32', 2),
          output: {
            communities: view('uint32', 32),
            objectiveTrace: view('float32', 2049),
            summary: view('uint32', 4)
          }
        })
      )
  },
  {
    exportName: 'GPUNetworkPointRouting',
    name: '128 edges,4096 turns,binding-limited queries,65536 relaxations',
    build: ({graph, view, maximumRows}) => {
      const queryCount = maximumRows(128 * 4);
      graph.add(
        new GPUNetworkPointRouting({
          maximumRelaxations: 65536,
          costs: view('float32', 128),
          turns: view('uint32x2', 4096),
          queries: view('uint32x2', queryCount),
          fractions: view('float32x2', queryCount),
          output: {
            costs: view('float32', queryCount),
            status: view('uint32', queryCount),
            predecessorEdges: view('uint32', queryCount * 128)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUNetworkTimeRouting',
    name: '256 states,4096 transitions,65536 relaxations',
    build: ({graph, view}) =>
      graph.add(
        new GPUNetworkTimeRouting({
          maximumRelaxations: 65536,
          stateVertices: view('uint32', 256),
          transitions: view('uint32x2', 4096),
          costs: view('float32x2', 4096),
          endpoints: view('uint32', 2),
          departure: view('float32', 1),
          output: {
            arrivals: view('float32', 256),
            predecessorTransitions: view('uint32', 256),
            summary: view('uint32', 4)
          }
        })
      )
  },
  ...(['backup-coverage', 'p-dispersion', 'capacitated-p-median'] as const).map(
    (operation): KernelAuditCase => ({
      exportName: 'GPUNetworkFacilityOptimization',
      name: `16 facilities,maximal bounded ${operation}`,
      build: ({graph, view}) => {
        const demandCount = operation === 'capacitated-p-median' ? 5 : 64;
        graph.add(
          new GPUNetworkFacilityOptimization({
            operation,
            facilityCount: 16,
            selectCount: operation === 'backup-coverage' ? undefined : 16,
            maximumCandidates: 1048576,
            costs: view('float32', 16 * (operation === 'p-dispersion' ? 16 : demandCount)),
            demand: view('float32', demandCount),
            capacities: view('float32', 16),
            threshold: view('float32', 1),
            output: {
              selectedFacilities: view('uint32', 16),
              assignments: view('uint32', demandCount),
              objective: view('float32', 1),
              status: view('uint32', 4)
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUNetworkRationalAccess',
    name: '64 origins,32 facilities,4096 cycles and initial limiting',
    build: ({graph, view}) =>
      graph.add(
        new GPUNetworkRationalAccess({
          costs: view('float32', 2048),
          demand: view('float32', 64),
          supply: view('float32', 32),
          parameters: view('float32', 6),
          maximumCycles: 4096,
          initialLimitCycles: 4096,
          output: {
            assignments: view('float32', 2048),
            scores: view('float32', 64),
            unservedDemand: view('float32', 64),
            status: view('uint32', 4)
          }
        })
      )
  },
  {
    exportName: 'GPUTrajectoryKalman',
    name: 'binding-limited observations,filter and smoother',
    build: ({graph, view, maximumRows}) => {
      const observationCount = maximumRows(16);
      graph.add(
        new GPUTrajectoryKalman({
          positions: view('float32x2', observationCount),
          times: view('float32', observationCount),
          offsets: view('uint32', 2),
          parameters: view('float32', 4),
          output: {
            filteredState: view('float32x4', observationCount),
            filteredCovariance: view('float32x4', observationCount),
            smoothedState: view('float32x4', observationCount),
            smoothedCovariance: view('float32x4', observationCount),
            status: view('uint32', observationCount)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUTrajectoryRestriction',
    name: 'binding-limited observations,all interval and relationship outputs',
    build: ({graph, view, maximumRows}) => {
      const observationCount = maximumRows(16);
      graph.add(
        new GPUTrajectoryRestriction({
          positions: view('float32x2', observationCount),
          times: view('float32', observationCount),
          offsets: view('uint32', 2),
          parameters: view('float32', 7),
          output: {
            positions: view('float32x4', observationCount),
            intervals: view('float32x2', observationCount),
            status: view('uint32', observationCount),
            relationships: view('uint32x4', 1)
          }
        })
      );
    }
  },
  ...(['dynamic-time-warping', 'frechet', 'longest-common-subsequence'] as const).map(
    (metric): KernelAuditCase => ({
      exportName: 'GPUTrackSimilaritySearch',
      name: `${metric},2048 DP columns,binding-limited pair scratch,maximal bands`,
      build: ({graph, view, maximumRows}) => {
        const pairCount = maximumRows(2048 * 4);
        const subsequence = metric === 'longest-common-subsequence';
        graph.add(
          new GPUTrackSimilaritySearch({
            metric,
            maximumFrechetVertexCount: 2048,
            dynamicTimeWarpingBandRadius: 2 ** 30,
            lcssBandRadius: 2 ** 30,
            lcssTolerance: 1000,
            linesA: {
              kind: 'lines',
              positions: view('float32x2', 4096),
              lineOffsets: view('uint32', 3)
            },
            linesB: {
              kind: 'lines',
              positions: view('float32x2', 4096),
              lineOffsets: view('uint32', 3)
            },
            pairA: view('uint32', pairCount),
            pairB: view('uint32', pairCount),
            threshold: view('float32', 1),
            output: {
              distance: subsequence ? undefined : view('float32', pairCount),
              lowerBound: subsequence ? undefined : view('float32', pairCount),
              similarity: subsequence ? view('float32', pairCount) : undefined,
              upperBound: subsequence ? view('float32', pairCount) : undefined,
              matched: view('uint32', pairCount),
              status: view('uint32', pairCount)
            }
          })
        );
      }
    })
  )
];
