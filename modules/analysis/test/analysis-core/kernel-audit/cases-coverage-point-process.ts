// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUInhomogeneousPointPattern} from '../../../src/gpu-spatial-analysis/point-process/gpu-inhomogeneous-point-pattern';
import {GPUPointPatternEnvelope} from '../../../src/gpu-spatial-analysis/point-process/gpu-point-pattern-envelope';
import {GPUSpatialBalancedSample} from '../../../src/gpu-spatial-analysis/point-process/gpu-spatial-balanced-sample';
import {GPUOPTICS} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics';
import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window/gpu-study-window-preparation';
import {GPUNetworkKernelDensity} from '../../../src/gpu-network/gpu-network-kernel-density';
import {GPUNetworkInhomogeneousK} from '../../../src/gpu-network/gpu-network-inhomogeneous-k';
import type {KernelAuditCase} from './kernel-audit-context';

const cases: KernelAuditCase[] = [];
for (const intensity of ['supplied', 'include', 'exclude'] as const) {
  cases.push({
    exportName: 'GPUInhomogeneousPointPattern',
    name: `4096 events,256 radii,${intensity} intensity,polygon edges`,
    build({graph, view}) {
      const studyWindow = new GPUStudyWindowPreparation({
        positions: view('float32x2', 1024),
        ringOffsets: view('uint32', 17),
        polygonOffsets: view('uint32', 9),
        featureOffsets: view('uint32', 2),
        samplingGrid: [32, 32]
      });
      graph.add(studyWindow);
      graph.add(
        new GPUInhomogeneousPointPattern({
          positions: view('float32x2', 4096),
          studyWindow,
          radii: view('float32', 256),
          ...(intensity === 'supplied'
            ? {suppliedIntensity: view('float32', 4096)}
            : {bandwidth: 4, selfContribution: intensity}),
          output: {
            intensity: view('float32', 4096),
            k: view('float32', 256),
            pairCorrelation: view('float32', 256),
            status: view('uint32', 256)
          }
        })
      );
    }
  });
}
for (const statistic of ['maximum-deviation', 'integrated-squared-deviation'] as const) {
  cases.push({
    exportName: 'GPUPointPatternEnvelope',
    name: `4096 radii,4096 simulations,rank2048,${statistic}`,
    build({graph, view}) {
      graph.add(
        new GPUPointPatternEnvelope({
          observed: view('float32', 4096),
          simulations: view('float32', 16777216),
          reference: view('float32', 4096),
          integrationWeights: view('float32', 4096),
          simulationCount: 4096,
          rank: 2048,
          statistic,
          nullModel: 'Audit supplied simulation curves',
          fittedParameters: 'refit-each-replicate',
          output: {
            lower: view('float32', 4096),
            upper: view('float32', 4096),
            statistics: view('float32', 4),
            status: view('uint32', 1)
          }
        })
      );
    }
  });
}
cases.push({
  exportName: 'GPUSpatialBalancedSample',
  name: '4096 frame rows,256 strata,all optional resident inputs,4096 sample slots',
  build({graph, view}) {
    graph.add(
      new GPUSpatialBalancedSample({
        positions: view('float32x2', 4096),
        frameIds: view('uint32', 4096),
        strata: view('uint32', 4096),
        inclusionTickets: view('uint32', 4096),
        parameters: view('uint32', 2),
        allocations: Array.from({length: 256}, (_, stratumId) => ({stratumId, sampleCount: 16})),
        bounds: [0, 0, 1, 1],
        seed: 0xffffffff,
        replicate: 0xffffffff,
        output: {
          frameIds: view('uint32', 4096),
          inclusionProbabilities: view('float32', 4096),
          designWeights: view('float32', 4096),
          status: view('uint32', 512)
        }
      })
    );
  }
});
cases.push({
  exportName: 'GPUOPTICS',
  name: '4096 rows,256 minimum samples,4096 visits,all extraction outputs',
  build({graph, view}) {
    graph.add(
      new GPUOPTICS({
        positions: view('float32x2', 4096),
        minimumSamples: 256,
        maximumIterations: 4096,
        maximumDistance: 10,
        extractionDistance: 10,
        output: {
          ordering: view('uint32', 4096),
          coreDistances: view('float32', 4096),
          reachability: view('float32', 4096),
          predecessors: view('uint32', 4096),
          labels: view('uint32', 4096),
          ports: {
            status: {
              iterationCount: view('uint32', 1),
              converged: view('uint32', 1),
              invalidCount: view('uint32', 1)
            }
          }
        }
      })
    );
  }
});
cases.push({
  exportName: 'GPUOPTICS',
  name: '65536 rows,grid reachability,256 minimum samples,256-visit batches,candidate overflow',
  build({graph, view}) {
    graph.add(
      new GPUOPTICS({
        positions: view('float32x2', 65536),
        minimumSamples: 256,
        maximumDistance: 10,
        extractionDistance: 10,
        frontierBatchSize: 256,
        neighborIndex: 'grid',
        neighborCapacity: 65536 * 256,
        output: {
          ordering: view('uint32', 65536),
          coreDistances: view('float32', 65536),
          reachability: view('float32', 65536),
          predecessors: view('uint32', 65536),
          labels: view('uint32', 65536),
          ports: {
            status: {
              candidateOverflow: view('uint32', 1),
              iterationCount: view('uint32', 1),
              converged: view('uint32', 1),
              invalidCount: view('uint32', 1)
            }
          }
        }
      })
    );
  }
});
for (const [eventCount, queryCount] of [
  [65536, 256],
  [256, 65536]
]) {
  cases.push({
    exportName: 'GPUNetworkKernelDensity',
    name: `${eventCount} events,${queryCount} queries,maximum product,leave-one-out`,
    build({graph, view}) {
      graph.add(
        new GPUNetworkKernelDensity({
          eventCount,
          distances: view('float32', eventCount * queryCount),
          quadratureLengths: view('float32', queryCount),
          excludedEventIds: view('uint32', queryCount),
          bandwidth: 2,
          output: {
            intensity: view('float32', queryCount),
            kernelMass: view('float32', eventCount),
            validity: view('uint32', queryCount)
          }
        })
      );
    }
  });
}
for (const correction of ['none', 'ang'] as const) {
  for (const normalization of [
    'network-length',
    'inverse-intensity',
    'inverse-intensity-squared'
  ] as const) {
    for (const [eventCount, nodeCount] of [
      [1024, 16384],
      [256, 65536]
    ]) {
      cases.push({
        exportName: 'GPUNetworkInhomogeneousK',
        name: `${eventCount} events,${nodeCount} nodes,65536 edges,256 radii,${correction},${normalization}`,
        build({graph, view}) {
          graph.add(
            new GPUNetworkInhomogeneousK({
              eventNodeIds: view('uint32', eventCount),
              intensity: view('float32', eventCount),
              nodeDistances: view('float32', eventCount * nodeCount),
              nodeCount,
              edgeNodes: view('uint32x2', 65536),
              edgeLengths: view('float32', 65536),
              radii: view('float32', 256),
              correction,
              normalization,
              maximumPairCount: 1048576,
              output: {
                k: view('float32', 256),
                numerator: view('float32', 256),
                normalization: view('float32', 3),
                status: view('uint32', 256)
              }
            })
          );
        }
      });
    }
  }
}
export const COVERAGE_POINT_PROCESS_CASES: readonly KernelAuditCase[] = cases;

cases.push({
  exportName: 'GPUPointPatternEnvelope',
  name: '4095 simulations,one ordinate,maximum extreme-rank comparison budget,p interval',
  build({graph, view}) {
    graph.add(
      new GPUPointPatternEnvelope({
        observed: view('float32', 1),
        simulations: view('float32', 4095),
        reference: view('float32', 1),
        integrationWeights: view('float32', 1),
        simulationCount: 4095,
        statistic: 'extreme-rank',
        significanceLevel: 0.05,
        nullModel: 'Audit supplied curves',
        fittedParameters: 'none',
        output: {
          lower: view('float32', 1),
          upper: view('float32', 1),
          statistics: view('float32', 4),
          status: view('uint32', 1),
          pValueInterval: view('float32', 2)
        }
      })
    );
  }
});
for (const predecessorCorrection of [false, true]) {
  cases.push({
    exportName: 'GPUOPTICS',
    name: `4096 rows,xi,all hierarchy outputs,predecessor correction${predecessorCorrection},parallel frontier`,
    build({graph, view}) {
      graph.add(
        new GPUOPTICS({
          positions: view('float32x2', 4096),
          minimumSamples: 256,
          maximumDistance: 10,
          maximumIterations: 4096,
          frontierBatchSize: 256,
          extractionMethod: 'xi',
          xi: 0.1,
          minimumClusterSize: 2,
          predecessorCorrection,
          output: {
            ordering: view('uint32', 4096),
            coreDistances: view('float32', 4096),
            reachability: view('float32', 4096),
            predecessors: view('uint32', 4096),
            labels: view('uint32', 4096),
            ports: {
              status: {
                iterationCount: view('uint32', 1),
                converged: view('uint32', 1),
                invalidCount: view('uint32', 1)
              }
            },
            clusterIntervals: view('uint32x2', 4096),
            clusterSummary: view('uint32', 2)
          }
        })
      );
    }
  });
}

for (const divisor of ['radius', 'distance'] as const) {
  cases.push({
    exportName: 'GPUInhomogeneousPointPattern',
    name: `4096 events,256 radii,translation,1024 vertices,kernel ${divisor},reflection,maximum edge budget`,
    build({graph, view}) {
      const studyWindow = new GPUStudyWindowPreparation({
        positions: view('float32x2', 1024),
        ringOffsets: view('uint32', 17),
        polygonOffsets: view('uint32', 9),
        featureOffsets: view('uint32', 2),
        samplingGrid: [32, 32]
      });
      graph.add(studyWindow);
      graph.add(
        new GPUInhomogeneousPointPattern({
          positions: view('float32x2', 4096),
          studyWindow,
          radii: view('float32', 256),
          suppliedIntensity: view('float32', 4096),
          edgeCorrection: 'translation',
          maximumOverlapEdgePairs: 1048576,
          pairCorrelation: {
            kernel: 'epanechnikov',
            bandwidth: 4,
            divisor,
            zeroCorrection: 'reflection'
          },
          output: {
            intensity: view('float32', 4096),
            k: view('float32', 256),
            pairCorrelation: view('float32', 256),
            status: view('uint32', 256)
          }
        })
      );
    }
  });
}
