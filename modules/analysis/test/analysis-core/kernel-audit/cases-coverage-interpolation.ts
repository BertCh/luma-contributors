// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPULocalKriging,
  GPULocalRBF,
  GPUNaturalNeighbor,
  GPUConditionalGaussianSimulation
} from '../../../src/gpu-spatial-analysis/index';
import {GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-interpolation/prediction-covariance';
import {GPUSpaceTimeKriging} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-space-time-kriging';
import {getGPULocalKrigingParameterLength} from '../../../src/gpu-spatial-analysis/spatial-interpolation/prediction-covariance';
import {GPUVariogramModelFit} from '@luma-contributors/analysis/dataframe-analysis';
import {GAUSSIAN_RANDOM_FIELD_KERNEL_CASES} from './cases-gaussian-random-field';
import type {KernelAuditCase} from './kernel-audit-context';

const COVARIANCE = {model: 'exponential' as const, range: 1, sill: 1};

/** Maximal option combinations, including all optional output bindings and bounded private arrays. */
export const COVERAGE_INTERPOLATION_KERNEL_CASES: KernelAuditCase[] = [
  ...GAUSSIAN_RANDOM_FIELD_KERNEL_CASES,
  ...(['simple', 'ordinary', 'universal', 'external-drift', 'co-kriging'] as const).map(
    mode =>
      ({
        exportName: 'GPULocalKriging',
        name: `maximum-neighborhood-and-support-${mode}`,
        build({graph, view}) {
          graph.add(
            new GPULocalKriging({
              positions: view('float32x2', 16),
              observations: view('float32x4', 16),
              supports: view('float32x2', 256),
              neighborCount: 16,
              supportCount: 16,
              mode,
              covariance: COVARIANCE,
              sampleDrift: view('float32x4', 16),
              queryDrift: view('float32x4', 16),
              driftCount: 4,
              coregionalization: [1, 0, 0.4, 0.8],
              output: {
                values: view('float32x4', 16),
                status: view('uint32', 16),
                weights: view('float32', 256)
              }
            })
          );
        }
      }) satisfies KernelAuditCase
  ),
  ...(['simple', 'ordinary', 'universal', 'external-drift', 'co-kriging'] as const).map(
    mode =>
      ({
        exportName: 'GPULocalKriging',
        name: `per-frame-parameters-${mode}`,
        build({graph, view}) {
          const isCoKriging = mode === 'co-kriging';
          graph.add(
            new GPULocalKriging({
              positions: view('float32x2', 16),
              observations: view('float32x4', 16),
              supports: view('float32x2', 16),
              neighborCount: 16,
              mode,
              parameters: view(
                'float32',
                getGPULocalKrigingParameterLength(isCoKriging ? 4 : 1, isCoKriging ? 4 : 1)
              ),
              covarianceStructureCount: isCoKriging ? 4 : undefined,
              variableCount: isCoKriging ? 4 : undefined,
              sampleDrift: view('float32x4', 16),
              queryDrift: view('float32x4', 16),
              driftCount: 4,
              output: {
                values: view('float32x4', 16),
                status: view('uint32', 16)
              }
            })
          );
        }
      }) satisfies KernelAuditCase
  ),
  ...(['simple', 'external-drift', 'co-kriging'] as const).map(
    mode =>
      ({
        exportName: 'GPULocalKriging',
        name: `grid-sample-index-${mode}`,
        build({graph, view}) {
          graph.add(
            new GPULocalKriging({
              positions: view('float32x2', 16),
              observations: view('float32x4', 16),
              supports: view('float32x4', 64),
              neighborCount: 32,
              supportCount: 4,
              sampleIndex: 'grid',
              mode,
              covariance: COVARIANCE,
              sampleDrift: view('float32', 16 * 8),
              queryDrift: view('float32', 16 * 8),
              driftCount: 8,
              coregionalization: [1, 0, 0.4, 0.8],
              // External drift with weights would need a ninth storage binding.
              output: {
                values: view('float32x4', 16),
                status: view('uint32', 16),
                ...(mode === 'external-drift' ? {} : {weights: view('float32', 16 * 32)})
              }
            })
          );
        }
      }) satisfies KernelAuditCase
  ),
  ...(['metric', 'octant', 'staged'] as const).flatMap(neighborSearch =>
    (['simple', 'ordinary'] as const).map(
      mode =>
        ({
          exportName: 'GPUSpaceTimeKriging',
          name: `maximum-neighborhood-${neighborSearch}-${mode}`,
          build({graph, view}) {
            graph.add(
              new GPUSpaceTimeKriging({
                temporalContext: {
                  unit: 'second',
                  epoch: 0n,
                  encoding: 'float32-relative',
                  axis: {kind: 'irregular'}
                },
                positions: view('float32x2', 64),
                times: view('float32', 64),
                observations: view('float32x4', 64),
                queries: view('float32x2', 16),
                queryTimes: view('float32', 16),
                mode,
                neighborCount: 32,
                neighborSearch,
                neighborsPerOctant: 8,
                stagedCandidateCount: 64,
                parameters: view('float32', GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH),
                output: {values: view('float32x4', 16), status: view('uint32', 16)}
              })
            );
          }
        }) satisfies KernelAuditCase
    )
  ),
  ...([0.5, 1.5, 2.5] as const).map(
    smoothness =>
      ({
        exportName: 'GPULocalKriging',
        name: `matern-${smoothness}`,
        build({graph, view}) {
          graph.add(
            new GPULocalKriging({
              positions: view('float32x2', 16),
              observations: view('float32x4', 16),
              supports: view('float32x2', 16),
              neighborCount: 16,
              mode: 'ordinary',
              covariance: {model: 'matern', smoothness, range: 1, sill: 1},
              output: {
                values: view('float32x4', 16),
                status: view('uint32', 16),
                weights: view('float32', 256)
              }
            })
          );
        }
      }) satisfies KernelAuditCase
  ),
  ...(['thin-plate-spline', 'cubic', 'gaussian', 'linear'] as const).map(
    kernel =>
      ({
        exportName: 'GPULocalRBF',
        name: `maximum-neighborhood-affine-${kernel}`,
        build({graph, view}) {
          graph.add(
            new GPULocalRBF({
              positions: view('float32x2', 16),
              values: view('float32', 16),
              queries: view('float32x2', 16),
              neighborCount: 16,
              degree: 1,
              kernel,
              output: {values: view('float32x4', 16), status: view('uint32', 16)}
            })
          );
        }
      }) satisfies KernelAuditCase
  ),
  {
    exportName: 'GPUNaturalNeighbor',
    name: 'maximum-samples-and-vertices-with-weights',
    build({graph, view}) {
      graph.add(
        new GPUNaturalNeighbor({
          positions: view('float32x2', 32),
          values: view('float32', 32),
          queries: view('float32x2', 16),
          maximumVertexCount: 64,
          output: {
            values: view('float32x4', 16),
            status: view('uint32', 16),
            weights: view('float32', 512)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUConditionalGaussianSimulation',
    name: 'small-conditioning-and-joint-query-sites',
    build({graph, view}) {
      graph.add(
        new GPUConditionalGaussianSimulation({
          positions: view('float32x2', 16),
          observations: view('float32x4', 16),
          queries: view('float32x2', 16),
          covariance: COVARIANCE,
          seed: 1,
          replicateCount: 128,
          output: {
            values: view('float32', 2048),
            moments: view('float32x2', 16),
            status: view('uint32', 128)
          }
        })
      );
    }
  },
  ...(['none', 'pairs', 'cressie'] as const).map(
    weighting =>
      ({
        exportName: 'GPUVariogramModelFit',
        name: `maximum-range-search-${weighting}`,
        build({graph, view}) {
          graph.add(
            new GPUVariogramModelFit({
              distances: view('float32', 256),
              semivariances: view('float32', 256),
              pairCounts: view('uint32', 256),
              rangeSteps: 1024,
              maximumIterations: 64,
              weighting,
              output: {
                parameters: view('float32x4', 3),
                diagnostics: view('float32x4', 3),
                status: view('uint32', 3)
              }
            })
          );
        }
      }) satisfies KernelAuditCase
  )
];

COVERAGE_INTERPOLATION_KERNEL_CASES.push(
  {
    exportName: 'GPULocalKriging',
    name: 'weighted64-support32-neighbor8-column-design',
    build({graph, view}) {
      graph.add(
        new GPULocalKriging({
          positions: view('float32x2', 32),
          observations: view('float32x4', 32),
          supports: view('float32x4', 64),
          supportCount: 64,
          neighborCount: 32,
          mode: 'external-drift',
          driftCount: 8,
          sampleDrift: view('float32', 256),
          queryDrift: view('float32', 8),
          covariance: COVARIANCE,
          output: {
            values: view('float32x4', 1),
            status: view('uint32', 1),
            weights: view('float32', 32)
          }
        })
      );
    }
  },
  {
    exportName: 'GPULocalKriging',
    name: 'four-variable-four-structure-unknown-mean',
    build({graph, view}) {
      graph.add(
        new GPULocalKriging({
          positions: view('float32x2', 32),
          observations: view('float32x4', 32),
          supports: view('float32x4', 64),
          supportCount: 64,
          neighborCount: 32,
          mode: 'co-kriging',
          variableCount: 4,
          coKrigingMean: 'ordinary',
          targetVariable: 3,
          coregionalizationModels: Array.from({length: 4}, () => ({
            covariance: COVARIANCE,
            loadings: Array.from({length: 16}, (_, index) => (index % 5 === 0 ? 1 : 0))
          })),
          covariance: COVARIANCE,
          output: {
            values: view('float32x4', 1),
            status: view('uint32', 1),
            weights: view('float32', 32)
          }
        })
      );
    }
  },
  {
    exportName: 'GPULocalRBF',
    name: 'eight-bindings32-neighbor-quadratic-CSR-smoothed',
    build({graph, view}) {
      graph.add(
        new GPULocalRBF({
          positions: view('float32x2', 256),
          values: view('float32', 256),
          queries: view('float32x2', 32),
          kernel: 'quintic',
          degree: 2,
          neighborCount: 32,
          neighborhoods: view('uint32', 1024),
          neighborhoodOffsets: view('uint32', 33),
          sampleSmoothing: view('float32', 256),
          output: {values: view('float32x4', 32), status: view('uint32', 32)}
        })
      );
    }
  },
  {
    exportName: 'GPUNaturalNeighbor',
    name: 'indexed64-cavity64-polygon-eight-bindings',
    build({graph, view}) {
      graph.add(
        new GPUNaturalNeighbor({
          positions: view('float32x2', 65536),
          values: view('float32', 65536),
          queries: view('float32x2', 128),
          triangles: view('uint32', 131068 * 6),
          queryTriangles: view('uint32', 128),
          maximumNeighborCount: 64,
          maximumVertexCount: 64,
          output: {
            values: view('float32x4', 128),
            status: view('uint32', 128),
            weights: view('float32x2', 128 * 64)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUConditionalGaussianSimulation',
    name: 'shared-storage128-observations128-joint-sites',
    build({graph, view}) {
      graph.add(
        new GPUConditionalGaussianSimulation({
          positions: view('float32x2', 128),
          observations: view('float32x4', 128),
          queries: view('float32x2', 128),
          covariance: COVARIANCE,
          seed: 1,
          replicateCount: 128,
          output: {
            values: view('float32', 16384),
            moments: view('float32x2', 128),
            status: view('uint32', 128)
          }
        })
      );
    }
  }
);

for (const [observationCount, queryCount, replicateCount, reusePreparation] of [
  [129, 129, 17, false],
  [129, 129, 17, true],
  [0, 129, 17, false],
  [2048, 2048, 16, false]
] as const) {
  COVERAGE_INTERPOLATION_KERNEL_CASES.push({
    exportName: 'GPUConditionalGaussianSimulation',
    name: `blocked ${observationCount} observations, ${queryCount} joint sites, retained=${reusePreparation}`,
    build({graph, view}) {
      graph.add(
        new GPUConditionalGaussianSimulation({
          positions: view('float32x2', observationCount),
          observations: view('float32x4', observationCount),
          queries: view('float32x2', queryCount),
          covariance: COVARIANCE,
          seed: 1,
          replicateCount,
          reusePreparation,
          output: {
            values: view('float32', queryCount * replicateCount),
            moments: view('float32x2', queryCount),
            status: view('uint32', replicateCount)
          }
        })
      );
    }
  });
}
