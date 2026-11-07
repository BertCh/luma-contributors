// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPULocalKriging,
  GPULocalRBF,
  GPUNaturalNeighbor,
  GPUConditionalGaussianSimulation
} from '../../../src/gpu-spatial-analysis/index';
import {GPUVariogramModelFit} from '@luma-contributors/analysis/dataframe-analysis';
import type {KernelAuditCase} from './kernel-audit-context';

const COVARIANCE = {model: 'exponential' as const, range: 1, sill: 1};

/** Maximal option combinations, including all optional output bindings and bounded private arrays. */
export const COVERAGE_INTERPOLATION_KERNEL_CASES: KernelAuditCase[] = [
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
