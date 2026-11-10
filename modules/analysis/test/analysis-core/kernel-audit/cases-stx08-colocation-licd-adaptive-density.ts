// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getGPUAdaptiveBandwidthKernelDensityParameterValues,
  GPUAdaptiveBandwidthKernelDensity,
  GPUColocationQuotient,
  GPULocalCategoryDependence,
  GPU_ADAPTIVE_BANDWIDTH_KERNEL_DENSITY_PARAMETER_LENGTH,
  GPU_ADAPTIVE_BANDWIDTH_MAXIMUM_FACTOR_LIMIT,
  GPU_CATEGORY_PAIR_COUNTS_MAXIMUM_CATEGORY_COUNT,
  GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import type {KernelAuditCase} from './kernel-audit-context';

/** Cases of the second STX-08 group: colocation quotient, LICD, adaptive-bandwidth density. */
export const STX08_SECOND_GROUP_AUDIT_CASES: readonly KernelAuditCase[] = [
  {
    exportName: 'GPUColocationQuotient',
    name: 'every output at the maximum category count, 1000 rows',
    build: ({graph, view, weights}) => {
      const categoryCount = GPU_CATEGORY_PAIR_COUNTS_MAXIMUM_CATEGORY_COUNT;
      graph.add(
        new GPUColocationQuotient({
          weights: weights(1000, 8),
          categories: view('uint32', 1000),
          categoryCount,
          output: {
            globalQuotients: view('float32', categoryCount ** 2),
            localQuotients: view('float32', 1000 * categoryCount),
            categoryRowCounts: view('uint32', categoryCount)
          }
        })
      );
    }
  },
  {
    exportName: 'GPULocalCategoryDependence',
    name: 'every output at the maximum category count, 1000 rows',
    build: ({graph, view, weights}) => {
      const categoryCount = GPU_CATEGORY_PAIR_COUNTS_MAXIMUM_CATEGORY_COUNT;
      graph.add(
        new GPULocalCategoryDependence({
          weights: weights(1000, 8),
          categories: view('uint32', 1000),
          categoryCount,
          output: {
            statistics: view('float32', 1000 * GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS.length),
            categoryRowCounts: view('uint32', categoryCount)
          }
        })
      );
    }
  },
  ...(['gaussian', 'quartic'] as const).map(
    (kernel): KernelAuditCase => ({
      exportName: 'GPUAdaptiveBandwidthKernelDensity',
      name: `every output, ${kernel} kernel at the maximum factor limit`,
      build: ({graph, view}) => {
        const grid = makeGPURasterMetadata({
          width: 64,
          height: 64,
          cellSize: [1, 1],
          origin: [0, 0],
          rowDirection: 'north'
        });
        graph.add(
          new GPUAdaptiveBandwidthKernelDensity({
            positions: view('float32x2', 1000),
            weights: view('float32', 1000),
            parameters: view('float32', GPU_ADAPTIVE_BANDWIDTH_KERNEL_DENSITY_PARAMETER_LENGTH),
            grid,
            indexGridSize: [32, 32],
            indexBounds: [0, 0, 64, 64],
            kernel,
            normalization: 'probability',
            bandwidthFactorLimit: GPU_ADAPTIVE_BANDWIDTH_MAXIMUM_FACTOR_LIMIT,
            output: {
              values: view('float32', 64 * 64),
              weightSum: view('float32', 1),
              pointBandwidths: view('float32', 1000),
              pilotDensities: view('float32', 1000)
            }
          })
        );
        // The packer is part of the surface; it must accept a valid sensitivity.
        getGPUAdaptiveBandwidthKernelDensityParameterValues({bandwidth: 2, sensitivity: 0.5}, grid);
      }
    })
  )
];
