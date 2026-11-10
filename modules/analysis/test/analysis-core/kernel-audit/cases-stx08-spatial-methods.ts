// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  addContiguityOrderCorrelogramRecipe,
  getGPURedcapSingleLinkageParameterValues,
  GPUCategoryPairCounts,
  GPUGeographicalDetector,
  GPURedcapSingleLinkage,
  GPUSpatialEntropy,
  GPU_CATEGORY_PAIR_COUNTS_MAXIMUM_CATEGORY_COUNT,
  GPU_CONTIGUITY_ORDER_CORRELOGRAM_COLUMNS,
  GPU_GEOGRAPHICAL_DETECTOR_LAYOUT,
  GPU_MINIMUM_SPANNING_TREE_MAXIMUM_COLUMNS,
  GPU_REDCAP_SINGLE_LINKAGE_MAXIMUM_COMPONENT_ITERATIONS,
  GPU_REDCAP_SINGLE_LINKAGE_PARAMETER_LENGTH,
  GPU_SPATIAL_ENTROPY_LAYOUT,
  getGPURegionPartitionEvaluationMaximumRowCount,
  GPUHealpixCellAggregation,
  GPUHealpixPointToCell
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {KernelAuditCase} from './kernel-audit-context';

/** Cases of the first STX-08 group: geographical detector, REDCAP single linkage, entropy, correlogram. */
export const STX08_SPATIAL_METHODS_AUDIT_CASES: readonly KernelAuditCase[] = [
  {
    exportName: 'GPUGeographicalDetector',
    name: 'every output, 1000 rows, 128 labels',
    build: ({graph, view}) => {
      graph.add(
        new GPUGeographicalDetector({
          values: view('float32', 1000),
          labels: view('uint32', 1000),
          labelCapacity: 128,
          output: {summary: view('float32', GPU_GEOGRAPHICAL_DETECTOR_LAYOUT.length)}
        })
      );
    }
  },
  {
    exportName: 'GPUGeographicalDetector',
    name: 'rows at the evaluation maximum, default label capacity limit',
    build: ({graph, view, maximumRows}) => {
      const rows = Math.min(
        getGPURegionPartitionEvaluationMaximumRowCount(graph.device),
        maximumRows(4)
      );
      graph.add(
        new GPUGeographicalDetector({
          values: view('float32', rows),
          labels: view('uint32', rows),
          labelCapacity: 4096,
          output: {summary: view('float32', GPU_GEOGRAPHICAL_DETECTOR_LAYOUT.length)}
        })
      );
    }
  },
  ...[1, GPU_MINIMUM_SPANNING_TREE_MAXIMUM_COLUMNS].map(
    (columnCount): KernelAuditCase => ({
      exportName: 'GPURedcapSingleLinkage',
      name: `every output ${columnCount} columns at the maximum component iterations`,
      build: ({graph, view, weights}) => {
        const rows = 1000;
        graph.add(
          new GPURedcapSingleLinkage({
            weights: weights(rows, 8),
            values: view('float32', rows * columnCount),
            columnCount,
            componentIterations: GPU_REDCAP_SINGLE_LINKAGE_MAXIMUM_COMPONENT_ITERATIONS,
            parameters: view('uint32', GPU_REDCAP_SINGLE_LINKAGE_PARAMETER_LENGTH),
            output: {
              labels: view('uint32', rows),
              regionCount: view('uint32', 1),
              status: {
                converged: view('uint32', 1),
                iterationLimitReached: view('uint32', 1),
                invalidCount: view('uint32', 1)
              }
            }
          })
        );
        // The packer is part of the surface; it must accept a valid count.
        getGPURedcapSingleLinkageParameterValues({regionCount: 8});
      }
    })
  ),
  {
    exportName: 'GPURedcapSingleLinkage',
    name: 'rows and slots at the binding limit, 1 column',
    build: ({graph, view, maximumRows}) => {
      const rows = maximumRows(8);
      const slots = maximumRows(4);
      graph.add(
        new GPURedcapSingleLinkage({
          weights: {
            offsets: view('uint32', rows + 1),
            neighbors: view('uint32', slots),
            weights: view('float32', slots)
          },
          values: view('float32', rows),
          parameters: view('uint32', GPU_REDCAP_SINGLE_LINKAGE_PARAMETER_LENGTH),
          output: {labels: view('uint32', rows)}
        })
      );
    }
  },
  {
    exportName: 'GPUCategoryPairCounts',
    name: `${GPU_CATEGORY_PAIR_COUNTS_MAXIMUM_CATEGORY_COUNT} categories, 1000 rows`,
    build: ({graph, view, weights}) => {
      const categoryCount = GPU_CATEGORY_PAIR_COUNTS_MAXIMUM_CATEGORY_COUNT;
      graph.add(
        new GPUCategoryPairCounts({
          weights: weights(1000, 8),
          categories: view('uint32', 1000),
          categoryCount,
          output: {
            pairCounts: view('uint32', categoryCount ** 2),
            categoryRowCounts: view('uint32', categoryCount)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUSpatialEntropy',
    name: 'every output at the maximum category count',
    build: ({graph, view, weights}) => {
      const categoryCount = GPU_CATEGORY_PAIR_COUNTS_MAXIMUM_CATEGORY_COUNT;
      graph.add(
        new GPUSpatialEntropy({
          weights: weights(1000, 8),
          categories: view('uint32', 1000),
          categoryCount,
          output: {
            summary: view('float32', GPU_SPATIAL_ENTROPY_LAYOUT.length),
            pairCounts: view('uint32', categoryCount ** 2)
          }
        })
      );
    }
  },
  {
    exportName: 'addContiguityOrderCorrelogramRecipe',
    name: 'four orders on 1000 rows with a mask',
    build: ({graph, view, weights}) => {
      const maximumOrder = 4;
      addContiguityOrderCorrelogramRecipe(graph, {
        weights: weights(1000, 4),
        values: view('float32', 1000),
        mask: view('uint32', 1000),
        maximumOrder,
        outputs: {
          correlogram: view(
            'float32',
            maximumOrder * GPU_CONTIGUITY_ORDER_CORRELOGRAM_COLUMNS.length
          ),
          incompleteOrders: view('uint32', maximumOrder)
        }
      });
    }
  },
  {
    exportName: 'GPUHealpixPointToCell',
    name: 'every output, mask, order 29 uniq',
    build: ({graph, view}) => {
      graph.add(
        new GPUHealpixPointToCell({
          order: 29,
          scheme: 'uniq',
          positions: view('float32x2', 1000),
          mask: view('uint32', 1000),
          output: {cells: view('uint32x2', 1000), validity: view('uint32', 1000)}
        })
      );
    }
  },
  {
    exportName: 'GPUHealpixCellAggregation',
    name: 'every output at order 29',
    build: ({graph, view}) => {
      graph.add(
        new GPUHealpixCellAggregation({
          order: 29,
          scheme: 'uniq',
          cells: view('uint32x2', 1000),
          mask: view('uint32', 1000),
          values: view('float32', 1000),
          cellCapacity: 512,
          output: {
            cells: view('uint32x2', 512),
            counts: view('uint32', 512),
            sums: view('float32', 512),
            minimums: view('float32', 512),
            maximums: view('float32', 512),
            status: {
              count: view('uint32', 1),
              requiredCount: view('uint32', 1),
              overflow: view('uint32', 1),
              invalidCount: view('uint32', 1)
            }
          }
        })
      );
    }
  }
];
