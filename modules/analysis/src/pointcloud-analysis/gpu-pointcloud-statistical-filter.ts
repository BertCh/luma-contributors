// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createTransientView, GPUReduction} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement,
  validateViewLength
} from '../analysis-core/kernels/gpu-contributor-utils';
import {GPU_POINT_CLOUD_MAXIMUM_NEIGHBOR_CAPACITY} from './pointcloud-limits';
import {getPointCloudKernelNodes} from './pointcloud-kernel-nodes';

export type GPUPointCloudStatisticalFilterProps = {
  id?: string;
  squaredDistances: GraphDataView<'float32'>;
  counts: GraphDataView<'uint32'>;
  capacity: number;
  standardDeviations?: number;
  /** Mean retained neighbor distance; -1 means no neighborhood. */
  meanDistances: GraphDataView<'float32'>;
  /**
   * Original-row inlier flags: one when the row has at least one neighbor and its mean distance is
   * at most the global threshold, zero otherwise. Uses population standard deviation of valid mean
   * distances. There is no per-row status word: a row without neighbors (`meanDistances` is -1) is
   * an outlier by definition.
   */
  validity: GraphDataView<'uint32'>;
};

/**
 * Global mean-neighbor-distance outlier filter (PDAL `filters.outlier`, statistical).
 *
 * The global mean and population variance of the valid mean distances come from three published
 * `GPUReduction` sums, so no kernel loops over rows. Each term is divided by the valid count
 * before it is summed, so a finite distance cannot overflow the accumulated moment.
 */
export class GPUPointCloudStatisticalFilter implements GPUCommandNodeProducer {
  readonly id: string;
  /** The inlier flags, `props.validity`. */
  readonly validity: GraphDataView<'uint32'>;
  constructor(readonly props: GPUPointCloudStatisticalFilterProps) {
    this.id = props.id ?? 'gpu-pointcloud-statistical-filter';
    const {validity} = props;
    validateRequirement(this.id, validity !== undefined, 'validity must be given');
    this.validity = validity;
    validateCapacity(
      this.id,
      'capacity',
      props.capacity,
      1,
      GPU_POINT_CLOUD_MAXIMUM_NEIGHBOR_CAPACITY
    );
    validateViewLength(
      this.id,
      'squaredDistances',
      props.squaredDistances,
      props.counts.length * props.capacity
    );
    validateViewLength(this.id, 'meanDistances', props.meanDistances, props.counts.length);
    validateViewLength(this.id, 'validity', this.validity, props.counts.length);
    validateFiniteNumber(this.id, 'standardDeviations', props.standardDeviations ?? 2, 0);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {squaredDistances, counts, capacity, meanDistances} = this.props;
    const mask = this.validity;
    const nodes: GPUCommandNode<Parameters>[] = getPointCloudKernelNodes(graph, {
      id: `${this.id}-means`,
      length: counts.length,
      inputs: {squaredDistances, counts},
      outputs: {meanDistances},
      body: `let support = min(readCounts(index), ${capacity}u);
var distanceSum = 0.0;
for (var slot = 0u; slot < support; slot++) { distanceSum += sqrt(readSquaredDistances(index * ${capacity}u + slot)); }
writeMeanDistances(index, select(-1.0, distanceSum / f32(max(support, 1u)), support > 0u));`
    });
    const validity = createTransientView(
      graph,
      `${this.id}-validity-flags`,
      'uint32',
      counts.length
    );
    const validCount = createTransientView(graph, `${this.id}-valid-count-total`, 'uint32', 1);
    const scaledValues = createTransientView(
      graph,
      `${this.id}-scaled-values`,
      'float32',
      counts.length
    );
    const mean = createTransientView(graph, `${this.id}-mean-value`, 'float32', 1);
    const varianceTerms = createTransientView(
      graph,
      `${this.id}-variance-terms`,
      'float32',
      counts.length
    );
    const variance = createTransientView(graph, `${this.id}-variance-value`, 'float32', 1);
    const threshold = createTransientView(graph, `${this.id}-threshold-value`, 'float32', 1);
    const sum = (name: string, input: GraphDataView<'float32'>, output: GraphDataView<'float32'>) =>
      new GPUReduction({id: `${this.id}-${name}`, input, output, operation: 'sum'}).getCommandNodes(
        graph
      );
    nodes.push(
      ...getPointCloudKernelNodes(graph, {
        id: `${this.id}-validity`,
        length: counts.length,
        inputs: {meanDistances},
        outputs: {validity},
        body: 'let value = readMeanDistances(index); writeValidity(index, u32(value >= 0.0 && value <= 3.402823e38));'
      }),
      ...new GPUReduction({
        id: `${this.id}-valid-count`,
        input: validity,
        output: validCount,
        operation: 'sum'
      }).getCommandNodes(graph),
      ...getPointCloudKernelNodes(graph, {
        id: `${this.id}-scaled`,
        length: counts.length,
        inputs: {meanDistances, validity, validCount},
        outputs: {scaledValues},
        body: 'writeScaledValues(index, select(0.0, readMeanDistances(index) / f32(max(readValidCount(0u), 1u)), readValidity(index) != 0u));'
      }),
      ...sum('mean', scaledValues, mean),
      ...getPointCloudKernelNodes(graph, {
        id: `${this.id}-terms`,
        length: counts.length,
        inputs: {meanDistances, validity, validCount, mean},
        outputs: {varianceTerms},
        // Scaling the difference by 1 / sqrt(count) before squaring keeps each term finite.
        body: `let scaledDifference = (readMeanDistances(index) - readMean(0u)) / sqrt(f32(max(readValidCount(0u), 1u)));
writeVarianceTerms(index, select(0.0, scaledDifference * scaledDifference, readValidity(index) != 0u));`
      }),
      ...sum('variance', varianceTerms, variance),
      ...getPointCloudKernelNodes(graph, {
        id: `${this.id}-threshold`,
        length: 1,
        inputs: {mean, variance},
        outputs: {threshold},
        body: `writeThreshold(0u, readMean(0u) + ${Math.fround(this.props.standardDeviations ?? 2)} * sqrt(max(readVariance(0u), 0.0)));`
      }),
      ...getPointCloudKernelNodes(graph, {
        id: `${this.id}-classify`,
        length: counts.length,
        inputs: {meanDistances, threshold},
        outputs: {mask},
        body: `let value = readMeanDistances(index);
writeMask(index, u32(value >= 0.0 && value <= readThreshold(0u)));`
      })
    );
    return nodes;
  }
}
