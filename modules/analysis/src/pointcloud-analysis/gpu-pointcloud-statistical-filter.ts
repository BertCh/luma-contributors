// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {getPointCloudKernelNodes} from './pointcloud-kernel-nodes';

export type GPUPointCloudStatisticalFilterProps = {
  id?: string;
  squaredDistances: GraphDataView<'float32'>;
  counts: GraphDataView<'uint32'>;
  capacity: number;
  standardDeviations?: number;
  /** Mean retained neighbor distance; -1 means no neighborhood. */
  meanDistances: GraphDataView<'float32'>;
  /** Original-row inlier flags. Uses population standard deviation of valid mean distances. */
  mask: GraphDataView<'uint32'>;
};

export class GPUPointCloudStatisticalFilter {
  readonly id: string;
  constructor(readonly props: GPUPointCloudStatisticalFilterProps) {
    this.id = props.id ?? 'gpu-pointcloud-statistical-filter';
    assert(Number.isInteger(props.capacity) && props.capacity > 0 && props.capacity <= 64);
    assert(props.squaredDistances.length === props.counts.length * props.capacity);
    assert(
      props.meanDistances.length === props.counts.length &&
        props.mask.length === props.counts.length
    );
    assert(
      (props.standardDeviations ?? 2) >= 0 &&
        Number.isFinite(Math.fround(props.standardDeviations ?? 2))
    );
  }
  getCommandNodes<Parameters>(graph: GPUCommandGraph<Parameters>) {
    const {squaredDistances, counts, capacity, meanDistances, mask} = this.props;
    const nodes = getPointCloudKernelNodes(graph, {
      id: `${this.id}-means`,
      length: counts.length,
      inputs: {squaredDistances, counts},
      outputs: {meanDistances},
      body: `let support = min(readCounts(index), ${capacity}u);
var distanceSum = 0.0;
for (var slot = 0u; slot < support; slot++) { distanceSum += sqrt(readSquaredDistances(index * ${capacity}u + slot)); }
writeMeanDistances(index, select(-1.0, distanceSum / f32(max(support, 1u)), support > 0u));`
    });
    const blockCount = Math.max(1, Math.ceil(counts.length / 256));
    const moments = createTransientView(graph, `${this.id}-moments`, 'float32x3', blockCount);
    const threshold = createTransientView(graph, `${this.id}-threshold`, 'float32', 1);
    nodes.push(
      ...getPointCloudKernelNodes(graph, {
        id: `${this.id}-block-statistics`,
        length: blockCount,
        inputs: {meanDistances},
        outputs: {moments},
        body: `var support = 0u; var mean = 0.0; var variance = 0.0;
for (var point = index * 256u; point < min((index + 1u) * 256u, ${counts.length}u); point++) {
  let value = readMeanDistances(point);
  if (!(value >= 0.0 && value <= 3.402823e38)) { continue; }
  support++; let delta = value - mean; mean += delta / f32(support); variance += (delta * (value - mean) - variance) / f32(support);
}
writeMoments(index, vec3f(f32(support), mean, variance));`
      }),
      ...getPointCloudKernelNodes(graph, {
        id: `${this.id}-global-statistics`,
        length: 1,
        inputs: {moments},
        outputs: {threshold},
        body: `var support = 0.0; var mean = 0.0; var variance = 0.0;
for (var batchIndex = 0u; batchIndex < ${blockCount}u; batchIndex++) {
  let moment = readMoments(batchIndex);
  if (moment.x == 0.0) { continue; }
  let total = support + moment.x; let delta = moment.y - mean;
  // Store population variance rather than its sum, which can overflow on finite distances.
  let firstWeight = support / total; let secondWeight = moment.x / total;
  variance = variance * firstWeight + moment.z * secondWeight + delta * delta * firstWeight * secondWeight;
  mean += delta * secondWeight; support = total;
}
writeThreshold(0u, mean + ${Math.fround(this.props.standardDeviations ?? 2)} * sqrt(max(variance, 0.0)));`
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
