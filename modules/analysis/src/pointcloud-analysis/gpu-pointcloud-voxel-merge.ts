// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createTransientView, doGraphDataViewsOverlap} from '@luma.gl/graph/gpu-core';
import {
  flattenPointCloudAttribute,
  readPointCloudAttribute,
  writePointCloudAttribute
} from './pointcloud-attribute';
import {getPointCloudKernelNodes} from './pointcloud-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateRequirement,
  validateViewLength
} from '../analysis-core/kernels/gpu-contributor-utils';

type AttributeFormat = 'float32' | 'float32x2' | 'float32x3' | 'float32x4';
export type GPUPointCloudVoxelMergeProps = {
  id?: string;
  /** Caller certifies every point contributes to exactly one part; halo replicas must be excluded before reduction. All parts use the identical grid. */
  partitionPolicy: 'disjoint-owned-rows';
  parts: readonly {
    counts: GraphDataView<'uint32'>;
    /**
     * Arithmetic mean of the member positions per voxel. The published
     * `GPUPointCloudVoxelDownsample` (10.0.0-alpha.10) does not produce it: its `output` is the
     * representative point, and merging those gives a count-weighted mean of representatives, not
     * a centroid. The caller computes the per-voxel means (see `docs/api/pointcloud-analysis.md`).
     */
    centroids: GraphDataView<'float32x3'>;
    representatives: GraphDataView<'uint32'>;
    /** Adds a stable original-row offset to each nonempty part representative. */
    sourceRowOffset: number;
    attribute?: {means: GraphDataView<AttributeFormat>; counts: GraphDataView<'uint32'>};
  }[];
  counts: GraphDataView<'uint32'>;
  centroids: GraphDataView<'float32x3'>;
  representatives: GraphDataView<'uint32'>;
  attribute?: {means: GraphDataView<AttributeFormat>; counts: GraphDataView<'uint32'>};
  /**
   * Per-voxel bit set, zero when clean. Bit 1 (`GPU_POINT_CLOUD_VOXEL_MERGE_STATUS_COUNT_OVERFLOW`):
   * the merged point count exceeds uint32. Bit 2 (`..._REPRESENTATIVE_OVERFLOW`): a representative
   * plus its part's `sourceRowOffset` exceeds the largest row ID. A voxel with either bit set must
   * be rejected by the caller. Bits above 2 are zero.
   */
  status: GraphDataView<'uint32'>;
};

/** `GPUPointCloudVoxelMerge` status bit 1: the merged count overflows uint32. */
export const GPU_POINT_CLOUD_VOXEL_MERGE_STATUS_COUNT_OVERFLOW = 1;
/** `GPUPointCloudVoxelMerge` status bit 2: a global representative row ID overflows. */
export const GPU_POINT_CLOUD_VOXEL_MERGE_STATUS_REPRESENTATIVE_OVERFLOW = 2;

/** Explicit associative sufficient-statistic merge; no concatenation or halo deduplication is inferred. */
export class GPUPointCloudVoxelMerge implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUPointCloudVoxelMergeProps) {
    this.id = props.id ?? 'gpu-pointcloud-voxel-merge';
    validateCapacity(this.id, 'parts.length', props.parts.length);
    validateRequirement(
      this.id,
      props.partitionPolicy === 'disjoint-owned-rows',
      "partitionPolicy must be 'disjoint-owned-rows'"
    );
    const outputs = [
      props.counts,
      props.centroids,
      props.representatives,
      props.status,
      ...(props.attribute ? [props.attribute.means, props.attribute.counts] : [])
    ];
    // Distinct passes publish metadata, centroids and attributes; their final records must not alias.
    for (const [index, output] of outputs.entries()) {
      validateRequirement(
        this.id,
        !outputs.slice(0, index).some(previous => doGraphDataViewsOverlap(previous, output)),
        'output views must not overlap each other'
      );
    }
    validateGraphOutputsDisjointFromInputs(
      this.id,
      outputs,
      props.parts.flatMap(part => [
        part.counts,
        part.centroids,
        part.representatives,
        part.attribute?.means,
        part.attribute?.counts
      ])
    );
    const length = props.counts.length;
    validateViewLength(this.id, 'centroids', props.centroids, length);
    validateViewLength(this.id, 'representatives', props.representatives, length);
    validateViewLength(this.id, 'status', props.status, length);
    for (const [partIndex, part] of props.parts.entries()) {
      const name = `parts[${partIndex}]`;
      validateViewLength(this.id, `${name}.counts`, part.counts, length);
      validateViewLength(this.id, `${name}.centroids`, part.centroids, length);
      validateViewLength(this.id, `${name}.representatives`, part.representatives, length);
      validateCapacity(this.id, `${name}.sourceRowOffset`, part.sourceRowOffset, 0, 0xfffffffe);
      validateRequirement(
        this.id,
        Boolean(part.attribute) === Boolean(props.attribute),
        `${name}.attribute must be present exactly when attribute is`
      );
      if (part.attribute && props.attribute) {
        validateRequirement(
          this.id,
          part.attribute.means.format === props.attribute.means.format,
          `${name}.attribute.means must have the format of attribute.means`
        );
        validateViewLength(this.id, `${name}.attribute.means`, part.attribute.means, length);
        validateViewLength(this.id, `${name}.attribute.counts`, part.attribute.counts, length);
        validateViewLength(this.id, 'attribute.means', props.attribute.means, length);
        validateViewLength(this.id, 'attribute.counts', props.attribute.counts, length);
      }
    }
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {parts, counts, centroids, representatives, status, attribute} = this.props;
    const nodes: GPUCommandNode<Parameters>[] = [];
    let previousCounts = createTransientView(
      graph,
      `${this.id}-count-initial`,
      'uint32',
      counts.length
    );
    let previousMeans = createTransientView(
      graph,
      `${this.id}-mean-initial`,
      'float32x3',
      counts.length
    );
    let previousRepresentatives = createTransientView(
      graph,
      `${this.id}-representative-initial`,
      'uint32',
      counts.length
    );
    let previousStatus = createTransientView(
      graph,
      `${this.id}-status-initial`,
      'uint32',
      counts.length
    );
    nodes.push(
      ...getPointCloudKernelNodes(graph, {
        id: `${this.id}-clear`,
        length: counts.length,
        inputs: {},
        outputs: {previousCounts, previousMeans, previousRepresentatives, previousStatus},
        body: 'writePreviousCounts(index,0u);writePreviousMeans(index,vec3f(0.0));writePreviousRepresentatives(index,0xffffffffu);writePreviousStatus(index,0u);'
      })
    );
    const components = Number(attribute?.means.format.match(/x([234])$/)?.[1] ?? 1);
    let previousAttributeCount = attribute
      ? createTransientView(graph, `${this.id}-attribute-count-initial`, 'uint32', counts.length)
      : undefined;
    let previousAttributeMean = attribute
      ? createTransientView(
          graph,
          `${this.id}-attribute-mean-initial`,
          'float32',
          counts.length * components
        )
      : undefined;

    if (previousAttributeCount && previousAttributeMean)
      nodes.push(
        ...getPointCloudKernelNodes(graph, {
          id: `${this.id}-attribute-clear`,
          length: counts.length,
          inputs: {},
          outputs: {count: previousAttributeCount, mean: previousAttributeMean},
          body: `writeCount(index,0u);${Array.from({length: components}, (_, component) => `writeMean(index * ${components}u + ${component}u, 0.0);`).join('')}`
        })
      );
    parts.forEach((part, partIndex) => {
      const final = partIndex === parts.length - 1;
      const nextCounts = final
        ? counts
        : createTransientView(graph, `${this.id}-count-${partIndex}`, 'uint32', counts.length);
      const nextMeans = final
        ? centroids
        : createTransientView(graph, `${this.id}-mean-${partIndex}`, 'float32x3', counts.length);
      const nextRepresentatives = final
        ? representatives
        : createTransientView(
            graph,
            `${this.id}-representative-${partIndex}`,
            'uint32',
            counts.length
          );
      const nextStatus = final
        ? status
        : createTransientView(graph, `${this.id}-status-${partIndex}`, 'uint32', counts.length);
      nodes.push(
        ...getPointCloudKernelNodes(graph, {
          id: `${this.id}-metadata-${partIndex}`,
          length: counts.length,
          inputs: {
            previousCounts,
            previousRepresentatives,
            previousStatus,
            partCounts: part.counts,
            partRepresentatives: part.representatives
          },
          outputs: {nextCounts, nextRepresentatives, nextStatus},
          body: `let first=readPreviousCounts(index);let second=readPartCounts(index);let overflow=first>0xffffffffu-second;let representative=readPartRepresentatives(index);let invalid=second>0u&&representative>=0xffffffffu-${part.sourceRowOffset}u;writeNextCounts(index,first+second);writeNextRepresentatives(index,min(readPreviousRepresentatives(index),select(0xffffffffu,representative+${part.sourceRowOffset}u,second>0u&&!invalid)));writeNextStatus(index,readPreviousStatus(index)|select(0u,${GPU_POINT_CLOUD_VOXEL_MERGE_STATUS_COUNT_OVERFLOW}u,overflow)|select(0u,${GPU_POINT_CLOUD_VOXEL_MERGE_STATUS_REPRESENTATIVE_OVERFLOW}u,invalid));`
        })
      );
      nodes.push(
        ...getPointCloudKernelNodes(graph, {
          id: `${this.id}-centroid-${partIndex}`,
          length: counts.length,
          inputs: {
            firstCounts: previousCounts,
            firstMeans: previousMeans,
            secondCounts: part.counts,
            secondMeans: part.centroids
          },
          outputs: {nextMeans},
          body: 'let first=f32(readFirstCounts(index));let second=f32(readSecondCounts(index));var mean=readFirstMeans(index);if(second>0.0){mean=mean*(first/(first+second))+readSecondMeans(index)*(second/(first+second));}writeNextMeans(index,mean);'
        })
      );
      if (attribute && part.attribute && previousAttributeCount && previousAttributeMean) {
        const nextAttributeCount = final
          ? attribute.counts
          : createTransientView(
              graph,
              `${this.id}-attribute-count-${partIndex}`,
              'uint32',
              counts.length
            );
        const nextAttributeMean = final
          ? flattenPointCloudAttribute(graph, attribute.means, this.id)
          : createTransientView(
              graph,
              `${this.id}-attribute-mean-${partIndex}`,
              'float32',
              counts.length * components
            );
        nodes.push(
          ...getPointCloudKernelNodes(graph, {
            id: `${this.id}-attribute-${partIndex}`,
            length: counts.length,
            inputs: {
              firstCounts: previousAttributeCount,
              firstMeans: previousAttributeMean,
              secondCounts: part.attribute.counts,
              secondMeans: flattenPointCloudAttribute(graph, part.attribute.means, this.id)
            },
            outputs: {nextAttributeCount, nextAttributeMean},
            body: `let first=readFirstCounts(index);let second=readSecondCounts(index);var mean=${readPointCloudAttribute('FirstMeans', 'index', components)};if(second>0u){mean=mean*(f32(first)/(f32(first)+f32(second)))+${readPointCloudAttribute('SecondMeans', 'index', components)}*(f32(second)/(f32(first)+f32(second)));}writeNextAttributeCount(index,first+second);${writePointCloudAttribute('NextAttributeMean', 'index', 'mean', components)}`
          })
        );
        previousAttributeCount = nextAttributeCount;
        previousAttributeMean = nextAttributeMean;
      }
      previousCounts = nextCounts;
      previousMeans = nextMeans;
      previousRepresentatives = nextRepresentatives;
      previousStatus = nextStatus;
    });
    return nodes;
  }
}
