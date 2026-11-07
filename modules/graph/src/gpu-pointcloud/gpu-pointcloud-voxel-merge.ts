// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '../gpu-core/gpu-command-graph';
import {createTransientView, doGraphDataViewsOverlap} from '../gpu-core/graph-data-view-utils';
import type {GPUCommandNode} from '../gpu-core/gpu-command-node';
import {
  flattenPointCloudAttribute,
  readPointCloudAttribute,
  writePointCloudAttribute
} from './pointcloud-attribute';
import {getSpatialOperationNodes} from '../utils/gpu-spatial-operation';
import {validateGraphOutputsDisjointFromInputs} from '../utils/gpu-contributor-utils';

type AttributeFormat = 'float32' | 'float32x2' | 'float32x3' | 'float32x4';
export type GPUPointCloudVoxelMergeProps = {
  id?: string;
  /** Caller certifies every point contributes to exactly one part; halo replicas must be excluded before reduction. All parts use the identical grid. */
  partitionPolicy: 'disjoint-owned-rows';
  parts: readonly {
    counts: GraphDataView<'uint32'>;
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
  /** Bit1 count overflow, bit2 global representative ID overflow; affected results must be rejected. */
  status: GraphDataView<'uint32'>;
};

/** Explicit associative sufficient-statistic merge; no concatenation or halo deduplication is inferred. */
export class GPUPointCloudVoxelMerge {
  readonly id: string;
  constructor(readonly props: GPUPointCloudVoxelMergeProps) {
    this.id = props.id ?? 'gpu-pointcloud-voxel-merge';
    assert(props.parts.length > 0 && props.partitionPolicy === 'disjoint-owned-rows');
    const outputs = [
      props.counts,
      props.centroids,
      props.representatives,
      props.status,
      ...(props.attribute ? [props.attribute.means, props.attribute.counts] : [])
    ];
    // Distinct passes publish metadata, centroids and attributes; their final records must not alias.
    for (const [index, output] of outputs.entries()) {
      assert(!outputs.slice(0, index).some(previous => doGraphDataViewsOverlap(previous, output)));
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
    assert(
      [props.centroids, props.representatives, props.status].every(view => view.length === length)
    );
    for (const part of props.parts) {
      assert(
        [part.counts, part.centroids, part.representatives].every(view => view.length === length)
      );
      assert(
        Number.isSafeInteger(part.sourceRowOffset) &&
          part.sourceRowOffset >= 0 &&
          part.sourceRowOffset < 0xffffffff
      );
      assert(Boolean(part.attribute) === Boolean(props.attribute));
      if (part.attribute && props.attribute) {
        assert(part.attribute.means.format === props.attribute.means.format);
        assert(
          [
            part.attribute.means,
            part.attribute.counts,
            props.attribute.means,
            props.attribute.counts
          ].every(view => view.length === length)
        );
      }
    }
  }
  getCommandNodes<Parameters>(graph: GPUCommandGraph<Parameters>) {
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
      ...getSpatialOperationNodes(graph, {
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
        ...getSpatialOperationNodes(graph, {
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
        ...getSpatialOperationNodes(graph, {
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
          body: `let first=readPreviousCounts(index);let second=readPartCounts(index);let overflow=first>0xffffffffu-second;let representative=readPartRepresentatives(index);let invalid=second>0u&&representative>=0xffffffffu-${part.sourceRowOffset}u;writeNextCounts(index,first+second);writeNextRepresentatives(index,min(readPreviousRepresentatives(index),select(0xffffffffu,representative+${part.sourceRowOffset}u,second>0u&&!invalid)));writeNextStatus(index,readPreviousStatus(index)|u32(overflow)|select(0u,2u,invalid));`
        })
      );
      nodes.push(
        ...getSpatialOperationNodes(graph, {
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
          ? flattenPointCloudAttribute(graph, attribute.means)
          : createTransientView(
              graph,
              `${this.id}-attribute-mean-${partIndex}`,
              'float32',
              counts.length * components
            );
        nodes.push(
          ...getSpatialOperationNodes(graph, {
            id: `${this.id}-attribute-${partIndex}`,
            length: counts.length,
            inputs: {
              firstCounts: previousAttributeCount,
              firstMeans: previousAttributeMean,
              secondCounts: part.attribute.counts,
              secondMeans: flattenPointCloudAttribute(graph, part.attribute.means)
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
