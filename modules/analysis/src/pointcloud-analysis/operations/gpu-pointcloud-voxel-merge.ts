// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createTransientView, GraphVectorView} from '@luma.gl/graph/gpu-core';
import {
  flattenPointCloudAttribute,
  readPointCloudAttribute,
  writePointCloudAttribute
} from './pointcloud-attribute';
import {
  validatePointCloudPackedViews,
  validatePointCloudScratchBindings
} from './pointcloud-kernel-nodes';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';

import {
  planGraphChunkDispatch,
  validateGraphChunkBindingRanges,
  type GraphChunkDispatchViews
} from '../../analysis-core/kernels/graph-chunk-dispatch';

import {
  getPointCloudRuntimeKernelNodes,
  validatePointCloudDispatchParameterBindings,
  validatePointCloudChunkViews
} from './pointcloud-chunk-kernel-nodes';

// Eight bindings plus one source-row word and the common three words fit this fixed slot.
const POINT_CLOUD_VOXEL_MAXIMUM_DISPATCH_WORD_COUNT = 12;

type AttributeFormat = 'float32' | 'float32x2' | 'float32x3' | 'float32x4';
type GPUPointCloudVoxelMergePackedOutput = {
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

type GPUPointCloudVoxelMergePackedProps = {
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
  /** Caller-owned result views; their contents are written by the graph. */
  output: GPUPointCloudVoxelMergePackedOutput;
};

type PointCloudVoxelChunkedProps<Value> =
  Value extends GraphDataView<infer Format>
    ? GraphDataView<Format> | GraphVectorView<Format>
    : Value extends readonly (infer Element)[]
      ? readonly PointCloudVoxelChunkedProps<Element>[]
      : Value extends object
        ? {[Key in keyof Value]: PointCloudVoxelChunkedProps<Value[Key]>}
        : Value;

/** All voxel-aligned fields accept identical ordered chunks; parts stay disjoint sufficient statistics. */
export type GPUPointCloudVoxelMergeProps =
  PointCloudVoxelChunkedProps<GPUPointCloudVoxelMergePackedProps>;

/** Caller-owned voxel statistics with identical ordered chunks for every field. */
export type GPUPointCloudVoxelMergeOutput =
  PointCloudVoxelChunkedProps<GPUPointCloudVoxelMergePackedOutput>;

/** `GPUPointCloudVoxelMerge` status bit 1: the merged count overflows uint32. */
export const GPU_POINT_CLOUD_VOXEL_MERGE_STATUS_COUNT_OVERFLOW = 1;
/** `GPUPointCloudVoxelMerge` status bit 2: a global representative row ID overflows. */
export const GPU_POINT_CLOUD_VOXEL_MERGE_STATUS_REPRESENTATIVE_OVERFLOW = 2;

/** Explicit associative sufficient-statistic merge; no concatenation or halo deduplication is inferred. */
export class GPUPointCloudVoxelMerge implements GPUCommandNodeProducer {
  readonly id: string;
  readonly estimatedCommandNodeCount: number;
  private readonly chunkContributors?: readonly GPUPointCloudVoxelMerge[];
  private readonly chunkViews?: GraphChunkDispatchViews;
  constructor(readonly props: GPUPointCloudVoxelMergeProps) {
    props = snapshotConstructionFields(props, ['id', 'partitionPolicy', 'parts', 'output']);
    this.id = props.id ?? 'gpu-pointcloud-voxel-merge';
    validateRequirement(
      this.id,
      props.output !== undefined && props.output !== null,
      'output must be given'
    );
    const output = snapshotConstructionFields(props.output, [
      'counts',
      'centroids',
      'representatives',
      'attribute',
      'status'
    ]);
    this.props = props = Object.freeze({
      ...props,
      output: Object.freeze({
        ...output,
        attribute: output.attribute
          ? snapshotConstructionFields(output.attribute, ['means', 'counts'])
          : undefined
      })
    });
    validateRequirement(
      this.id,
      props.partitionPolicy === 'disjoint-owned-rows',
      "partitionPolicy must be 'disjoint-owned-rows'"
    );
    const views = {
      counts: props.output.counts,
      centroids: props.output.centroids,
      representatives: props.output.representatives,
      status: props.output.status,
      attributeMeans: props.output.attribute?.means,
      attributeCounts: props.output.attribute?.counts,
      ...Object.fromEntries(
        props.parts.flatMap((part, index) => [
          [`part-${index}-counts`, part.counts],
          [`part-${index}-centroids`, part.centroids],
          [`part-${index}-representatives`, part.representatives],
          [`part-${index}-attributeMeans`, part.attribute?.means],
          [`part-${index}-attributeCounts`, part.attribute?.counts]
        ])
      )
    };
    if (Object.values(views).some(view => view instanceof GraphVectorView)) {
      validateCapacity(this.id, 'parts.length', props.parts.length);
      const plan = planGraphChunkDispatch(this.id, views);
      const last = props.parts.length - 1;
      const metadataNames = props.parts.flatMap((_, index) => [
        `part-${index}-counts`,
        `part-${index}-representatives`
      ]);
      validatePointCloudChunkViews(
        this.id,
        plan,
        Object.keys(views).filter(name => name.startsWith('part-')),
        ['counts', 'centroids', 'representatives', 'status', 'attributeMeans', 'attributeCounts'],
        {},
        [
          {
            inputNames: [`part-${last}-counts`, `part-${last}-representatives`],
            outputNames: ['counts', 'representatives', 'status']
          },
          {inputNames: metadataNames, outputNames: ['status']},
          {
            inputNames: [`part-${last}-counts`, `part-${last}-centroids`],
            outputNames: ['centroids']
          },
          ...(props.output.attribute
            ? [
                {
                  inputNames: [`part-${last}-attributeMeans`, `part-${last}-attributeCounts`],
                  outputNames: ['attributeMeans', 'attributeCounts']
                }
              ]
            : [])
        ]
      );
      this.chunkViews = views;
      this.chunkContributors = plan.chunks.map(
        chunk =>
          new GPUPointCloudVoxelMerge({
            id: `${this.id}-chunk-${chunk.chunkIndex}`,
            partitionPolicy: props.partitionPolicy,
            output: {
              counts: chunk.views['counts'],
              centroids: chunk.views['centroids'],
              representatives: chunk.views['representatives'],
              status: chunk.views['status'],
              ...(props.output.attribute
                ? {
                    attribute: {
                      means: chunk.views['attributeMeans'],
                      counts: chunk.views['attributeCounts']
                    }
                  }
                : {})
            },
            parts: props.parts.map((part, index) => ({
              sourceRowOffset: part.sourceRowOffset,
              counts: chunk.views[`part-${index}-counts`],
              centroids: chunk.views[`part-${index}-centroids`],
              representatives: chunk.views[`part-${index}-representatives`],
              ...(part.attribute
                ? {
                    attribute: {
                      means: chunk.views[`part-${index}-attributeMeans`],
                      counts: chunk.views[`part-${index}-attributeCounts`]
                    }
                  }
                : {})
            }))
          } as GPUPointCloudVoxelMergePackedProps)
      );
      this.estimatedCommandNodeCount = this.chunkContributors.reduce(
        (count, contributor) =>
          count +
          (contributor.props.output.counts.length ? contributor.estimatedCommandNodeCount : 0),
        0
      );
      validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
      return;
    }
    const packedProps = props as GPUPointCloudVoxelMergePackedProps;
    validateCapacity(this.id, 'parts.length', packedProps.parts.length);
    this.estimatedCommandNodeCount = packedProps.output.counts.length
      ? 2 +
        (packedProps.output.attribute ? 2 : 0) +
        packedProps.parts.length * (packedProps.output.attribute ? 6 : 4)
      : 0;
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    validateRequirement(
      this.id,
      packedProps.partitionPolicy === 'disjoint-owned-rows',
      "partitionPolicy must be 'disjoint-owned-rows'"
    );
    const outputs = [
      packedProps.output.counts,
      packedProps.output.centroids,
      packedProps.output.representatives,
      packedProps.output.status,
      ...(packedProps.output.attribute
        ? [packedProps.output.attribute.means, packedProps.output.attribute.counts]
        : [])
    ];
    const inputs = Object.fromEntries(
      packedProps.parts.flatMap((part, partIndex) => [
        [`parts[${partIndex}].counts`, part.counts],
        [`parts[${partIndex}].centroids`, part.centroids],
        [`parts[${partIndex}].representatives`, part.representatives],
        [`parts[${partIndex}].attribute.means`, part.attribute?.means],
        [`parts[${partIndex}].attribute.counts`, part.attribute?.counts]
      ])
    );
    const lastPart = packedProps.parts.at(-1)!;
    const metadataInputs = Object.fromEntries(
      packedProps.parts.flatMap((part, index) => [
        [`count ${index}`, part.counts],
        [`representative ${index}`, part.representatives]
      ])
    );
    validatePointCloudPackedViews(
      this.id,
      inputs,
      Object.fromEntries(outputs.map((view, index) => [`output ${index}`, view])),
      [
        {
          inputs: {counts: lastPart.counts, representatives: lastPart.representatives},
          outputs: {
            counts: packedProps.output.counts,
            representatives: packedProps.output.representatives,
            status: packedProps.output.status
          }
        },
        {inputs: metadataInputs, outputs: {status: packedProps.output.status}},
        {
          inputs: {counts: lastPart.counts, centroids: lastPart.centroids},
          outputs: {centroids: packedProps.output.centroids}
        },
        ...(packedProps.output.attribute
          ? [
              {
                inputs: {
                  attributeCounts: lastPart.attribute?.counts,
                  attributeMeans: lastPart.attribute?.means
                },
                outputs: {
                  attributeCounts: packedProps.output.attribute.counts,
                  attributeMeans: packedProps.output.attribute.means
                }
              }
            ]
          : [])
      ]
    );
    const length = packedProps.output.counts.length;
    validateViewLength(this.id, 'centroids', packedProps.output.centroids, length);
    validateViewLength(this.id, 'representatives', packedProps.output.representatives, length);
    validateViewLength(this.id, 'status', packedProps.output.status, length);
    for (const [partIndex, part] of packedProps.parts.entries()) {
      const name = `parts[${partIndex}]`;
      validateViewLength(this.id, `${name}.counts`, part.counts, length);
      validateViewLength(this.id, `${name}.centroids`, part.centroids, length);
      validateViewLength(this.id, `${name}.representatives`, part.representatives, length);
      validateCapacity(this.id, `${name}.sourceRowOffset`, part.sourceRowOffset, 0, 0xfffffffe);
      validateRequirement(
        this.id,
        Boolean(part.attribute) === Boolean(packedProps.output.attribute),
        `${name}.attribute must be present exactly when attribute is`
      );
      if (part.attribute && packedProps.output.attribute) {
        validateRequirement(
          this.id,
          part.attribute.means.format === packedProps.output.attribute.means.format,
          `${name}.attribute.means must have the format of attribute.means`
        );
        validateViewLength(this.id, `${name}.attribute.means`, part.attribute.means, length);
        validateViewLength(this.id, `${name}.attribute.counts`, part.attribute.counts, length);
        validateViewLength(this.id, 'attribute.means', packedProps.output.attribute.means, length);
        validateViewLength(
          this.id,
          'attribute.counts',
          packedProps.output.attribute.counts,
          length
        );
      }
    }
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    if (this.chunkContributors) {
      if (!this.estimatedCommandNodeCount) return [];
      validateGraphChunkBindingRanges(graph, this.id, this.chunkViews!);
      validatePointCloudDispatchParameterBindings(
        graph,
        this.id,
        POINT_CLOUD_VOXEL_MAXIMUM_DISPATCH_WORD_COUNT,
        1
      );
      for (const contributor of this.chunkContributors)
        validatePointCloudScratchBindings(graph, this.id, {
          'chunked voxel means': contributor.props.output.centroids.length * 12,
          'chunked voxel attribute means': contributor.props.output.attribute
            ? contributor.props.output.attribute.means.length *
              contributor.props.output.attribute.means.rowByteLength
            : 0
        });
      return this.chunkContributors.flatMap(contributor =>
        contributor.props.output.counts.length ? [...contributor.getCommandNodes(graph)] : []
      );
    }
    const {
      parts,
      output: {counts, centroids, representatives, status, attribute}
    } = this.props as GPUPointCloudVoxelMergePackedProps;
    if (!counts.length) return [];
    validateGraphChunkBindingRanges(graph, this.id, {
      counts,
      centroids,
      representatives,
      status,
      'attribute.means': attribute?.means,
      'attribute.counts': attribute?.counts,
      ...Object.fromEntries(
        parts.flatMap((part, partIndex) => [
          [`parts[${partIndex}].counts`, part.counts],
          [`parts[${partIndex}].centroids`, part.centroids],
          [`parts[${partIndex}].representatives`, part.representatives],
          [`parts[${partIndex}].attribute.means`, part.attribute?.means],
          [`parts[${partIndex}].attribute.counts`, part.attribute?.counts]
        ])
      )
    });
    validatePointCloudScratchBindings(graph, this.id, {
      'voxel counts, representatives and status': counts.length * 4,
      'voxel means': counts.length * 12,
      'voxel attribute means': attribute
        ? attribute.means.length * attribute.means.rowByteLength
        : 0
    });
    validatePointCloudDispatchParameterBindings(
      graph,
      this.id,
      POINT_CLOUD_VOXEL_MAXIMUM_DISPATCH_WORD_COUNT,
      1
    );
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
    const previousStatus = status;
    nodes.push(
      ...getPointCloudRuntimeKernelNodes(graph, {
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
        ...getPointCloudRuntimeKernelNodes(graph, {
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
      const nextStatus = status;
      nodes.push(
        ...getPointCloudRuntimeKernelNodes(graph, {
          id: `${this.id}-metadata-${partIndex}`,
          length: counts.length,
          inputs: {
            previousCounts,
            previousRepresentatives,
            partCounts: part.counts,
            partRepresentatives: part.representatives
          },
          outputs: {nextCounts, nextRepresentatives, nextStatus},
          runtimeWords: [part.sourceRowOffset],
          body: `let first=readPreviousCounts(index);let second=readPartCounts(index);let overflow=first>0xffffffffu-second;let representative=readPartRepresentatives(index);let invalid=second>0u&&representative>=0xffffffffu-readDispatchParameter(0u);writeNextCounts(index,first+second);writeNextRepresentatives(index,min(readPreviousRepresentatives(index),select(0xffffffffu,representative+readDispatchParameter(0u),second>0u&&!invalid)));writeNextStatus(index,readNextStatus(index)|select(0u,${GPU_POINT_CLOUD_VOXEL_MERGE_STATUS_COUNT_OVERFLOW}u,overflow)|select(0u,${GPU_POINT_CLOUD_VOXEL_MERGE_STATUS_REPRESENTATIVE_OVERFLOW}u,invalid));`
        })
      );
      nodes.push(
        ...getPointCloudRuntimeKernelNodes(graph, {
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
          ...getPointCloudRuntimeKernelNodes(graph, {
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
    });
    return nodes;
  }
}
