// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotRefinementOperationProps} from './refinement-construction-snapshot';

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUVectorStraightSkeleton,
  type GPUVectorStraightSkeletonProps
} from './gpu-vector-straight-skeleton';

/** Graph-value names in place of packed views. Every property is compile-time. */
export type VectorStraightSkeletonOperationProps = Pick<
  GPUVectorStraightSkeletonProps,
  | 'id'
  | 'maximumVertices'
  | 'maximumEvents'
  | 'offsetDistance'
  | 'eventTolerance'
  | 'segmentCapacity'
> & {
  positions: string;
  ringOffsets: string;
  polygonOffsets: string;
  /** Declared input, although the contributor does not use feature grouping. */
  featureOffsets: string;
  /** polygonOffsets has polygonCount + 1 rows. */
  polygonCount: number;
  output: {
    /** float32x4, polygonCount * segmentCapacity rows. */
    segments: string;
    /** uint32x2, polygonCount * segmentCapacity rows; original position-row IDs. */
    sourceEdges: string;
    /** float32x2, polygonCount * (maximumVertices + 2 * maximumEvents) rows. */
    offsetPositions: string;
    /** uint32x2, same blocks as offsetPositions; missing links are 0xffffffff. */
    offsetLinks: string;
    /** uint32x4: [writtenArcs, requiredArcs, activeOffsetVertices, flags]. */
    status: string;
  };
};

/** GPU linked wavefront; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUVectorStraightSkeletonOperation extends GPUContributorOperation {
  constructor(props: VectorStraightSkeletonOperationProps) {
    props = snapshotRefinementOperationProps(props, 'vector');
    const slots = (props.maximumVertices ?? 128) + 2 * (props.maximumEvents ?? 128);
    super({
      id: props.id ?? 'vector-straight-skeleton',
      inputs: {
        positions: props.positions,
        ringOffsets: props.ringOffsets,
        polygonOffsets: props.polygonOffsets,
        featureOffsets: props.featureOffsets
      },
      outputs: {
        segments: {
          value: props.output.segments,
          format: 'float32x4',
          length: props.polygonCount * props.segmentCapacity
        },
        sourceEdges: {
          value: props.output.sourceEdges,
          format: 'uint32x2',
          length: props.polygonCount * props.segmentCapacity
        },
        offsetPositions: {
          value: props.output.offsetPositions,
          format: 'float32x2',
          length: props.polygonCount * slots
        },
        offsetLinks: {
          value: props.output.offsetLinks,
          format: 'uint32x2',
          length: props.polygonCount * slots
        },
        status: {value: props.output.status, format: 'uint32x4', length: props.polygonCount}
      },
      createContributor: getView =>
        new GPUVectorStraightSkeleton({
          ...props,
          positions: getView<'float32x2'>('positions'),
          ringOffsets: getView<'uint32'>('ringOffsets'),
          polygonOffsets: getView<'uint32'>('polygonOffsets'),
          featureOffsets: getView<'uint32'>('featureOffsets'),
          output: {
            segments: getView<'float32x4'>('segments'),
            sourceEdges: getView<'uint32x2'>('sourceEdges'),
            offsetPositions: getView<'float32x2'>('offsetPositions'),
            offsetLinks: getView<'uint32x2'>('offsetLinks'),
            status: getView<'uint32x4'>('status')
          }
        }),
      emptyOutputPorts:
        props.polygonCount === 0
          ? ['segments', 'sourceEdges', 'offsetPositions', 'offsetLinks', 'status']
          : []
    });
  }
}
