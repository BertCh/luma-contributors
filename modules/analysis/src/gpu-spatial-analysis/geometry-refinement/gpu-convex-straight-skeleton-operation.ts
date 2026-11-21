// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotRefinementOperationProps} from './refinement-construction-snapshot';

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUConvexStraightSkeleton,
  type GPUConvexStraightSkeletonProps
} from './gpu-convex-straight-skeleton';

/** Graph-value names in place of views; all properties are compile-time. */
export type ConvexStraightSkeletonOperationProps = Pick<
  GPUConvexStraightSkeletonProps,
  | 'id'
  | 'spatialContext'
  | 'maximumVertices'
  | 'maximumIterations'
  | 'offsetDistance'
  | 'eventTolerance'
  | 'segmentCapacity'
> & {
  positions: string;
  ringOffsets: string;
  /** Number of independent rings (ringOffsets has polygonCount + 1 rows). */
  polygonCount: number;
  output: {
    /** float32x4, polygonCount * segmentCapacity rows. */
    segments: string;
    /** uint32x2, polygonCount * segmentCapacity rows. */
    sourceEdges: string;
    /** float32x2, polygonCount * maximumVertices rows. */
    offsetPositions: string;
    /** uint32x4: [writtenArcs, requiredArcs, offsetVertexCount, flags]. */
    status: string;
  };
};

/** GPU wavefront operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUConvexStraightSkeletonOperation extends GPUContributorOperation {
  constructor(props: ConvexStraightSkeletonOperationProps) {
    props = snapshotRefinementOperationProps(props, 'convex');
    const maximumVertices = props.maximumVertices ?? 32;
    const segmentCapacity = props.segmentCapacity ?? maximumVertices * 2;
    super({
      id: props.id ?? 'convex-straight-skeleton',
      inputs: {positions: props.positions, ringOffsets: props.ringOffsets},
      outputs: {
        segments: {
          value: props.output.segments,
          format: 'float32x4',
          length: props.polygonCount * segmentCapacity
        },
        sourceEdges: {
          value: props.output.sourceEdges,
          format: 'uint32x2',
          length: props.polygonCount * segmentCapacity
        },
        offsetPositions: {
          value: props.output.offsetPositions,
          format: 'float32x2',
          length: props.polygonCount * maximumVertices
        },
        status: {value: props.output.status, format: 'uint32x4', length: props.polygonCount}
      },
      createContributor: getView =>
        new GPUConvexStraightSkeleton({
          ...props,
          positions: getView<'float32x2'>('positions'),
          ringOffsets: getView<'uint32'>('ringOffsets'),
          output: {
            segments: getView<'float32x4'>('segments'),
            sourceEdges: getView<'uint32x2'>('sourceEdges'),
            offsetPositions: getView<'float32x2'>('offsetPositions'),
            status: getView<'uint32x4'>('status')
          }
        })
    });
  }
}
