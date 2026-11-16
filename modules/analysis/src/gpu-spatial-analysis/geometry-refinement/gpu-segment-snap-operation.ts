// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPUSegmentSnap, type GPUSegmentSnapProps} from './gpu-segment-snap';

/** Packed graph-value form of segment snap; counts retain the contributor's uint32x4 layout. */
export type SegmentSnapOperationProps = Pick<
  GPUSegmentSnapProps,
  | 'id'
  | 'spatialContext'
  | 'tolerance'
  | 'maximumVerticesPerLine'
  | 'searchStructure'
  | 'candidateCapacity'
  | 'candidateOverflowPolicy'
> & {
  positions: string;
  lineOffsets: string;
  referencePositions: string;
  rowCount: number;
  referenceCount: number;
  lineCount: number;
  output: {
    positions: string;
    sourceEdges: string;
    counts: string;
    validity?: string;
    status?: {candidateOverflow?: string};
    /** @deprecated Use status.candidateOverflow. */
    candidateOverflow?: string;
  };
};

/** GPU segment snap operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUSegmentSnapOperation extends GPUContributorOperation {
  constructor(props: SegmentSnapOperationProps) {
    const id = props.id ?? 'segment-snap';
    const candidateOverflow =
      props.output.status?.candidateOverflow ?? props.output.candidateOverflow;
    validateRequirement(
      id,
      props.output.status?.candidateOverflow === undefined ||
        props.output.candidateOverflow === undefined ||
        props.output.status.candidateOverflow === props.output.candidateOverflow,
      'output.status.candidateOverflow and its deprecated alias must name the same output'
    );
    const length = props.lineCount * props.maximumVerticesPerLine;
    super({
      id,
      inputs: {
        positions: props.positions,
        lineOffsets: props.lineOffsets,
        referencePositions: props.referencePositions
      },
      outputs: {
        positionsOutput: {value: props.output.positions, format: 'float32x2', length},
        sourceEdges: {value: props.output.sourceEdges, format: 'uint32', length},
        counts: {value: props.output.counts, format: 'uint32x4', length: props.lineCount},
        ...(candidateOverflow !== undefined
          ? {
              candidateOverflow: {
                value: candidateOverflow,
                format: 'uint32' as const,
                length: props.lineCount
              }
            }
          : {}),
        ...(props.output.validity !== undefined
          ? {
              validity: {
                value: props.output.validity,
                format: 'uint32' as const,
                length: props.lineCount
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPUSegmentSnap({
          id: props.id,
          spatialContext: props.spatialContext,
          tolerance: props.tolerance,
          searchStructure: props.searchStructure,
          candidateCapacity: props.candidateCapacity,
          candidateOverflowPolicy: props.candidateOverflowPolicy,
          maximumVerticesPerLine: props.maximumVerticesPerLine,
          positions: getView<'float32x2'>('positions'),
          lineOffsets: getView<'uint32'>('lineOffsets'),
          referencePositions: getView<'float32x2'>('referencePositions'),
          output: {
            positions: getView<'float32x2'>('positionsOutput'),
            sourceEdges: getView<'uint32'>('sourceEdges'),
            counts: getView<'uint32x4'>('counts'),
            status:
              candidateOverflow !== undefined
                ? {candidateOverflow: getView<'uint32'>('candidateOverflow')}
                : undefined,
            validity:
              props.output.validity !== undefined ? getView<'uint32'>('validity') : undefined
          }
        })
    });
  }
}
