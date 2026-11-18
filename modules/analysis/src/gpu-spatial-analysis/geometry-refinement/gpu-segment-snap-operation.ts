// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPUSegmentSnap,
  validateSegmentSnapCompleteness,
  type GPUSegmentSnapProps
} from './gpu-segment-snap';
import {snapshotSegmentSnapConstruction} from './segment-snap-construction';

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
    issueCode?: string;
    status?: {candidateOverflow?: string};
    /** @deprecated Use status.candidateOverflow. */
    candidateOverflow?: string;
  };
};

/** Capture the same operation policy and nested ports for both execution forms. @internal */
export function snapshotSegmentSnapOperationProps(
  props: SegmentSnapOperationProps
): SegmentSnapOperationProps {
  return snapshotSegmentSnapConstruction(props, [
    'id',
    'spatialContext',
    'positions',
    'lineOffsets',
    'referencePositions',
    'rowCount',
    'referenceCount',
    'lineCount',
    'tolerance',
    'maximumVerticesPerLine',
    'searchStructure',
    'candidateCapacity',
    'candidateOverflowPolicy',
    'output'
  ]);
}

/** GPU segment snap operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUSegmentSnapOperation extends GPUContributorOperation {
  constructor(props: SegmentSnapOperationProps) {
    props = snapshotSegmentSnapOperationProps(props);
    const id = props.id ?? 'segment-snap';
    const candidateOverflow =
      props.output.status?.candidateOverflow ?? props.output.candidateOverflow;
    validateCapacity(id, 'rowCount', props.rowCount, 0);
    validateCapacity(id, 'referenceCount', props.referenceCount, 0);
    validateCapacity(id, 'lineCount', props.lineCount);
    validateSegmentSnapCompleteness(
      id,
      props.rowCount,
      props.referenceCount,
      props.lineCount,
      props,
      Boolean(props.output.validity || props.output.issueCode || candidateOverflow)
    );
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
          : {}),
        ...(props.output.issueCode !== undefined
          ? {
              issueCode: {
                value: props.output.issueCode,
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
              props.output.validity !== undefined ? getView<'uint32'>('validity') : undefined,
            issueCode:
              props.output.issueCode !== undefined ? getView<'uint32'>('issueCode') : undefined
          }
        })
    });
  }
}
