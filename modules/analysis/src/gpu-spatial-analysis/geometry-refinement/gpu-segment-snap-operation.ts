// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUSegmentSnap, type GPUSegmentSnapProps} from './gpu-segment-snap';

/** Packed graph-value form of segment snap; counts retain the contributor's uint32x4 layout. */
export type SegmentSnapOperationProps = Pick<
  GPUSegmentSnapProps,
  'id' | 'spatialContext' | 'tolerance' | 'maximumVerticesPerLine'
> & {
  positions: string;
  lineOffsets: string;
  referencePositions: string;
  rowCount: number;
  referenceCount: number;
  lineCount: number;
  output: {positions: string; sourceEdges: string; counts: string; validity?: string};
};

/** GPU segment snap operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUSegmentSnapOperation extends GPUContributorOperation {
  constructor(props: SegmentSnapOperationProps) {
    const length = props.lineCount * props.maximumVerticesPerLine;
    super({
      id: props.id ?? 'segment-snap',
      inputs: {
        positions: props.positions,
        lineOffsets: props.lineOffsets,
        referencePositions: props.referencePositions
      },
      outputs: {
        positionsOutput: {value: props.output.positions, format: 'float32x2', length},
        sourceEdges: {value: props.output.sourceEdges, format: 'uint32', length},
        counts: {value: props.output.counts, format: 'uint32x4', length: props.lineCount},
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
          maximumVerticesPerLine: props.maximumVerticesPerLine,
          positions: getView<'float32x2'>('positions'),
          lineOffsets: getView<'uint32'>('lineOffsets'),
          referencePositions: getView<'float32x2'>('referencePositions'),
          output: {
            positions: getView<'float32x2'>('positionsOutput'),
            sourceEdges: getView<'uint32'>('sourceEdges'),
            counts: getView<'uint32x4'>('counts'),
            validity:
              props.output.validity !== undefined ? getView<'uint32'>('validity') : undefined
          }
        })
    });
  }
}
