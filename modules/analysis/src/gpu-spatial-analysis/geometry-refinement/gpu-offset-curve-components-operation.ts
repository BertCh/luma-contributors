// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUOffsetCurveComponents,
  type GPUOffsetCurveComponentsProps
} from './gpu-offset-curve-components';

/** Graph names and explicit output lengths; all properties are compile-time. */
export type OffsetCurveComponentsOperationProps = Pick<
  GPUOffsetCurveComponentsProps,
  'id' | 'distance' | 'tolerance' | 'rowsPerVertex' | 'intersectionCapacity' | 'segmentCapacity'
> & {
  original: {positions: string; lineOffsets: string};
  raw: {positions: string; lineOffsets: string};
  lineCount: number;
  vertexCapacity: number;
  componentCapacity: number;
  output: {positions: string; components: string; sourceEdges: string; status: string};
};

/** GPU component operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUOffsetCurveComponentsOperation extends GPUContributorOperation {
  constructor(props: OffsetCurveComponentsOperationProps) {
    super({
      id: props.id ?? 'offset-curve-components',
      inputs: {
        originalPositions: props.original.positions,
        originalOffsets: props.original.lineOffsets,
        rawPositions: props.raw.positions,
        rawOffsets: props.raw.lineOffsets
      },
      outputs: {
        positions: {
          value: props.output.positions,
          format: 'float32x2',
          length: props.vertexCapacity
        },
        components: {
          value: props.output.components,
          format: 'uint32x4',
          length: props.componentCapacity
        },
        sourceEdges: {
          value: props.output.sourceEdges,
          format: 'uint32',
          length: props.vertexCapacity
        },
        status: {value: props.output.status, format: 'uint32x4', length: 2}
      },
      createContributor: getView =>
        new GPUOffsetCurveComponents({
          ...props,
          original: {
            positions: getView<'float32x2'>('originalPositions'),
            lineOffsets: getView<'uint32'>('originalOffsets')
          },
          raw: {
            positions: getView<'float32x2'>('rawPositions'),
            lineOffsets: getView<'uint32'>('rawOffsets')
          },
          output: {
            positions: getView<'float32x2'>('positions'),
            components: getView<'uint32x4'>('components'),
            sourceEdges: getView<'uint32'>('sourceEdges'),
            status: getView<'uint32x4'>('status')
          }
        }),
      emptyOutputPorts: ['positions', 'components', 'sourceEdges']
    });
  }
}
