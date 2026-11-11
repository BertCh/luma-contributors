// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPULineLoopRemoval, type GPULineLoopRemovalProps} from './gpu-line-loop-removal';

/** Graph names in place of views; lengths and every property are compile-time. */
export type LineLoopRemovalOperationProps = Pick<
  GPULineLoopRemovalProps,
  'id' | 'spatialContext' | 'intersectionCapacity'
> & {
  lines: {positions: string; lineOffsets: string};
  lineCount: number;
  vertexCapacity: number;
  output: {
    /** float32x2, vertexCapacity rows. */
    positions: string;
    /** uint32, lineCount + 1 rows; whole input lines are dropped on overflow. */
    lineOffsets: string;
    /** uint32, vertexCapacity rows; first input vertex of the outgoing edge. */
    sourceEdges: string;
    /** uint32x4: [writtenVertices, requiredVertices, outputOverflow, incompleteNoding]. */
    status: string;
  };
};

/** GPU loop erasure operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPULineLoopRemovalOperation extends GPUContributorOperation {
  constructor(props: LineLoopRemovalOperationProps) {
    super({
      id: props.id ?? 'line-loop-removal',
      inputs: {inputPositions: props.lines.positions, inputOffsets: props.lines.lineOffsets},
      outputs: {
        positions: {
          value: props.output.positions,
          format: 'float32x2',
          length: props.vertexCapacity
        },
        lineOffsets: {
          value: props.output.lineOffsets,
          format: 'uint32',
          length: props.lineCount + 1
        },
        sourceEdges: {
          value: props.output.sourceEdges,
          format: 'uint32',
          length: props.vertexCapacity
        },
        status: {value: props.output.status, format: 'uint32x4', length: 1}
      },
      createContributor: getView =>
        new GPULineLoopRemoval({
          ...props,
          lines: {
            kind: 'lines',
            positions: getView<'float32x2'>('inputPositions'),
            lineOffsets: getView<'uint32'>('inputOffsets')
          },
          output: {
            positions: getView<'float32x2'>('positions'),
            lineOffsets: getView<'uint32'>('lineOffsets'),
            sourceEdges: getView<'uint32'>('sourceEdges'),
            status: getView<'uint32x4'>('status')
          }
        }),
      emptyOutputPorts: ['positions', 'sourceEdges']
    });
  }
}
