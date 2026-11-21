// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPULineLoopRemoval, type GPULineLoopRemovalProps} from './gpu-line-loop-removal';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Graph names in place of views; lengths and every property are compile-time. */
export type LineLoopRemovalOperationProps = Pick<
  GPULineLoopRemovalProps,
  'id' | 'spatialContext' | 'intersectionCapacity'
> & {
  lines: {positions: string; lineOffsets: string; featureOffsets?: string};
  /** Logical feature count, required with featureOffsets input; otherwise inferred from lineCount. */
  featureCount?: number;
  lineCount: number;
  vertexCapacity: number;
  output: {
    /** float32x2, vertexCapacity rows. */
    positions: string;
    /** uint32, lineCount + 1 rows; whole input lines are dropped on overflow. */
    lineOffsets: string;
    featureOffsets?: string;
    /** uint32, vertexCapacity rows; first input vertex of the outgoing edge. */
    sourceEdges: string;
    /** uint32x4: [writtenVertices, requiredVertices, outputOverflow, incompleteNoding]. */
    status: string;
  };
};

/** GPU loop erasure operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPULineLoopRemovalOperation extends GPUContributorOperation {
  constructor(props: LineLoopRemovalOperationProps) {
    props = snapshotLineLoopRemovalOperationProps(props);
    if (props.lines.featureOffsets)
      validateCapacity(
        props.id ?? 'line-loop-removal',
        'featureCount',
        props.featureCount ?? Number.NaN,
        0
      );
    else if (props.featureCount !== undefined)
      validateRequirement(
        props.id ?? 'line-loop-removal',
        props.featureCount === props.lineCount,
        'featureCount must equal lineCount without input grouping'
      );
    super({
      id: props.id ?? 'line-loop-removal',
      inputs: {
        inputPositions: props.lines.positions,
        inputOffsets: props.lines.lineOffsets,
        ...(props.lines.featureOffsets ? {inputFeatures: props.lines.featureOffsets} : {})
      },
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
        status: {value: props.output.status, format: 'uint32x4', length: 1},
        ...(props.output.featureOffsets
          ? {
              featureOffsets: {
                value: props.output.featureOffsets,
                format: 'uint32' as const,
                length: (props.featureCount ?? props.lineCount) + 1
              }
            }
          : {})
      },
      createContributor: getView => {
        const features = props.lines.featureOffsets
          ? getView<'uint32'>('inputFeatures')
          : undefined;
        if (features)
          validateCapacity(
            props.id ?? 'line-loop-removal',
            'lines.featureOffsets.length',
            features.length,
            props.featureCount! + 1,
            props.featureCount! + 1
          );
        return new GPULineLoopRemoval({
          ...props,
          lines: {
            kind: 'lines',
            positions: getView<'float32x2'>('inputPositions'),
            lineOffsets: getView<'uint32'>('inputOffsets'),
            ...(features ? {featureOffsets: features} : {})
          },
          output: {
            positions: getView<'float32x2'>('positions'),
            lineOffsets: getView<'uint32'>('lineOffsets'),
            sourceEdges: getView<'uint32'>('sourceEdges'),
            status: getView<'uint32x4'>('status'),
            ...(props.output.featureOffsets
              ? {featureOffsets: getView<'uint32'>('featureOffsets')}
              : {})
          }
        });
      },
      emptyOutputPorts: ['positions', 'sourceEdges']
    });
  }
}

/** @internal Capture graph names and shapes once for both operation forms. */
export function snapshotLineLoopRemovalOperationProps(
  props: LineLoopRemovalOperationProps
): LineLoopRemovalOperationProps {
  const snapshot = snapshotConstructionFields(props, [
    'id',
    'spatialContext',
    'lines',
    'lineCount',
    'featureCount',
    'vertexCapacity',
    'intersectionCapacity',
    'output'
  ]);
  return Object.freeze({
    ...snapshot,
    lines: snapshotConstructionFields(snapshot.lines, [
      'positions',
      'lineOffsets',
      'featureOffsets'
    ]),
    output: snapshotConstructionFields(snapshot.output, [
      'positions',
      'lineOffsets',
      'featureOffsets',
      'sourceEdges',
      'status'
    ])
  });
}
