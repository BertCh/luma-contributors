// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPUOffsetCurveComponents,
  type GPUOffsetCurveComponentsProps
} from './gpu-offset-curve-components';

/** Graph names and explicit output lengths; all properties are compile-time. */
export type OffsetCurveComponentsOperationProps = Pick<
  GPUOffsetCurveComponentsProps,
  'id' | 'distance' | 'tolerance' | 'rowsPerVertex' | 'intersectionCapacity' | 'segmentCapacity'
> & {
  original: {positions: string; lineOffsets: string; featureOffsets?: string};
  raw: {positions: string; lineOffsets: string; featureOffsets?: string};
  lineCount: number;
  vertexCapacity: number;
  componentCapacity: number;
  output: {positions: string; components: string; sourceEdges: string; status: string};
};

/** GPU component operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUOffsetCurveComponentsOperation extends GPUContributorOperation {
  constructor(props: OffsetCurveComponentsOperationProps) {
    const snapshot = snapshotConstructionFields(props, [
      'id',
      'original',
      'raw',
      'lineCount',
      'distance',
      'tolerance',
      'rowsPerVertex',
      'intersectionCapacity',
      'segmentCapacity',
      'vertexCapacity',
      'componentCapacity',
      'output'
    ]);
    props = Object.freeze({
      ...snapshot,
      original: snapshotConstructionFields(snapshot.original, [
        'positions',
        'lineOffsets',
        'featureOffsets'
      ]),
      raw: snapshotConstructionFields(snapshot.raw, ['positions', 'lineOffsets', 'featureOffsets']),
      output: snapshotConstructionFields(snapshot.output, [
        'positions',
        'components',
        'sourceEdges',
        'status'
      ])
    });
    validateRequirement(
      props.id ?? 'offset-curve-components',
      Boolean(props.original.featureOffsets) === Boolean(props.raw.featureOffsets),
      'original.featureOffsets and raw.featureOffsets must both be absent or both name identical feature-to-part grouping; provide both input values'
    );
    super({
      id: props.id ?? 'offset-curve-components',
      inputs: {
        originalPositions: props.original.positions,
        originalOffsets: props.original.lineOffsets,
        rawPositions: props.raw.positions,
        rawOffsets: props.raw.lineOffsets,
        ...(props.original.featureOffsets ? {originalFeatures: props.original.featureOffsets} : {}),
        ...(props.raw.featureOffsets ? {rawFeatures: props.raw.featureOffsets} : {})
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
            lineOffsets: getView<'uint32'>('originalOffsets'),
            featureOffsets: props.original.featureOffsets
              ? getView<'uint32'>('originalFeatures')
              : undefined
          },
          raw: {
            positions: getView<'float32x2'>('rawPositions'),
            lineOffsets: getView<'uint32'>('rawOffsets'),
            featureOffsets: props.raw.featureOffsets ? getView<'uint32'>('rawFeatures') : undefined
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
