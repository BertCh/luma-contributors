// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPULabelPoint, type GPULabelPointProps} from './gpu-label-point';

/** Graph value names in place of polygon views; featureCount sizes every output. */
export type LabelPointOperationProps = Pick<
  GPULabelPointProps,
  | 'id'
  | 'spatialContext'
  | 'initialGridSize'
  | 'refinementGridSize'
  | 'refinementRounds'
  | 'refinementCandidates'
  | 'smallFeatureVertexLimit'
> & {
  /** CPU host interpretation, defaulting to resident-capacity for graph inputs. */
  storage?: 'compact' | 'resident-capacity';
  positions: string;
  ringOffsets: string;
  polygonOffsets: string;
  featureOffsets: string;
  featureCount: number;
  output: {
    points: string;
    publishedCenterClearance?: string;
    precisionLimited?: string;
    degenerate?: string;
  };
};

/** Captures topology and output names for either backend. @internal */
export function snapshotLabelPointOperationProps(
  props: LabelPointOperationProps
): LabelPointOperationProps {
  const captured = snapshotConstructionFields(props, [
    'id',
    'spatialContext',
    'initialGridSize',
    'refinementGridSize',
    'refinementRounds',
    'refinementCandidates',
    'smallFeatureVertexLimit',
    'storage',
    'positions',
    'ringOffsets',
    'polygonOffsets',
    'featureOffsets',
    'featureCount',
    'output'
  ]);
  validateRequirement(
    captured.id ?? 'label-point',
    !('distances' in captured.output),
    'output.distances was removed; use output.publishedCenterClearance'
  );
  return Object.freeze({
    ...captured,
    ...(captured.spatialContext
      ? {spatialContext: snapshotConstructionSpatialContext(captured.spatialContext)}
      : {}),
    output: snapshotConstructionFields(captured.output, [
      'points',

      'publishedCenterClearance',
      'precisionLimited',
      'degenerate'
    ])
  });
}

/** Operation form of GPULabelPoint. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPULabelPointOperation extends GPUContributorOperation {
  constructor(props: LabelPointOperationProps) {
    props = snapshotLabelPointOperationProps(props);
    const id = props.id ?? 'label-point';
    validateRequirement(
      id,
      props.storage === undefined || ['compact', 'resident-capacity'].includes(props.storage),
      `storage must be compact or resident-capacity; received ${props.storage}`
    );
    super({
      emptyOutputPorts: [
        'points',

        'publishedCenterClearance',
        'precisionLimited',
        'degenerate'
      ],
      id,
      inputs: {
        positions: props.positions,
        ringOffsets: props.ringOffsets,
        polygonOffsets: props.polygonOffsets,
        featureOffsets: props.featureOffsets
      },
      outputs: {
        ...(props.output.publishedCenterClearance === undefined
          ? {}
          : {
              publishedCenterClearance: {
                value: props.output.publishedCenterClearance,
                format: 'float32',
                length: props.featureCount
              }
            }),
        ...(props.output.precisionLimited === undefined
          ? {}
          : {
              precisionLimited: {
                value: props.output.precisionLimited,
                format: 'uint32',
                length: props.featureCount
              }
            }),
        points: {value: props.output.points, format: 'float32x2', length: props.featureCount},
        ...(props.output.degenerate === undefined
          ? {}
          : {
              degenerate: {
                value: props.output.degenerate,
                format: 'uint32',
                length: props.featureCount
              }
            })
      },
      createContributor: getView =>
        new GPULabelPoint({
          ...props,
          id,
          positions: getView<'float32x2'>('positions'),
          ringOffsets: getView<'uint32'>('ringOffsets'),
          polygonOffsets: getView<'uint32'>('polygonOffsets'),
          featureOffsets: getView<'uint32'>('featureOffsets'),
          output: {
            publishedCenterClearance:
              props.output.publishedCenterClearance === undefined
                ? undefined
                : getView<'float32'>('publishedCenterClearance'),
            precisionLimited:
              props.output.precisionLimited === undefined
                ? undefined
                : getView<'uint32'>('precisionLimited'),
            points: getView<'float32x2'>('points'),
            degenerate:
              props.output.degenerate === undefined ? undefined : getView<'uint32'>('degenerate')
          }
        })
    });
  }
}
