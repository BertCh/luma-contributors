// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

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
  positions: string;
  ringOffsets: string;
  polygonOffsets: string;
  featureOffsets: string;
  featureCount: number;
  output: {points: string; distances?: string; degenerate?: string};
};

/** Operation form of GPULabelPoint. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPULabelPointOperation extends GPUContributorOperation {
  constructor(props: LabelPointOperationProps) {
    const id = props.id ?? 'label-point';
    super({
      id,
      inputs: {
        positions: props.positions,
        ringOffsets: props.ringOffsets,
        polygonOffsets: props.polygonOffsets,
        featureOffsets: props.featureOffsets
      },
      outputs: {
        points: {value: props.output.points, format: 'float32x2', length: props.featureCount},
        ...(props.output.distances === undefined
          ? {}
          : {
              distances: {
                value: props.output.distances,
                format: 'float32',
                length: props.featureCount
              }
            }),
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
            points: getView<'float32x2'>('points'),
            distances:
              props.output.distances === undefined ? undefined : getView<'float32'>('distances'),
            degenerate:
              props.output.degenerate === undefined ? undefined : getView<'uint32'>('degenerate')
          }
        })
    });
  }
}
