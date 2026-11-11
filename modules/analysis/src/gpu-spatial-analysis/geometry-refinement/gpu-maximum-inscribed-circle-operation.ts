// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUMaximumInscribedCircle,
  type GPUMaximumInscribedCircleProps
} from './gpu-maximum-inscribed-circle';

/** Graph value names in place of views; featureCount sizes every output. */
export type MaximumInscribedCircleOperationProps = Pick<
  GPUMaximumInscribedCircleProps,
  'id' | 'spatialContext' | 'tolerance' | 'cellCapacity' | 'maximumIterations'
> & {
  positions: string;
  ringOffsets: string;
  polygonOffsets: string;
  featureOffsets: string;
  featureCount: number;
  output: {circles: string; witnesses: string; status: string};
};

/** Operation form of GPUMaximumInscribedCircle; uses GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUMaximumInscribedCircleOperation extends GPUContributorOperation {
  constructor(props: MaximumInscribedCircleOperationProps) {
    const id = props.id ?? 'maximum-inscribed-circle';
    super({
      id,
      inputs: {
        positions: props.positions,
        ringOffsets: props.ringOffsets,
        polygonOffsets: props.polygonOffsets,
        featureOffsets: props.featureOffsets
      },
      outputs: {
        circles: {value: props.output.circles, format: 'float32x4', length: props.featureCount},
        witnesses: {value: props.output.witnesses, format: 'float32x2', length: props.featureCount},
        status: {value: props.output.status, format: 'uint32', length: props.featureCount}
      },
      createContributor: getView =>
        new GPUMaximumInscribedCircle({
          ...props,
          id,
          positions: getView<'float32x2'>('positions'),
          ringOffsets: getView<'uint32'>('ringOffsets'),
          polygonOffsets: getView<'uint32'>('polygonOffsets'),
          featureOffsets: getView<'uint32'>('featureOffsets'),
          output: {
            circles: getView<'float32x4'>('circles'),
            witnesses: getView<'float32x2'>('witnesses'),
            status: getView<'uint32'>('status')
          }
        })
    });
  }
}
