// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUSpatialLagImpacts, type GPUSpatialLagImpactsProps} from './gpu-spatial-lag-impacts';

/** Graph-value names in place of the contributor's views. Outputs have their minimum size. */
export type SpatialLagImpactsOperationProps = Pick<
  GPUSpatialLagImpactsProps,
  | 'id'
  | 'coefficientStride'
  | 'rhoIndex'
  | 'firstPredictorIndex'
  | 'predictorCount'
  | 'rowCount'
  | 'maximumPower'
  | 'rhoBound'
> & {
  coefficients: string;
  traces: {powerTraces: string; powerSums: string; infinityNorm: string};
  status?: string;
  output: {impacts: string};
};

/** GPU impacts as a graph operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUSpatialLagImpactsOperation extends GPUContributorOperation {
  constructor(props: SpatialLagImpactsOperationProps) {
    const id = props.id ?? 'spatial-lag-impacts';
    super({
      id,
      inputs: {
        coefficients: props.coefficients,
        powerTraces: props.traces.powerTraces,
        powerSums: props.traces.powerSums,
        infinityNorm: props.traces.infinityNorm,
        ...(props.status ? {status: props.status} : {})
      },
      outputs: {
        impacts: {value: props.output.impacts, format: 'float32', length: props.predictorCount * 3}
      },
      createContributor: getView =>
        new GPUSpatialLagImpacts({
          ...props,
          id,
          coefficients: getView<'float32'>('coefficients'),
          traces: {
            powerTraces: getView<'float32'>('powerTraces'),
            powerSums: getView<'float32'>('powerSums'),
            infinityNorm: getView<'float32'>('infinityNorm')
          },
          status: props.status ? getView<'uint32'>('status') : undefined,
          output: {impacts: getView<'float32'>('impacts')}
        })
    });
  }
}
