// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPURadiationFlowModel, type GPURadiationFlowModelProps} from './gpu-radiation-flow-model';

/** Shared graph-value properties of the GPU operation and its CPU twin. */
export type RadiationFlowModelOperationProps = Pick<
  GPURadiationFlowModelProps,
  'id' | 'spatialContext' | 'normalization'
> & {
  /** Compile-time number of zones, inferred from positions by the contributor. */
  zoneCount: number;
  /** float32x2 centroids. */
  positions: string;
  /** float32 populations and departing trip totals, one row per zone. */
  populations: string;
  outflows: string;
  output: {flows: string};
};

/**
 * Operation form of GPURadiationFlowModel. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 * All properties are compile-time; input graph contents are per-frame.
 */
export class GPURadiationFlowModelOperation extends GPUContributorOperation {
  constructor(props: RadiationFlowModelOperationProps) {
    const id = props.id ?? 'radiation-flow-model';
    super({
      id,
      inputs: {
        positions: props.positions,
        populations: props.populations,
        outflows: props.outflows
      },
      outputs: {
        flows: {value: props.output.flows, format: 'float32', length: props.zoneCount ** 2}
      },
      createContributor: getView =>
        new GPURadiationFlowModel({
          id,
          spatialContext: props.spatialContext,
          normalization: props.normalization,
          positions: getView<'float32x2'>('positions'),
          populations: getView<'float32'>('populations'),
          outflows: getView<'float32'>('outflows'),
          output: {flows: getView<'float32'>('flows')}
        })
    });
  }
}
