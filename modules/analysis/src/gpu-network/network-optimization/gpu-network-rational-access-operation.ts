// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUNetworkRationalAccess,
  type GPUNetworkRationalAccessProps
} from './gpu-network-rational-access';

/** Graph value names replacing views, shared by GPU and CPU rational-access operations. */
export type NetworkRationalAccessOperationProps = Pick<
  GPUNetworkRationalAccessProps,
  'id' | 'maximumCycles' | 'demandMode' | 'stepMode' | 'initialLimitCycles'
> & {
  originCount: number;
  facilityCount: number;
  costs: string;
  demand: string;
  supply: string;
  /** float32: tau, rho, initial step, minimum step, half-life, gap tolerance. */
  parameters: string;
  output: {assignments: string; scores: string; unservedDemand: string; status: string};
};

/** GPU rational access as a graph operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUNetworkRationalAccessOperation extends GPUContributorOperation {
  constructor(props: NetworkRationalAccessOperationProps) {
    super({
      id: props.id ?? 'network-rational-access',
      inputs: {
        costs: props.costs,
        demand: props.demand,
        supply: props.supply,
        parameters: props.parameters
      },
      outputs: {
        assignments: {
          value: props.output.assignments,
          format: 'float32',
          length: props.originCount * props.facilityCount
        },
        scores: {value: props.output.scores, format: 'float32', length: props.originCount},
        unservedDemand: {
          value: props.output.unservedDemand,
          format: 'float32',
          length: props.originCount
        },
        status: {value: props.output.status, format: 'uint32', length: 4}
      },
      createContributor: getView =>
        new GPUNetworkRationalAccess({
          ...props,
          costs: getView<'float32'>('costs'),
          demand: getView<'float32'>('demand'),
          supply: getView<'float32'>('supply'),
          parameters: getView<'float32'>('parameters'),
          output: {
            assignments: getView<'float32'>('assignments'),
            scores: getView<'float32'>('scores'),
            unservedDemand: getView<'float32'>('unservedDemand'),
            status: getView<'uint32'>('status')
          }
        })
    });
  }
}
