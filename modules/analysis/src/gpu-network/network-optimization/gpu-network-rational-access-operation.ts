// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {withoutComplete, validateRationalAccessOptions} from './optimization-cost-completeness';
import {
  getNetworkOptimizationStatusOutputs,
  type NetworkOptimizationStatusValues
} from './network-optimization-status';
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
  costsComplete?: string;
  costLimit?: string;
  demand: string;
  supply: string;
  /** float32: tau, rho, initial step, minimum step, half-life, gap tolerance. */
  parameters: string;
  output: {
    assignments: string;
    scores: string;
    unservedDemand: string;
    summary: string;
    status?: NetworkOptimizationStatusValues & {complete?: string};
  };
};

/** GPU rational access as a graph operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUNetworkRationalAccessOperation extends GPUContributorOperation {
  constructor(props: NetworkRationalAccessOperationProps) {
    validateRationalAccessOptions(props.id ?? 'network-rational-access', props);
    validateRequirement(
      props.id ?? 'network-rational-access',
      (props.costsComplete === undefined && props.costLimit === undefined) ||
        props.output.status?.complete !== undefined,
      'costsComplete or costLimit requires output.status.complete; inspect it before using the decision'
    );
    super({
      id: props.id ?? 'network-rational-access',
      inputs: {
        costs: props.costs,
        ...(props.costsComplete !== undefined ? {costsComplete: props.costsComplete} : {}),
        ...(props.costLimit !== undefined ? {costLimit: props.costLimit} : {}),
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
        ...(props.output.status?.complete !== undefined
          ? {complete: {value: props.output.status.complete, format: 'uint32' as const, length: 1}}
          : {}),
        summary: {value: props.output.summary, format: 'uint32', length: 4},
        ...getNetworkOptimizationStatusOutputs(
          props.id ?? 'network-rational-access',
          withoutComplete(props.output.status)
        )
      },
      createContributor: getView =>
        new GPUNetworkRationalAccess({
          ...props,
          costs: getView<'float32'>('costs'),
          costsComplete:
            props.costsComplete !== undefined ? getView<'uint32'>('costsComplete') : undefined,
          costLimit: props.costLimit !== undefined ? getView<'float32'>('costLimit') : undefined,
          demand: getView<'float32'>('demand'),
          supply: getView<'float32'>('supply'),
          parameters: getView<'float32'>('parameters'),
          output: {
            assignments: getView<'float32'>('assignments'),
            scores: getView<'float32'>('scores'),
            unservedDemand: getView<'float32'>('unservedDemand'),
            summary: getView<'uint32'>('summary'),
            status: Object.fromEntries(
              Object.entries(props.output.status ?? {}).flatMap(([name, value]) =>
                value === undefined ? [] : [[name, getView<'uint32'>(name)]]
              )
            )
          }
        })
    });
  }
}
