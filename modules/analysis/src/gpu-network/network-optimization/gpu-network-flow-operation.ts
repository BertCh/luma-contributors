// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getNetworkOptimizationStatusOutputs,
  type NetworkOptimizationStatusValues
} from './network-optimization-status';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUNetworkFlow, type GPUNetworkFlowProps} from './gpu-network-flow';

/** Shared operation properties, with graph value names replacing packed views. */
export type NetworkFlowOperationProps = Pick<
  GPUNetworkFlowProps,
  'id' | 'nodeCount' | 'maximumAugmentations'
> & {
  /** At least one slot: mixed-executor output vectors cannot be empty. Zero capacity is allowed. */
  edgeCount: number;
  edges: string;
  capacities: string;
  costs?: string;
  /** uint32: source, sink, required flow (0xffffffff maximizes). */
  parameters: string;
  output: {
    flow: string;
    cut: string;
    summary: string;
    exactCost?: string;
    status?: NetworkOptimizationStatusValues;
  };
};

/** GPU flow operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUNetworkFlowOperation extends GPUContributorOperation {
  constructor(props: NetworkFlowOperationProps) {
    super({
      id: props.id ?? 'network-flow',
      inputs: {
        edges: props.edges,
        capacities: props.capacities,
        parameters: props.parameters,
        ...(props.costs === undefined ? {} : {costs: props.costs})
      },
      outputs: {
        flow: {value: props.output.flow, format: 'uint32', length: props.edgeCount},
        cut: {value: props.output.cut, format: 'uint32', length: props.nodeCount},
        summary: {value: props.output.summary, format: 'uint32', length: 8},
        ...getNetworkOptimizationStatusOutputs(props.id ?? 'network-flow', props.output.status),
        ...(props.output.exactCost === undefined
          ? {}
          : {exactCost: {value: props.output.exactCost, format: 'uint32x2' as const, length: 1}})
      },
      createContributor: getView =>
        new GPUNetworkFlow({
          ...props,
          edges: getView<'uint32x2'>('edges'),
          capacities: getView<'uint32'>('capacities'),
          parameters: getView<'uint32'>('parameters'),
          costs: props.costs === undefined ? undefined : getView<'uint32'>('costs'),
          output: {
            flow: getView<'uint32'>('flow'),
            cut: getView<'uint32'>('cut'),
            summary: getView<'uint32'>('summary'),
            status: Object.fromEntries(
              Object.entries(props.output.status ?? {}).flatMap(([name, value]) =>
                value === undefined ? [] : [[name, getView<'uint32'>(name)]]
              )
            ),
            exactCost:
              props.output.exactCost === undefined ? undefined : getView<'uint32x2'>('exactCost')
          }
        })
    });
  }
}
