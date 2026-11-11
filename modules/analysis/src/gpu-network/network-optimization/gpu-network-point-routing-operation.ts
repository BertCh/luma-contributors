// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUNetworkPointRouting} from './gpu-network-point-routing';

/** Graph values with explicit edge and query counts for the predecessor matrix. */
export type NetworkPointRoutingOperationProps = {
  id?: string;
  edgeCount: number;
  queryCount: number;
  turns: string;
  costs: string;
  queries: string;
  fractions: string;
  maximumRelaxations?: number;
  output: {costs: string; status: string; predecessorEdges: string};
};
/** Vector operation form. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUNetworkPointRoutingOperation extends GPUContributorOperation {
  constructor(props: NetworkPointRoutingOperationProps) {
    super({
      id: props.id ?? 'network-point-routing',
      inputs: {
        turns: props.turns,
        edgeCosts: props.costs,
        queries: props.queries,
        fractions: props.fractions
      },
      outputs: {
        costs: {value: props.output.costs, format: 'float32', length: props.queryCount},
        status: {value: props.output.status, format: 'uint32', length: props.queryCount},
        predecessorEdges: {
          value: props.output.predecessorEdges,
          format: 'uint32',
          length: props.queryCount * props.edgeCount
        }
      },
      createContributor: getView =>
        new GPUNetworkPointRouting({
          id: props.id,
          turns: getView<'uint32x2'>('turns'),
          costs: getView<'float32'>('edgeCosts'),
          queries: getView<'uint32x2'>('queries'),
          fractions: getView<'float32x2'>('fractions'),
          maximumRelaxations: props.maximumRelaxations,
          output: {
            costs: getView<'float32'>('costs'),
            status: getView<'uint32'>('status'),
            predecessorEdges: getView<'uint32'>('predecessorEdges')
          }
        })
    });
  }
}
