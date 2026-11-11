// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUTerrainBranchingFlow,
  type GPUTerrainBranchingFlowProps
} from './gpu-terrain-branching-flow';

/** Graph names in place of views; grid dimensions specify every cell output length. */
export type TerrainBranchingFlowOperationProps = Omit<
  GPUTerrainBranchingFlowProps,
  'fractions' | 'values' | 'edgeLengths' | 'validity' | 'output'
> & {
  fractions: string;
  values: string;
  edgeLengths?: string;
  validity?: string;
  output: {values: string; status: string; unresolvedCount: string};
};

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUTerrainBranchingFlowOperation extends GPUContributorOperation {
  constructor(props: TerrainBranchingFlowOperationProps) {
    const length = props.grid.width * props.grid.height;
    super({
      id: props.id ?? 'terrain-branching-flow',
      inputs: {
        fractions: props.fractions,
        values: props.values,
        ...(props.edgeLengths !== undefined ? {edgeLengths: props.edgeLengths} : {}),
        ...(props.validity !== undefined ? {validity: props.validity} : {})
      },
      outputs: {
        valuesOutput: {value: props.output.values, format: 'float32', length},
        status: {value: props.output.status, format: 'uint32', length},
        unresolvedCount: {value: props.output.unresolvedCount, format: 'uint32', length: 1}
      },
      createContributor: getView =>
        new GPUTerrainBranchingFlow({
          ...props,
          fractions: getView<'float32'>('fractions'),
          values: getView<'float32'>('values'),
          edgeLengths:
            props.edgeLengths !== undefined ? getView<'float32'>('edgeLengths') : undefined,
          validity: props.validity !== undefined ? getView<'uint32'>('validity') : undefined,
          output: {
            values: getView<'float32'>('valuesOutput'),
            status: getView<'uint32'>('status'),
            unresolvedCount: getView<'uint32'>('unresolvedCount')
          }
        })
    });
  }
}
