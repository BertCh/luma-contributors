// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUCoregionalizationFit} from './gpu-coregionalization-fit';
import type {CoregionalizationFitOptions} from './coregionalization-parameters';

/** Same logical values and props for the GPU operation and its CPU twin. */
export type CoregionalizationFitOperationProps = CoregionalizationFitOptions & {
  id?: string;
  distances: string;
  semivariances: string;
  pairCounts: string;
  parameters: string;
  output: {
    matrices: string;
    loadings: string;
    diagnostics: string;
    status: {
      validity: string;
      iterationCount: string;
      converged: string;
      iterationLimitReached: string;
      invalidCount: string;
    };
  };
};

/**
 * LMC fitter as a GPUGraph operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 * Compile-time: shape and sweep budget. Per-frame: input graph values.
 */
export class GPUCoregionalizationFitOperation extends GPUContributorOperation {
  constructor(props: CoregionalizationFitOperationProps) {
    const id = props.id ?? 'coregionalization-fit';
    const length = props.structureCount * props.variableCount ** 2;
    super({
      id,
      inputs: {
        distances: props.distances,
        semivariances: props.semivariances,
        pairCounts: props.pairCounts,
        parameters: props.parameters
      },
      outputs: {
        matrices: {value: props.output.matrices, format: 'float32', length},
        loadings: {value: props.output.loadings, format: 'float32', length},
        diagnostics: {value: props.output.diagnostics, format: 'float32x4', length: 1},
        ...Object.fromEntries(
          Object.entries(props.output.status).map(([name, value]) => [
            name,
            {value, format: 'uint32' as const, length: 1}
          ])
        )
      },
      createContributor: getView =>
        new GPUCoregionalizationFit({
          ...props,
          distances: getView<'float32'>('distances'),
          semivariances: getView<'float32'>('semivariances'),
          pairCounts: getView<'uint32'>('pairCounts'),
          parameters: getView<'float32'>('parameters'),
          output: {
            matrices: getView<'float32'>('matrices'),
            loadings: getView<'float32'>('loadings'),
            diagnostics: getView<'float32x4'>('diagnostics'),
            status: {
              validity: getView<'uint32'>('validity'),
              iterationCount: getView<'uint32'>('iterationCount'),
              converged: getView<'uint32'>('converged'),
              iterationLimitReached: getView<'uint32'>('iterationLimitReached'),
              invalidCount: getView<'uint32'>('invalidCount')
            }
          }
        })
    });
  }
}
