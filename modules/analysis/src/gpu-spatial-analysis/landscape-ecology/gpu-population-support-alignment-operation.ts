// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUPopulationSupportAlignment} from './gpu-population-support-alignment';

/** Logical graph values for support alignment. All output rows follow the target support. */
export type PopulationSupportAlignmentOperationProps = {
  id?: string;
  populations: string;
  sourceIds: string;
  targetIds: string;
  /** Positive number of target IDs; determines the output lengths. */
  targetCount: number;
  output: {populations: string; validity: string; status: string};
};

/** GPU support-ID join in a mixed GPUGraph; use GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUPopulationSupportAlignmentOperation extends GPUContributorOperation {
  constructor(props: PopulationSupportAlignmentOperationProps) {
    super({
      id: props.id ?? 'population-support-alignment',
      inputs: {
        inputPopulations: props.populations,
        sourceIds: props.sourceIds,
        targetIds: props.targetIds
      },
      outputs: {
        populations: {
          value: props.output.populations,
          format: 'float32x2',
          length: props.targetCount
        },
        validity: {value: props.output.validity, format: 'uint32', length: props.targetCount},
        status: {value: props.output.status, format: 'uint32x4', length: 1}
      },
      createContributor: getView =>
        new GPUPopulationSupportAlignment({
          id: props.id,
          populations: getView<'float32x2'>('inputPopulations'),
          sourceIds: getView<'uint32'>('sourceIds'),
          targetIds: getView<'uint32'>('targetIds'),
          output: {
            populations: getView<'float32x2'>('populations'),
            validity: getView<'uint32'>('validity'),
            status: getView<'uint32x4'>('status')
          }
        })
    });
  }
}
