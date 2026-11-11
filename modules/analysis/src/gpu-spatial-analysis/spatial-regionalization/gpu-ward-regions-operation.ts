// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUWardRegions, GPU_WARD_STATUS, type GPUWardMergeSchedule} from './gpu-ward-regions';

/** Whole-CSR Ward graph values. Unused distances and weights metadata are omitted. */
export type WardRegionsOperationProps = {
  id?: string;
  weights: {offsets: string; neighbors: string; weights: string};
  /** Explicit count normally inferred from offsets. */
  rowCount: number;
  values: string;
  columnCount?: number;
  standardize?: boolean;
  targetRegionCount: number;
  maximumIterations: number;
  mergeSchedule?: GPUWardMergeSchedule;
  /** uint32 words from `getGPUWardRegionsParameterValues`. */
  parameters: string;
  /** Status words: four by default, or five to include the mutual-nearest round count. */
  statusLength?: 4 | 5;
  output: {labels: string; objective: string; status: string};
};

/** Ward contributor operation. Compile with `GPU_CONTRIBUTOR_OPERATION_LOWERING`. */
export class GPUWardRegionsOperation extends GPUContributorOperation {
  constructor(props: WardRegionsOperationProps) {
    const id = props.id ?? 'ward-regions';
    super({
      id,
      inputs: {...props.weights, values: props.values, parameters: props.parameters},
      outputs: {
        labels: {value: props.output.labels, format: 'uint32', length: props.rowCount},
        objective: {value: props.output.objective, format: 'float32', length: 1},
        status: {
          value: props.output.status,
          format: 'uint32',
          length: props.statusLength ?? GPU_WARD_STATUS.length
        }
      },
      createContributor: getView =>
        new GPUWardRegions({
          ...props,
          id,
          weights: {
            offsets: getView<'uint32'>('offsets'),
            neighbors: getView<'uint32'>('neighbors'),
            weights: getView<'float32'>('weights')
          },
          values: getView<'float32'>('values'),
          parameters: getView<'uint32'>('parameters'),
          output: {
            labels: getView<'uint32'>('labels'),
            objective: getView<'float32'>('objective'),
            status: getView<'uint32'>('status')
          }
        })
    });
  }
}
