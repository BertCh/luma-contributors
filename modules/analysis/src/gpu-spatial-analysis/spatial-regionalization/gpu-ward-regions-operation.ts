// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {GPUWardRegions, GPU_WARD_STATUS, type GPUWardMergeSchedule} from './gpu-ward-regions';

/** Whole-CSR Ward graph values, including optional adjacency loss scalars. */
export type WardRegionsOperationProps = {
  id?: string;
  weights: {
    offsets: string;
    neighbors: string;
    weights: string;
    status?: {overflow?: string; candidateOverflow?: string};
  };
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

/** @internal Capture declared graph names and fit options, including inherited getters. */
export function snapshotWardOperationProps(
  props: WardRegionsOperationProps
): WardRegionsOperationProps {
  const snapshot = snapshotConstructionFields(props, [
    'id',
    'weights',
    'rowCount',
    'values',
    'columnCount',
    'standardize',
    'targetRegionCount',
    'maximumIterations',
    'mergeSchedule',
    'parameters',
    'statusLength',
    'output'
  ]);
  const weights = snapshotConstructionFields(snapshot.weights, [
    'offsets',
    'neighbors',
    'weights',
    'status'
  ]);
  return Object.freeze({
    ...snapshot,
    weights: Object.freeze({
      ...weights,
      status: weights.status
        ? snapshotConstructionFields(weights.status, ['overflow', 'candidateOverflow'])
        : undefined
    }),
    output: snapshotConstructionFields(snapshot.output, ['labels', 'objective', 'status'])
  });
}

/** Ward contributor operation. Compile with `GPU_CONTRIBUTOR_OPERATION_LOWERING`. */
export class GPUWardRegionsOperation extends GPUContributorOperation {
  constructor(props: WardRegionsOperationProps) {
    props = snapshotWardOperationProps(props);
    const id = props.id ?? 'ward-regions';
    super({
      id,
      inputs: {
        offsets: props.weights.offsets,
        neighbors: props.weights.neighbors,
        weights: props.weights.weights,
        ...(props.weights.status?.overflow ? {weightsOverflow: props.weights.status.overflow} : {}),
        ...(props.weights.status?.candidateOverflow
          ? {weightsCandidateOverflow: props.weights.status.candidateOverflow}
          : {}),
        values: props.values,
        parameters: props.parameters
      },
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
            weights: getView<'float32'>('weights'),
            status: {
              overflow: props.weights.status?.overflow
                ? getView<'uint32'>('weightsOverflow')
                : undefined,
              candidateOverflow: props.weights.status?.candidateOverflow
                ? getView<'uint32'>('weightsCandidateOverflow')
                : undefined
            }
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
