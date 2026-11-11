// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import type {GPUSpatialWeightsInformation} from '../spatial-weights/spatial-weights';
import {
  GPUNeighborhoodSummary,
  type GPUNeighborhoodSummaryStatistic
} from './gpu-neighborhood-summary';

/** Graph-value properties shared by both neighborhood operations. */
export type NeighborhoodSummaryOperationProps = {
  id?: string;
  rowCount: number;
  /** Optional distances and unused status ports are omitted: the summary never reads them. */
  weights: {
    offsets: string;
    neighbors: string;
    weights: string;
    information?: GPUSpatialWeightsInformation;
    status?: {overflow?: string};
  };
  values?: string;
  valuesFormat?: 'float32' | 'uint32';
  categories?: string;
  mask?: string;
  includeFocal?: boolean;
  focalWeight?: number;
  statistics?: readonly GPUNeighborhoodSummaryStatistic[];
  maximumNeighbors?: number;
  output: {table?: string; modes?: string; entropy?: string; status?: {overflow: string}};
};

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUNeighborhoodSummaryOperation extends GPUContributorOperation {
  constructor(props: NeighborhoodSummaryOperationProps) {
    const outputs: Record<string, GPUContributorOperationOutput> = {};
    for (const port of ['table', 'modes', 'entropy'] as const) {
      const value = props.output[port];
      if (value !== undefined)
        outputs[port] = {
          value,
          format: port === 'modes' ? 'uint32' : 'float32',
          length: props.rowCount * (port === 'table' ? (props.statistics ?? []).length : 1)
        };
    }
    if (props.output.status)
      outputs['overflow'] = {value: props.output.status.overflow, format: 'uint32', length: 1};
    super({
      id: props.id ?? 'neighborhood-summary',
      inputs: {
        offsets: props.weights.offsets,
        neighbors: props.weights.neighbors,
        weights: props.weights.weights,
        ...(props.weights.status?.overflow !== undefined
          ? {weightsOverflow: props.weights.status.overflow}
          : {}),
        ...(props.values !== undefined ? {values: props.values} : {}),
        ...(props.categories !== undefined ? {categories: props.categories} : {}),
        ...(props.mask !== undefined ? {mask: props.mask} : {})
      },
      outputs,
      createContributor: getView =>
        new GPUNeighborhoodSummary({
          id: props.id,
          weights: {
            offsets: getView<'uint32'>('offsets'),
            neighbors: getView<'uint32'>('neighbors'),
            weights: getView<'float32'>('weights'),
            information: props.weights.information,
            status:
              props.weights.status?.overflow !== undefined
                ? {overflow: getView<'uint32'>('weightsOverflow')}
                : undefined
          },
          values: props.values !== undefined ? getView<'float32' | 'uint32'>('values') : undefined,
          categories: props.categories !== undefined ? getView<'uint32'>('categories') : undefined,
          mask: props.mask !== undefined ? getView<'uint32'>('mask') : undefined,
          includeFocal: props.includeFocal,
          focalWeight: props.focalWeight,
          statistics: props.statistics,
          maximumNeighbors: props.maximumNeighbors,
          output: {
            table: props.output.table !== undefined ? getView<'float32'>('table') : undefined,
            modes: props.output.modes !== undefined ? getView<'uint32'>('modes') : undefined,
            entropy: props.output.entropy !== undefined ? getView<'float32'>('entropy') : undefined,
            status:
              props.output.status !== undefined
                ? {overflow: getView<'uint32'>('overflow')}
                : undefined
          }
        })
    });
  }
}
