// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPUSpatialWeightsAlgebra,
  type GPUSpatialWeightsAlgebraProps,
  type GPUSpatialWeightsCombineRule
} from './gpu-spatial-weights-algebra';
import type {GPUSpatialWeights} from './spatial-weights';
import type {
  GPUSpaceTimeWeightsLayout,
  GPUSpatialWeightsSpaceTimeLag
} from './spatial-weights-space-time-stages';

type AlgebraInputs<Props = GPUSpatialWeightsAlgebraProps> =
  Props extends GPUSpatialWeightsAlgebraProps ? Omit<Props, 'output'> : never;

/** Graph names for one packed CSR, preserving optional context and metadata. @internal */
export type SpatialWeightsAlgebraGraphInput = Omit<
  GPUSpatialWeights,
  'offsets' | 'neighbors' | 'weights' | 'distances' | 'status'
> & {
  offsets: string;
  neighbors: string;
  weights: string;
  distances?: string;
};

/** Validates declared weights metadata before CPU graph execution. @internal */
export function validateSpatialWeightsOperationInformation(
  id: string,
  name: string,
  information: GPUSpatialWeights['information'],
  rowCount: number
): void {
  if (!information) return;
  validateRequirement(
    id,
    information.rowCount === rowCount,
    `${name}.information.rowCount must equal the row count`
  );
  validateRequirement(
    id,
    ['unknown', 'raw', 'binary', 'row', 'double', 'variance', 'kernel', 'symmetric'].includes(
      information.transform
    ),
    `${name}.information.transform is unknown`
  );
  validateRequirement(
    id,
    [true, false, 'unknown'].includes(information.symmetric),
    `${name}.information.symmetric must be a boolean or 'unknown'`
  );
}

/** Shared graph-value properties; rowCount is the output CSR row count. */
export type SpatialWeightsAlgebraOperationProps = {
  id?: string;
  rowCount: number;
  capacity: number;
  output: {
    offsets: string;
    neighbors: string;
    weights: string;
    distances?: string;
    status: {overflow: string; requiredCount?: string};
  };
} & (
  | {
      operation: 'union' | 'intersection' | 'difference' | 'symmetricDifference';
      left: SpatialWeightsAlgebraGraphInput;
      right: SpatialWeightsAlgebraGraphInput;
      weightRule?: GPUSpatialWeightsCombineRule;
    }
  | {
      operation: 'higherOrder';
      weights: SpatialWeightsAlgebraGraphInput;
      order: number;
      cumulative?: boolean;
      workCapacity?: number;
    }
  | {operation: 'selfWeight'; weights: SpatialWeightsAlgebraGraphInput; selfWeight: number | string}
  | {operation: 'subgraph'; weights: SpatialWeightsAlgebraGraphInput; mask: string}
  | {operation: 'block'; groupIds: string; groupCount: number}
  | {
      operation: 'spaceTime';
      weights: SpatialWeightsAlgebraGraphInput;
      periodCount: number;
      layout: GPUSpaceTimeWeightsLayout;
      lags: readonly GPUSpatialWeightsSpaceTimeLag[];
    }
);

/**
 * Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 * Every prop is compile-time except graph values. Operand status is caller-checked metadata,
 * not an input port. candidateOverflow is omitted: the contributor never writes that optional
 * port. As on the contributor, offsets[rowCount] is the clamped slot count (there is no count port).
 */
export class GPUSpatialWeightsAlgebraOperation extends GPUContributorOperation {
  constructor(props: SpatialWeightsAlgebraOperationProps) {
    const inputs: Record<string, string> = {};
    const addWeights = (prefix: string, weights: SpatialWeightsAlgebraGraphInput) => {
      for (const field of ['offsets', 'neighbors', 'weights', 'distances'] as const) {
        const value = weights[field];
        if (value !== undefined)
          inputs[`${prefix}${field[0].toUpperCase()}${field.slice(1)}`] = value;
      }
    };
    if ('left' in props) {
      addWeights('left', props.left);
      addWeights('right', props.right);
    } else if ('weights' in props) addWeights('source', props.weights);
    if (props.operation === 'selfWeight' && typeof props.selfWeight === 'string')
      inputs['selfWeight'] = props.selfWeight;
    if (props.operation === 'subgraph') inputs['mask'] = props.mask;
    if (props.operation === 'block') inputs['groupIds'] = props.groupIds;
    const outputs: Record<string, GPUContributorOperationOutput> = {
      offsets: {value: props.output.offsets, format: 'uint32', length: props.rowCount + 1},
      neighbors: {value: props.output.neighbors, format: 'uint32', length: props.capacity},
      weights: {value: props.output.weights, format: 'float32', length: props.capacity},
      overflow: {value: props.output.status.overflow, format: 'uint32', length: 1}
    };
    if (props.output.distances !== undefined)
      outputs['distances'] = {
        value: props.output.distances,
        format: 'float32',
        length: props.capacity
      };
    if (props.output.status.requiredCount !== undefined)
      outputs['requiredCount'] = {
        value: props.output.status.requiredCount,
        format: 'uint32',
        length: 1
      };
    super({
      id: props.id ?? 'spatial-weights-algebra',
      inputs,
      outputs,
      createContributor: getView => {
        const getWeights = (
          prefix: string,
          weights: SpatialWeightsAlgebraGraphInput
        ): GPUSpatialWeights => ({
          ...weights,
          offsets: getView<'uint32'>(`${prefix}Offsets`),
          neighbors: getView<'uint32'>(`${prefix}Neighbors`),
          weights: getView<'float32'>(`${prefix}Weights`),
          distances:
            weights.distances === undefined ? undefined : getView<'float32'>(`${prefix}Distances`)
        });
        let operation: AlgebraInputs;
        if ('left' in props)
          operation = {
            ...props,
            left: getWeights('left', props.left),
            right: getWeights('right', props.right)
          };
        else if (props.operation === 'block')
          operation = {...props, groupIds: getView<'uint32'>('groupIds')};
        else if (props.operation === 'selfWeight')
          operation = {
            ...props,
            weights: getWeights('source', props.weights),
            selfWeight:
              typeof props.selfWeight === 'number'
                ? props.selfWeight
                : getView<'float32'>('selfWeight')
          };
        else if (props.operation === 'subgraph')
          operation = {
            ...props,
            weights: getWeights('source', props.weights),
            mask: getView<'uint32'>('mask')
          };
        else operation = {...props, weights: getWeights('source', props.weights)};
        return new GPUSpatialWeightsAlgebra({
          ...operation,
          output: {
            offsets: getView<'uint32'>('offsets'),
            neighbors: getView<'uint32'>('neighbors'),
            weights: getView<'float32'>('weights'),
            distances:
              props.output.distances === undefined ? undefined : getView<'float32'>('distances'),
            status: {
              overflow: getView<'uint32'>('overflow'),
              requiredCount:
                props.output.status.requiredCount === undefined
                  ? undefined
                  : getView<'uint32'>('requiredCount')
            }
          }
        } as GPUSpatialWeightsAlgebraProps);
      }
    });
  }
}
