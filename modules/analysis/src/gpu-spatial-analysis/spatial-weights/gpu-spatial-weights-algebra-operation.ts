// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {snapshotWeightsAlgebraProps} from './weights-construction-snapshot';
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

/** Graph input used by algebra, including optional upstream loss values. */
type LossAwareAlgebraGraphInput = SpatialWeightsAlgebraGraphInput & {
  status?: {overflow?: string; candidateOverflow?: string};
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
    status: {overflow: string; requiredCount?: string; candidateOverflow?: string};
  };
} & (
  | {
      operation: 'union' | 'intersection' | 'difference' | 'symmetricDifference';
      left: LossAwareAlgebraGraphInput;
      right: LossAwareAlgebraGraphInput;
      weightRule?: GPUSpatialWeightsCombineRule;
    }
  | {
      operation: 'higherOrder';
      weights: LossAwareAlgebraGraphInput;
      order: number;
      cumulative?: boolean;
      workCapacity?: number;
    }
  | {operation: 'selfWeight'; weights: LossAwareAlgebraGraphInput; selfWeight: number | string}
  | {operation: 'subgraph'; weights: LossAwareAlgebraGraphInput; mask: string}
  | {operation: 'block'; groupIds: string; groupCount: number}
  | {
      operation: 'spaceTime';
      weights: LossAwareAlgebraGraphInput;
      periodCount: number;
      layout: GPUSpaceTimeWeightsLayout;
      lags: readonly GPUSpatialWeightsSpaceTimeLag[];
    }
);

/**
 * Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 * Every prop is compile-time except graph values. Optional operand loss ports are read and
 * propagated to candidateOverflow, which must have a named destination whenever input loss
 * ports or higher-order intermediate work are present. Overflow is destination truncation only.
 * As on the contributor,
 * offsets[rowCount] is the clamped slot count (there is no count port).
 */
export class GPUSpatialWeightsAlgebraOperation extends GPUContributorOperation {
  constructor(props: SpatialWeightsAlgebraOperationProps) {
    props = snapshotWeightsAlgebraProps(props);
    const operands =
      'left' in props ? [props.left, props.right] : 'weights' in props ? [props.weights] : [];
    validateRequirement(
      props.id ?? 'spatial-weights-algebra',
      Boolean(props.output.status.candidateOverflow) ||
        !(
          (props.operation === 'higherOrder' && props.order > 1) ||
          operands.some(operand => operand.status?.overflow || operand.status?.candidateOverflow)
        ),
      'output.status.candidateOverflow is required for operand loss or bounded intermediate work'
    );
    const inputs: Record<string, string> = {};
    const addWeights = (prefix: string, weights: LossAwareAlgebraGraphInput) => {
      for (const name of ['overflow', 'candidateOverflow'] as const)
        if (weights.status?.[name]) {
          inputs[`${prefix}${name[0].toUpperCase()}${name.slice(1)}`] = weights.status[name]!;
        }
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
    if (props.output.status.candidateOverflow)
      outputs['candidateOverflow'] = {
        value: props.output.status.candidateOverflow,
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
          weights: LossAwareAlgebraGraphInput
        ): GPUSpatialWeights => ({
          ...weights,
          offsets: getView<'uint32'>(`${prefix}Offsets`),
          neighbors: getView<'uint32'>(`${prefix}Neighbors`),
          weights: getView<'float32'>(`${prefix}Weights`),
          distances:
            weights.distances === undefined ? undefined : getView<'float32'>(`${prefix}Distances`),
          status: {
            overflow: weights.status?.overflow ? getView<'uint32'>(`${prefix}Overflow`) : undefined,
            candidateOverflow: weights.status?.candidateOverflow
              ? getView<'uint32'>(`${prefix}CandidateOverflow`)
              : undefined
          }
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
              candidateOverflow: props.output.status.candidateOverflow
                ? getView<'uint32'>('candidateOverflow')
                : undefined,
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
