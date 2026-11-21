// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateSamplingSeed} from './sampling-seed';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {validateViewLength} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPUSpatialBalancedSample,
  type GPUSpatialBalancedSampleProps
} from './gpu-spatial-balanced-sample';

/** Graph-value props shared by the GPU operation and its CPU twin. */
export type SpatialBalancedSampleOperationProps = Pick<
  GPUSpatialBalancedSampleProps,
  'id' | 'spatialContext' | 'allocations' | 'bounds' | 'seed' | 'replicate'
> & {
  rowCount: number;
  positions: string;
  frameIds: string;
  strata?: string;
  inclusionTickets?: string;
  /** uint32 [seed, replicate], exactly as in the contributor. */
  parameters?: string;
  output: {
    frameIds: string;
    inclusionProbabilities: string;
    designWeights: string;
    status: string;
  };
};

/** @internal Capture schema, allocations and population settings for both backends. */
export function snapshotBalancedSampleOperation(
  props: SpatialBalancedSampleOperationProps
): SpatialBalancedSampleOperationProps {
  const snapshot = snapshotConstructionFields(props, [
    'id',
    'spatialContext',
    'allocations',
    'bounds',
    'seed',
    'replicate',
    'rowCount',
    'positions',
    'frameIds',
    'strata',
    'inclusionTickets',
    'parameters',
    'output'
  ]);
  return Object.freeze({
    ...snapshot,
    allocations: Object.freeze(
      snapshot.allocations.map(allocation =>
        snapshotConstructionFields(allocation, ['stratumId', 'sampleCount'])
      )
    ),
    bounds: Object.freeze([...snapshot.bounds]) as typeof snapshot.bounds,
    output: snapshotConstructionFields(snapshot.output, [
      'frameIds',
      'inclusionProbabilities',
      'designWeights',
      'status'
    ])
  });
}

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUSpatialBalancedSampleOperation extends GPUContributorOperation {
  constructor(props: SpatialBalancedSampleOperationProps) {
    props = snapshotBalancedSampleOperation(props);
    const id = props.id ?? 'spatial-balanced-sample';
    validateSamplingSeed(id, props.seed, props.parameters !== undefined);
    const rowCount = props.rowCount;
    const length = props.allocations.reduce(
      (total, allocation) => total + allocation.sampleCount,
      0
    );
    super({
      id: props.id ?? 'spatial-balanced-sample',
      inputs: {
        positions: props.positions,
        frameIds: props.frameIds,
        ...(props.strata !== undefined ? {strata: props.strata} : {}),
        ...(props.inclusionTickets !== undefined ? {inclusionTickets: props.inclusionTickets} : {}),
        ...(props.parameters !== undefined ? {parameters: props.parameters} : {})
      },
      outputs: {
        frameIdsOutput: {value: props.output.frameIds, format: 'uint32', length},
        inclusionProbabilities: {
          value: props.output.inclusionProbabilities,
          format: 'float32',
          length
        },
        designWeights: {value: props.output.designWeights, format: 'float32', length},
        status: {value: props.output.status, format: 'uint32', length: props.allocations.length * 2}
      },
      createContributor: getView => {
        const positions = getView<'float32x2'>('positions');
        validateViewLength(id, 'positions (rowCount)', positions, rowCount);
        return new GPUSpatialBalancedSample({
          id: props.id,
          spatialContext: props.spatialContext,
          positions,
          frameIds: getView<'uint32'>('frameIds'),
          strata: props.strata !== undefined ? getView<'uint32'>('strata') : undefined,
          inclusionTickets:
            props.inclusionTickets !== undefined
              ? getView<'uint32'>('inclusionTickets')
              : undefined,
          parameters: props.parameters !== undefined ? getView<'uint32'>('parameters') : undefined,
          allocations: props.allocations,
          bounds: props.bounds,
          seed: props.seed,
          replicate: props.replicate,
          output: {
            frameIds: getView<'uint32'>('frameIdsOutput'),
            inclusionProbabilities: getView<'float32'>('inclusionProbabilities'),
            designWeights: getView<'float32'>('designWeights'),
            status: getView<'uint32'>('status')
          }
        });
      }
    });
  }
}
