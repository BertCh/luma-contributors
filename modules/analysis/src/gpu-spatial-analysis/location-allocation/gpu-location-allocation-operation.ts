// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPULocationAllocation,
  GPU_LOCATION_ALLOCATION_STATUS,
  type GPULocationAllocationOperation as LocationAllocationObjective
} from './gpu-location-allocation';

/** Packed vector graph properties shared by the GPU operation and CPU twin. */
export type LocationAllocationOperationProps = {
  id?: string;
  demandCount: number;
  facilityCount: number;
  costs?: string;
  sparseCosts?: {offsets: string; facilities: string; costs: string};
  demandWeights?: string;
  operation: LocationAllocationObjective;
  maximumFacilityCount: number;
  maximumSwapIterations?: number;
  /** float32 graph value holding the per-frame threshold in its first row. */
  coverageDistance?: string;
  output: {
    selectedFacilities: string;
    assignments: string;
    assignedCosts?: string;
    objective: string;
    status: string;
  };
};

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPULocationAllocationGraphOperation extends GPUContributorOperation {
  constructor(props: LocationAllocationOperationProps) {
    super({
      id: props.id ?? 'location-allocation',
      inputs: {
        ...(props.costs === undefined ? {} : {costs: props.costs}),
        ...(props.sparseCosts === undefined
          ? {}
          : {
              sparseOffsets: props.sparseCosts.offsets,
              sparseFacilities: props.sparseCosts.facilities,
              sparseCosts: props.sparseCosts.costs
            }),
        ...(props.demandWeights === undefined ? {} : {demandWeights: props.demandWeights}),
        ...(props.coverageDistance === undefined ? {} : {coverageDistance: props.coverageDistance})
      },
      outputs: {
        selectedFacilities: {
          value: props.output.selectedFacilities,
          format: 'uint32',
          length: props.facilityCount
        },
        assignments: {value: props.output.assignments, format: 'uint32', length: props.demandCount},
        ...(props.output.assignedCosts === undefined
          ? {}
          : {
              assignedCosts: {
                value: props.output.assignedCosts,
                format: 'float32' as const,
                length: props.demandCount
              }
            }),
        objective: {value: props.output.objective, format: 'float32', length: 1},
        status: {
          value: props.output.status,
          format: 'uint32',
          length: GPU_LOCATION_ALLOCATION_STATUS.length
        }
      },
      createContributor: getView =>
        new GPULocationAllocation({
          id: props.id,
          facilityCount: props.facilityCount,
          maximumFacilityCount: props.maximumFacilityCount,
          maximumSwapIterations: props.maximumSwapIterations,
          operation: props.operation,
          costs: props.costs === undefined ? undefined : getView<'float32'>('costs'),
          sparseCosts:
            props.sparseCosts === undefined
              ? undefined
              : {
                  offsets: getView<'uint32'>('sparseOffsets'),
                  facilities: getView<'uint32'>('sparseFacilities'),
                  costs: getView<'float32'>('sparseCosts')
                },
          demandWeights:
            props.demandWeights === undefined ? undefined : getView<'float32'>('demandWeights'),
          coverageDistance:
            props.coverageDistance === undefined
              ? undefined
              : getView<'float32'>('coverageDistance'),
          output: {
            selectedFacilities: getView<'uint32'>('selectedFacilities'),
            assignments: getView<'uint32'>('assignments'),
            assignedCosts:
              props.output.assignedCosts === undefined
                ? undefined
                : getView<'float32'>('assignedCosts'),
            objective: getView<'float32'>('objective'),
            status: getView<'uint32'>('status')
          }
        })
    });
  }
}
