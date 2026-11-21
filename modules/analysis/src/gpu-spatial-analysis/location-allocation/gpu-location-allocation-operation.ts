// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {snapshotLocationAllocationOperationProps} from './location-allocation-construction';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPULocationAllocation,
  GPU_LOCATION_ALLOCATION_STATUS_LAYOUT,
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
  /** uint32 graph value: the matrix completeness word, see `GPULocationAllocationProps.costsComplete`. */
  costsComplete?: string;
  /** float32 graph value: the matrix cost limit, see `GPULocationAllocationProps.costLimit`. */
  costLimit?: string;
  output: {
    selectedFacilities: string;
    assignments: string;
    assignedCosts?: string;
    objective: string;
    status: string;
  };
};

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 * Lowering requires costs.length = demandCount * facilityCount, or sparse offsets.length =
 * demandCount + 1. Direct-contributor padded outputs do not change these declared input shapes.
 */
export class GPULocationAllocationGraphOperation extends GPUContributorOperation {
  constructor(props: LocationAllocationOperationProps) {
    props = snapshotLocationAllocationOperationProps(props);
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
        ...(props.coverageDistance === undefined ? {} : {coverageDistance: props.coverageDistance}),
        ...(props.costsComplete === undefined ? {} : {costsComplete: props.costsComplete}),
        ...(props.costLimit === undefined ? {} : {costLimit: props.costLimit})
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
          length: GPU_LOCATION_ALLOCATION_STATUS_LAYOUT.length
        }
      },
      createContributor: getView => {
        const id = props.id ?? 'location-allocation';
        const costs = props.costs === undefined ? undefined : getView<'float32'>('costs');
        const sparseCosts =
          props.sparseCosts === undefined
            ? undefined
            : {
                offsets: getView<'uint32'>('sparseOffsets'),
                facilities: getView<'uint32'>('sparseFacilities'),
                costs: getView<'float32'>('sparseCosts')
              };
        if (costs)
          validateRequirement(
            id,
            costs.length === props.demandCount * props.facilityCount,
            'costs length must equal demandCount * facilityCount'
          );
        if (sparseCosts)
          validateRequirement(
            id,
            sparseCosts.offsets.length === props.demandCount + 1,
            'sparseCosts.offsets length must equal demandCount + 1'
          );
        return new GPULocationAllocation({
          id: props.id,
          facilityCount: props.facilityCount,
          maximumFacilityCount: props.maximumFacilityCount,
          maximumSwapIterations: props.maximumSwapIterations,
          operation: props.operation,
          costs,
          sparseCosts,
          demandWeights:
            props.demandWeights === undefined ? undefined : getView<'float32'>('demandWeights'),
          coverageDistance:
            props.coverageDistance === undefined
              ? undefined
              : getView<'float32'>('coverageDistance'),
          costsComplete:
            props.costsComplete === undefined ? undefined : getView<'uint32'>('costsComplete'),
          costLimit: props.costLimit === undefined ? undefined : getView<'float32'>('costLimit'),
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
        });
      }
    });
  }
}
