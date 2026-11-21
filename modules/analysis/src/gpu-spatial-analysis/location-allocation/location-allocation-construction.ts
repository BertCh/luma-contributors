// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPULocationAllocationProps} from './gpu-location-allocation';
import type {GPUBoundedFacilityAllocationProps} from './gpu-bounded-facility-allocation';
import type {LocationAllocationOperationProps} from './gpu-location-allocation-operation';

/** @internal Snapshot metadata while retaining graph view identity and per-frame buffer contents. */
export function snapshotLocationAllocationProps(
  props: GPULocationAllocationProps
): GPULocationAllocationProps {
  const snapshot = snapshotConstructionFields(props, [
    'id',
    'costs',
    'sparseCosts',
    'facilityCount',
    'demandWeights',
    'operation',
    'maximumFacilityCount',
    'maximumSwapIterations',
    'coverageDistance',
    'costsComplete',
    'costLimit',
    'output'
  ]);
  return Object.freeze({
    ...snapshot,
    ...(snapshot.sparseCosts
      ? {
          sparseCosts: snapshotConstructionFields(snapshot.sparseCosts, [
            'offsets',
            'facilities',
            'costs'
          ])
        }
      : {}),
    output: snapshotConstructionFields(snapshot.output, [
      'selectedFacilities',
      'assignments',
      'assignedCosts',
      'objective',
      'status'
    ])
  });
}

/** @internal The operation factory and CPU closure must capture the same construction snapshot. */
export function snapshotLocationAllocationOperationProps(
  props: LocationAllocationOperationProps
): LocationAllocationOperationProps {
  const snapshot = snapshotConstructionFields(props, [
    'id',
    'demandCount',
    'facilityCount',
    'costs',
    'sparseCosts',
    'demandWeights',
    'operation',
    'maximumFacilityCount',
    'maximumSwapIterations',
    'coverageDistance',
    'costsComplete',
    'costLimit',
    'output'
  ]);
  return Object.freeze({
    ...snapshot,
    ...(snapshot.sparseCosts
      ? {
          sparseCosts: snapshotConstructionFields(snapshot.sparseCosts, [
            'offsets',
            'facilities',
            'costs'
          ])
        }
      : {}),
    output: snapshotConstructionFields(snapshot.output, [
      'selectedFacilities',
      'assignments',
      'assignedCosts',
      'objective',
      'status'
    ])
  });
}

/** @internal Capture bounded solver dimensions, objective and output bindings. */
export function snapshotBoundedFacilityProps(
  props: GPUBoundedFacilityAllocationProps
): GPUBoundedFacilityAllocationProps {
  const snapshot = snapshotConstructionFields(props, [
    'id',
    'operation',
    'facilityCount',
    'selectCount',
    'costs',
    'costsComplete',
    'costLimit',
    'demand',
    'capacities',
    'output'
  ]);
  validateRequirement(
    snapshot.id ?? 'bounded-facility-allocation',
    !('complete' in snapshot.output),
    'output.complete was removed; use output.ports.status.complete'
  );
  const output = snapshotConstructionFields(snapshot.output, [
    'selectedFacilities',
    'assignments',
    'objective',
    'status',
    'ports'
  ]);
  const ports = output.ports && snapshotConstructionFields(output.ports, ['status']);
  return Object.freeze({
    ...snapshot,
    output: Object.freeze({
      ...output,
      ...(ports && {
        ports: Object.freeze({status: snapshotConstructionFields(ports.status, ['complete'])})
      })
    })
  });
}
