// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import type {GPUNetworkFacilityOptimizationProps} from './gpu-network-facility-optimization';

/** Capture compile-time controls and ports without copying graph storage. @internal */
export function snapshotFacilityConstructionProps(
  props: GPUNetworkFacilityOptimizationProps
): GPUNetworkFacilityOptimizationProps {
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
    'threshold',
    'maximumCandidates',
    'output'
  ]);
  const output = snapshotConstructionFields(snapshot.output, [
    'selectedFacilities',
    'assignments',
    'objective',
    'summary',
    'status'
  ]);
  return Object.freeze({
    ...snapshot,
    output: Object.freeze({
      ...output,
      // Preserve unsupported own fields so construction still rejects them.
      ...(output.status
        ? {
            status: snapshotConstructionFields(output.status, [
              ...new Set([
                ...Object.keys(output.status),
                'complete',
                'invalidCount',
                'iterationCount',
                'converged',
                'iterationLimitReached'
              ])
            ] as (keyof typeof output.status)[])
          }
        : {})
    })
  });
}
