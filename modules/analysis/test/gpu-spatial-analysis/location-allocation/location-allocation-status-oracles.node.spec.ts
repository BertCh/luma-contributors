// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeLocationAllocationOnCPU} from '../../../src/gpu-spatial-analysis/location-allocation/cpu-location-allocation-operation';
import {ALLOCATION_STATUS_CASES} from './location-allocation-status-oracles-fixture';

for (const sparse of [false, true])
  it.each(
    ALLOCATION_STATUS_CASES
  )(`CPU allocation full status follows enumerated objectives (sparse=${sparse}): $maximumSwapIterations`, scene => {
    const costs = Float32Array.from(scene.costs);
    const result = computeLocationAllocationOnCPU(
      sparse
        ? {
            sparseCosts: {
              costs,
              offsets: Uint32Array.of(0, 4, 8, 12),
              facilities: Uint32Array.from([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
            }
          }
        : {costs},
      {
        operation: 'p-median',
        facilityCount: 4,
        demandCount: 3,
        maximumFacilityCount: 2,
        maximumSwapIterations: scene.maximumSwapIterations
      }
    );
    expect(Array.from(result.status)).toEqual(scene.status);
    expect(Array.from(result.selectedFacilities)).toEqual(scene.selected);
    expect(Array.from(result.objective)).toEqual([scene.objective]);
  });
