// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPULocationAllocationGraphOperation} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-location-allocation-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {addOracleOperation, createOracleFixture} from '../global-coordinates/output-oracle-fixture';
import {ALLOCATION_STATUS_CASES} from './location-allocation-status-oracles-fixture';

for (const sparse of [false, true])
  it.each(
    ALLOCATION_STATUS_CASES
  )(`allocation operation/contributor full status overwrites poison (sparse=${sparse}): $maximumSwapIterations`, async scene => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOracleFixture(device);
    const operation = new GPULocationAllocationGraphOperation({
      operation: 'p-median',
      facilityCount: 4,
      demandCount: 3,
      maximumFacilityCount: 2,
      maximumSwapIterations: scene.maximumSwapIterations,
      ...(sparse
        ? {sparseCosts: {offsets: 'offsets', facilities: 'facilities', costs: 'costs'}}
        : {costs: 'costs'}),
      output: {
        selectedFacilities: 'selected',
        assignments: 'assignments',
        objective: 'objective',
        status: 'status'
      }
    });
    const costs = fixture.input('float32', scene.costs);
    const outputs = addOracleOperation(
      fixture,
      operation,
      sparse
        ? {
            sparseCosts: costs,
            sparseOffsets: fixture.input('uint32', [0, 4, 8, 12]),
            sparseFacilities: fixture.input('uint32', [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
          }
        : {costs}
    );
    const compiled = fixture.graph.compile();
    try {
      for (let submission = 0; submission < 2; submission++) {
        for (const output of Object.values(outputs)) output.poison();
        submitGraph(device, compiled, undefined);
        expect(await outputs['status'].read()).toEqual(scene.status);
        expect(await outputs['selectedFacilities'].read()).toEqual(scene.selected);
        expect(await outputs['objective'].read()).toEqual([scene.objective]);
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
