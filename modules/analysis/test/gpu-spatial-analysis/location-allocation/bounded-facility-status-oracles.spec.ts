// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUBoundedFacilityAllocation} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-bounded-facility-allocation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createOracleFixture} from '../global-coordinates/output-oracle-fixture';

it.each([
  'p-center',
  'capacitated-p-median',
  'unassigned',
  'invalid demand'
] as const)('bounded facility allocation overwrites every status word: %s', async scene => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOracleFixture(device);
  const status = fixture.output('uint32', 3);
  // One demand row, two candidate facilities, choose one. Both costs are finite.
  // Demand 1 fits capacity 1; demand 2 cannot fit either selected facility.
  // Exhaustive p-center certifies its optimum; demand-order greedy capacity
  // assignment deliberately carries no global-optimum certificate.
  fixture.graph.add(
    new GPUBoundedFacilityAllocation({
      operation:
        scene === 'p-center' || scene === 'invalid demand' ? 'p-center' : 'capacitated-p-median',
      facilityCount: 2,
      selectCount: 1,
      costs: fixture.input('float32', [1, 2]),
      demand: fixture.input('float32', [
        scene === 'invalid demand' ? -1 : scene === 'unassigned' ? 2 : 1
      ]),
      capacities: fixture.input('float32', [1, 1]),
      output: {
        selectedFacilities: fixture.output('uint32', 2).view,
        assignments: fixture.output('uint32', 1).view,
        objective: fixture.output('float32', 1).view,
        status: status.view
      }
    })
  );
  const compiled = fixture.graph.compile();
  try {
    for (let submission = 0; submission < 2; submission++) {
      status.poison();
      submitGraph(device, compiled, undefined);
      expect(await status.read()).toEqual(
        scene === 'invalid demand'
          ? [1, 0, 0]
          : [Number(scene === 'unassigned'), 1, Number(scene === 'p-center')]
      );
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
