// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {addFleetDwellZoneEventsRecipe} from '../../../src/gpu-spatial-analysis/recipes/fleet-dwell-recipe';
import {RecipeTestFixture} from './recipe-harness';

it.each([
  'scratch',
  'track-list',
  'final-list'
] as const)('TRJ-105 distinguishes %s loss from exact dwell', async mode => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, `fleet-loss-${mode}`);
  try {
    const scratch = fixture.output('scratch', 'uint32', 1);
    const track = fixture.output('track', 'uint32', 1);
    const final = fixture.output('final', 'uint32', 1);
    const sums = fixture.output('sums', 'float32', 1);
    const visitOverflow = fixture.output('visit-overflow', 'uint32', 1);
    const visitDwell = fixture.output('visit-dwell', 'float32', 2);
    const result = addFleetDwellZoneEventsRecipe(fixture.graph, {
      trajectories: {
        kind: 'trajectories',
        positions: fixture.input('positions', Float32Array.of(-2, 0, 2, 0), 'float32x2', 2),
        timestamps: fixture.input('times', Float32Array.of(0, 4), 'float32', 2),
        trackOffsets: fixture.input('offsets', Uint32Array.of(0, 2), 'uint32', 2)
      },
      zones: {
        kind: 'polygons',
        positions: fixture.input(
          'zones',
          Float32Array.of(-1, -1, 1, -1, 1, 1, -1, 1),
          'float32x2',
          4
        ),
        featureOffsets: fixture.input('features', Uint32Array.of(0, 1), 'uint32', 2),
        polygonOffsets: fixture.input('polygons', Uint32Array.of(0, 1), 'uint32', 2),
        ringOffsets: fixture.input('rings', Uint32Array.of(0, 4), 'uint32', 2)
      },
      candidateCapacity: mode === 'scratch' ? 1 : 32,
      trackEventCapacity: mode === 'track-list' ? 1 : 8,
      eventCapacity: mode === 'final-list' ? 1 : 8,
      scratch: {
        visitTable: {
          ids: fixture.output('visit-ids', 'uint32', 2).view,
          dwellTimes: visitDwell.view,
          status: {
            count: fixture.output('visit-count', 'uint32', 1).view,
            overflow: visitOverflow.view
          }
        }
      },
      output: {
        status: {
          candidateOverflow: scratch.view,
          trackOverflow: track.view,
          eventOverflow: final.view
        },
        table: {sumValues: sums.view}
      }
    });
    const statistics = result.status.stages.find(stage => stage.stage === 'zone-statistics');
    expect(statistics?.status.candidateOverflow === scratch.view).toBe(true);
    expect(
      result.status.stages.find(stage => stage.stage === 'visit-table')?.status
        .candidateOverflow === scratch.view
    ).toBe(true);
    fixture.run();
    expect(await fixture.readUint32(scratch, 1)).toEqual([Number(mode === 'scratch')]);
    expect(await fixture.readUint32(track, 1)).toEqual([Number(mode === 'track-list')]);
    expect(await fixture.readUint32(final, 1)).toEqual([Number(mode === 'final-list')]);
    expect(await fixture.readUint32(visitOverflow, 1)).toEqual([0]);
    if (mode !== 'scratch') {
      // Linear motion enters at t=1 and exits at t=3: exactly two units of dwell,
      // even if either event-list capacity retains only one event.
      expect(await fixture.readFloat32(sums, 1)).toEqual([2]);
      expect((await fixture.readFloat32(visitDwell, 2))[0]).toBe(2);
    }
  } finally {
    fixture.destroy();
  }
});
