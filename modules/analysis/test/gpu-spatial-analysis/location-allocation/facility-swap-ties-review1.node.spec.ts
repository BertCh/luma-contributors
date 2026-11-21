// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeLocationAllocationOnCPU} from '../../../src/gpu-spatial-analysis/location-allocation/cpu-location-allocation-operation';
import {FACILITY_SWAP_TIE_SCENES} from './facility-swap-tie-fixtures';

it('contains a strictly improving tied swap, beyond a greedy tie (S1)', () => {
  const discriminates = FACILITY_SWAP_TIE_SCENES.some(scene => {
    const cost = (selected: number[]) =>
      Array.from({length: scene.demandCount}, (_, row) =>
        Math.min(...selected.map(facility => scene.costs[row * scene.facilityCount + facility]))
      ).reduce((sum, value) => sum + value, 0);
    const alternatives: number[] = [];
    for (const removed of scene.greedySelected)
      for (let added = 0; added < scene.facilityCount; added++) {
        if (scene.greedySelected.includes(added)) continue;
        alternatives.push(
          cost(scene.greedySelected.map(facility => (facility === removed ? added : facility)))
        );
      }
    const best = Math.min(...alternatives);
    return (
      best < cost(scene.greedySelected) && alternatives.filter(value => value === best).length > 1
    );
  });
  expect(discriminates).toBe(true);
});

for (const scene of FACILITY_SWAP_TIE_SCENES)
  it(`pins the literal CPU output for ${scene.name} (S1)`, () => {
    const result = computeLocationAllocationOnCPU(
      {costs: Float32Array.from(scene.costs)},
      {...scene, operation: 'p-median'}
    );
    expect(Array.from(result.selectedFacilities)).toEqual(scene.selected);
    expect(Array.from(result.assignments)).toEqual(scene.assignments);
    expect(Number.isFinite(result.objective[0])).toBe(true);
    expect(result.objective[0]).toBe(scene.objective);
    expect(Array.from(result.status)).toEqual(scene.status);
  });
