// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {addFacilityScenarioRecipe} from '../../../src/gpu-network/recipes/network-scenarios-recipe';
import {getGPUNetworkAccessibilityParameterValues} from '../../../src/gpu-network/network-accessibility/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {RecipeTestFixture} from '../../gpu-spatial-analysis/recipes/recipe-harness';

it('parallel facility summaries match a float64 demand oracle across partial groups and live masks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'facility-summary');
  let compiled: ReturnType<typeof fixture.graph.compile> | undefined;
  try {
    const demandCount = 513;
    const demand = fixture.output('demand', 'float32', demandCount);
    const mask = fixture.output('mask', 'uint32', 2);
    const parameters = fixture.output('parameters', 'float32', 4);
    const costs = Float32Array.from({length: 2 * demandCount}, (_, position) => {
      const row = position % demandCount;
      return row % 7 === 0 ? Infinity : position < demandCount ? row % 5 : 2;
    });
    const coveredDemand = fixture.output('covered', 'float32', 1);
    const uncoveredDemand = fixture.output('uncovered', 'float32', 1);
    const invalidDemandCount = fixture.output('invalid', 'uint32', 1);
    const nearestFacilities = fixture.output('nearest-facilities', 'uint32', demandCount);
    addFacilityScenarioRecipe(fixture.graph, {
      costs: fixture.input('costs', costs, 'float32', costs.length),
      demand: demand.view,
      supply: fixture.input('supply', new Float32Array([1, 1]), 'float32', 2),
      facilityMask: mask.view,
      parameters: parameters.view,
      outputs: {
        coveredDemand: coveredDemand.view,
        uncoveredDemand: uncoveredDemand.view,
        invalidDemandCount: invalidDemandCount.view,
        nearestFacilities: nearestFacilities.view
      }
    });
    compiled = fixture.graph.compile();
    for (const scenario of [
      {mask: [1, 1], threshold: 2, invalid: false},
      {mask: [0, 0], threshold: 2, invalid: false},
      {mask: [0, 1], threshold: 1, invalid: false},
      {mask: [1, 1], threshold: NaN, invalid: false},
      {mask: [1, 1], threshold: 2, invalid: true}
    ]) {
      const weights = Float32Array.from({length: demandCount}, (_, row) => {
        if (scenario.invalid) return NaN;
        return row % 13 === 0 ? NaN : row % 11 === 0 ? -1 : (row % 17) / 7;
      });
      demand.buffer.write(weights);
      mask.buffer.write(new Uint32Array(scenario.mask));
      const parameterValues = getGPUNetworkAccessibilityParameterValues({threshold: 2});
      parameterValues[0] = scenario.threshold;
      parameters.buffer.write(parameterValues);
      submitGraph(device, compiled, undefined);
      let covered = 0;
      let uncovered = 0;
      let invalid = 0;
      const facilities: number[] = [];
      weights.forEach((weight, row) => {
        let nearest = Infinity;
        let selected = 0xffffffff;
        scenario.mask.forEach((active, facility) => {
          const cost = costs[facility * demandCount + row];
          if (active && Number.isFinite(cost) && cost >= 0 && cost < nearest) {
            nearest = cost;
            selected = facility;
          }
        });
        facilities.push(selected);
        if (!Number.isFinite(weight) || weight < 0) invalid++;
        else if (Number.isFinite(nearest) && nearest <= scenario.threshold) covered += weight;
        else uncovered += weight;
      });
      expect(await fixture.readUint32(nearestFacilities, demandCount)).toEqual(facilities);
      expect(await fixture.readUint32(invalidDemandCount, 1)).toEqual([invalid]);
      const actualCovered = (await fixture.readFloat32(coveredDemand, 1))[0];
      const actualUncovered = (await fixture.readFloat32(uncoveredDemand, 1))[0];
      expect(Math.abs(actualCovered - covered)).toBeLessThanOrEqual(2e-6 * covered + 1e-6);
      expect(Math.abs(actualUncovered - uncovered)).toBeLessThanOrEqual(2e-6 * uncovered + 1e-6);
    }
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
});
