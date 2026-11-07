// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addFacilityScenarioRecipe: try a facility scenario against a reusable cost matrix. Opening or
// closing sites only rewrites a mask, so the picture updates without recompiling.

import {getGPUNetworkAccessibilityParameterValues} from '@luma-contributors/analysis/gpu-network';
import {addFacilityScenarioRecipe} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {addCostMatrix, createAccessScene, uploadSearch} from './shared/access';
import {getRoadContext, uploadRoads} from './shared/roads';
import {parameters} from './facility-scenario.parameters';

const FACILITY_COUNT = 8;
const DEMAND_COUNT = 240;

export default defineOperationRun({
  parameters,
  caption: 'Demand colored by its nearest open facility; closed sites are dim.',
  createInputs: ({random, world}) => createAccessScene(random, world, FACILITY_COUNT, DEMAND_COUNT),
  build(builder, scene) {
    const {network} = scene;
    const search = uploadSearch(builder, network);
    // Facility-major costs to the demand locations.
    const matrix = addCostMatrix(
      builder,
      search,
      scene.facilityNodes,
      network.nodeCount,
      scene.demandNodes
    );
    const mask = builder.allocate('facility-mask', 'uint32', FACILITY_COUNT);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const accessibility = builder.allocate('accessibility', 'float32', DEMAND_COUNT);
    const nearestCosts = builder.allocate('nearest-costs', 'float32', DEMAND_COUNT);
    const nearestFacilities = builder.allocate('nearest-facilities', 'uint32', DEMAND_COUNT);
    const covered = builder.allocate('covered', 'float32', 1);
    const uncovered = builder.allocate('uncovered', 'float32', 1);
    const invalid = builder.allocate('invalid', 'uint32', 1);
    addFacilityScenarioRecipe(builder.graph, {
      costs: matrix.view,
      demand: builder.upload('demand', 'float32', scene.population).view,
      supply: builder.upload('supply', 'float32', scene.supply).view,
      facilityMask: mask.view,
      parameters: parameterBuffer.view,
      outputs: {
        accessibility: accessibility.view,
        nearestCosts: nearestCosts.view,
        nearestFacilities: nearestFacilities.view,
        coveredDemand: covered.view,
        uncoveredDemand: uncovered.view,
        invalidDemandCount: invalid.view
      }
    });
    return {
      outputs: {
        ...uploadRoads(builder, network),
        mask,
        accessibility,
        nearestFacilities,
        demands: builder.upload('demands', 'float32x2', scene.demandPositions),
        facilities: builder.upload('facilities', 'float32x2', scene.facilityPositions)
      },
      update: values => {
        search.costLimit.write(Float32Array.of(4));
        mask.write(
          Uint32Array.from({length: FACILITY_COUNT}, (_, index) =>
            index < values.openFacilities ? 1 : 0
          )
        );
        parameterBuffer.write(
          getGPUNetworkAccessibilityParameterValues({
            threshold: values.threshold,
            decay: 'exponential',
            beta: 3
          })
        );
      }
    };
  },
  draw: ({outputs, values}) => [
    getRoadContext(outputs),
    values.measure === 0
      ? {layer: 'dots', positions: outputs.demands, values: outputs.nearestFacilities, radius: 5}
      : {layer: 'dots', positions: outputs.demands, values: outputs.accessibility, radius: 5},
    {
      layer: 'dots',
      positions: outputs.facilities,
      values: outputs.mask,
      ramp: 'sequential',
      range: [-1, 1],
      radius: 11
    }
  ]
});
