// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addFacilityScenarioRecipe: try a facility scenario against a reusable cost matrix. Opening or
// closing sites only rewrites a mask, so the picture updates without recompiling.

import {getGPUNetworkAccessibilityParameterValues} from '@luma-contributors/analysis/gpu-network';
import {addFacilityScenarioRecipe} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {addCostMatrix, createAccessScene, uploadSearch} from './shared/access';
import type {RunGlyph} from '../../interaction';
import {getFacilityMarkers} from './shared/access';
import {addFlaggedIds, addThresholdedIds} from './shared/display';
import {findEmptyPoint, getRoadContext, uploadRoads} from './shared/roads';
import {parameters} from './facility-scenario.parameters';

const FACILITY_COUNT = 8;
const DEMAND_COUNT = 240;

export default defineOperationRun({
  parameters,
  caption:
    'Demand colored by its nearest open facility; closed sites are dim; demand beyond the threshold of its site is muted.',
  createInputs: ({random, world}) => ({
    ...createAccessScene(random, world, FACILITY_COUNT, DEMAND_COUNT),
    closed: -1
  }),
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
    const thresholdBuffer = builder.allocate('threshold', 'float32', 1);
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
        covered,
        uncovered,
        facilityIds: addFlaggedIds(builder, {id: 'open', flags: mask, count: FACILITY_COUNT}),
        accessibility,
        nearestFacilities,
        coveredFacilities: addThresholdedIds(builder, {
          id: 'covered-facilities',
          facilities: nearestFacilities,
          costs: nearestCosts,
          threshold: thresholdBuffer
        }),
        demands: builder.upload('demands', 'float32x2', scene.demandPositions),
        facilities: builder.upload('facilities', 'float32x2', scene.facilityPositions)
      },
      update: values => {
        search.costLimit.write(Float32Array.of(4));
        thresholdBuffer.write(Float32Array.of(values.threshold));
        mask.write(
          Uint32Array.from({length: FACILITY_COUNT}, (_, index) =>
            index < values.openFacilities && index !== scene.closed ? 1 : 0
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
  // Probe: the hovered facility closes; demand that was served by it moves to another site.
  interaction: {
    kind: 'probe',
    hint: 'hover to close a site',
    rest: ({inputs, world, toUnit}) => toUnit(findEmptyPoint(inputs.facilityPositions, world)),
    resolve(pointer, {inputs, outputs, values, peek}) {
      let closed = -1;
      let nearestDistance = 0.09;
      for (let site = 0; site < Math.round(values.openFacilities); site++) {
        const distance = Math.hypot(
          inputs.facilityPositions[site * 2] - pointer.x,
          inputs.facilityPositions[site * 2 + 1] - pointer.y
        );
        if (distance < nearestDistance) {
          nearestDistance = distance;
          closed = site;
        }
      }
      const changed = closed !== inputs.closed;
      inputs.closed = closed;
      const covered = peek(outputs.covered);
      const uncovered = peek(outputs.uncovered);
      const glyphs: RunGlyph[] = [];
      if (closed >= 0) {
        const x = inputs.facilityPositions[closed * 2];
        const y = inputs.facilityPositions[closed * 2 + 1];
        glyphs.push(
          {shape: 'line', from: [x - 0.03, y - 0.03], to: [x + 0.03, y + 0.03]},
          {shape: 'line', from: [x - 0.03, y + 0.03], to: [x + 0.03, y - 0.03]}
        );
      }
      const total = covered && uncovered ? covered[0] + uncovered[0] : 0;
      return {
        spotlight: null,
        rerun: changed,
        glyphs,
        readout:
          closed >= 0
            ? `close site ${closed + 1} · covered ${covered ? Math.round(covered[0]) : '-'} of ${Math.round(total)}`
            : `open ${Math.round(values.openFacilities)} sites · covered ${covered ? Math.round(covered[0]) : '-'}`
      };
    }
  },
  draw: ({outputs, values, theme}) => [
    getRoadContext(outputs),
    values.measure === 0
      ? // Demand beyond `threshold` of its nearest open site is uncovered and draws muted.
        {layer: 'dots', positions: outputs.demands, values: outputs.coveredFacilities, radius: 5}
      : {
          layer: 'dots',
          positions: outputs.demands,
          values: outputs.accessibility,
          range: [0, 6],
          radius: 5
        },
    ...getFacilityMarkers(outputs.facilities, theme, {ids: outputs.facilityIds, radius: 11})
  ]
});
