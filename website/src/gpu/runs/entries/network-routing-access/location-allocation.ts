// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULocationAllocation on network costs: choose which candidate facilities to open and assign every
// demand location to its facility. Demand dots take the color of their assigned facility.

import {GPULocationAllocation} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {RunGlyph} from '../../interaction';
import {defineOperationRun} from '../../define-run';
import {addCostMatrix, createAccessScene, uploadSearch} from './shared/access';
import {addFlaggedIds} from './shared/display';
import {findParameter, snapToParameter} from '../../interaction';
import {findReachPoint, getRoadContext, uploadRoads} from './shared/roads';
import {getFacilityMarkers} from './shared/access';
import {parameters} from './location-allocation.parameters';

const COVERAGE = findParameter(parameters, 'coverageDistance');

export default defineOperationRun({
  parameters,
  caption: 'Selected facilities (bright) and the demand locations assigned to each.',
  createInputs: ({random, world}) => createAccessScene(random, world, 8, 120),
  build(builder, scene, values) {
    const {network} = scene;
    const candidateCount = scene.facilityNodes.length;
    const search = uploadSearch(builder, network);
    const matrix = addCostMatrix(
      builder,
      search,
      scene.demandNodes,
      network.nodeCount,
      scene.facilityNodes
    );
    const coverage = builder.allocate('coverage-distance', 'float32', 1);
    const selected = builder.allocate('selected', 'uint32', candidateCount);
    const assignments = builder.allocate('assignments', 'uint32', scene.demandNodes.length);
    const objective = builder.allocate('objective', 'float32', 1);
    const status = builder.allocate('status', 'uint32', 7);
    const isCoverage = values.operation === 1;
    builder.graph.add(
      new GPULocationAllocation({
        operation: isCoverage ? 'maximum-coverage' : 'p-median',
        costs: matrix.view,
        facilityCount: candidateCount,
        maximumFacilityCount: Math.round(values.maximumFacilityCount),
        demandWeights: builder.upload('demand-weights', 'float32', scene.population).view,
        coverageDistance: isCoverage ? coverage.view : undefined,
        output: {
          selectedFacilities: selected.view,
          assignments: assignments.view,
          objective: objective.view,
          status: status.view
        }
      })
    );
    return {
      outputs: {
        ...uploadRoads(builder, network),
        assignments,
        selected,
        objective,
        selectedIds: addFlaggedIds(builder, {
          id: 'selected',
          flags: selected,
          count: candidateCount
        }),
        demands: builder.upload('demands', 'float32x2', scene.demandPositions),
        candidates: builder.upload('candidates', 'float32x2', scene.facilityPositions)
      },
      update: current => {
        search.costLimit.write(Float32Array.of(4));
        coverage.write(Float32Array.of(current.coverageDistance));
      }
    };
  },
  rebuildOn: ['operation', 'maximumFacilityCount'],
  // Reach: the distance from the nearest candidate site to the pointer is the coverage distance.
  interaction: {
    kind: 'reach',
    hint: 'reach for coverage',
    rest: ({inputs, world, toUnit}) =>
      toUnit(findReachPoint(inputs.facilityPositions, COVERAGE.default, world)),
    resolve(pointer, {inputs, outputs, values, peek}) {
      let nearest = 0;
      let nearestDistance = Infinity;
      for (let site = 0; site < inputs.facilityNodes.length; site++) {
        const distance = Math.hypot(
          inputs.facilityPositions[site * 2] - pointer.x,
          inputs.facilityPositions[site * 2 + 1] - pointer.y
        );
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearest = site;
        }
      }
      const coverageDistance = snapToParameter(COVERAGE, nearestDistance);
      const selected = peek(outputs.selected);
      const objective = peek(outputs.objective);
      const total = inputs.population.reduce((sum, value) => sum + value, 0);
      const glyphs: RunGlyph[] = [
        {
          shape: 'line',
          from: [inputs.facilityPositions[nearest * 2], inputs.facilityPositions[nearest * 2 + 1]],
          to: [pointer.x, pointer.y],
          dashed: true
        }
      ];
      if (selected) {
        for (let site = 0; site < selected.length; site++) {
          if (selected[site] !== 1) continue;
          glyphs.push({
            shape: 'ring',
            at: [inputs.facilityPositions[site * 2], inputs.facilityPositions[site * 2 + 1]],
            radius: coverageDistance,
            dashed: true
          });
        }
      }
      const covered =
        objective && values.operation === 1
          ? ` · covered ${Math.round((100 * objective[0]) / total)}%`
          : '';
      return {
        values: {coverageDistance},
        glyphs,
        readout: `coverageDistance ${coverageDistance.toFixed(2)}${covered}`
      };
    }
  },
  draw: ({outputs, theme}) => [
    getRoadContext(outputs),
    {layer: 'dots', positions: outputs.demands, values: outputs.assignments, radius: 6.5},
    ...getFacilityMarkers(outputs.candidates, theme, {ids: outputs.selectedIds, radius: 11})
  ]
});
