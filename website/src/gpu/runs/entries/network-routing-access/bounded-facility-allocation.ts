// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUBoundedFacilityAllocation: exhaustive p-center and capacitated p-median over a small candidate
// set. Demand dots take the color of the facility that serves them; capacities bound how many can.

import {GPUBoundedFacilityAllocation} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addCostMatrix, createAccessScene, uploadSearch} from './shared/access';
import {getRoadContext, uploadRoads} from './shared/roads';
import {parameters} from './bounded-facility-allocation.parameters';

const DEMAND_COUNT = 48;
const CANDIDATE_COUNT = 10;

export default defineOperationRun({
  parameters,
  caption: 'The chosen facilities (bright) and the demand each serves within its capacity.',
  createInputs: ({random, world}) =>
    createAccessScene(random, world, CANDIDATE_COUNT, DEMAND_COUNT),
  build(builder, scene, values) {
    const {network} = scene;
    const search = uploadSearch(builder, network);
    const matrix = addCostMatrix(
      builder,
      search,
      scene.demandNodes,
      network.nodeCount,
      scene.facilityNodes
    );
    const capacities = builder.allocate('capacities', 'float32', CANDIDATE_COUNT);
    const selected = builder.allocate('selected', 'uint32', CANDIDATE_COUNT);
    const assignments = builder.allocate('assignments', 'uint32', DEMAND_COUNT);
    const objective = builder.allocate('objective', 'float32', 1);
    const status = builder.allocate('status', 'uint32', 3);
    const selectCount = Math.round(values.selectCount);
    builder.graph.add(
      new GPUBoundedFacilityAllocation({
        operation: values.operation === 0 ? 'p-center' : 'capacitated-p-median',
        facilityCount: CANDIDATE_COUNT,
        selectCount,
        costs: matrix.view,
        demand: builder.upload(
          'demand',
          'float32',
          Float32Array.from(scene.population, value => Math.round(value))
        ).view,
        capacities: capacities.view,
        output: {
          selectedFacilities: selected.view,
          assignments: assignments.view,
          objective: objective.view,
          status: status.view
        }
      })
    );
    const totalDemand = scene.population.reduce((sum, value) => sum + Math.round(value), 0);
    return {
      outputs: {
        ...uploadRoads(builder, network),
        assignments,
        selected,
        demands: builder.upload('demands', 'float32x2', scene.demandPositions),
        candidates: builder.upload('candidates', 'float32x2', scene.facilityPositions)
      },
      update: current => {
        search.costLimit.write(Float32Array.of(4));
        // Capacity per facility as a share of the average load for the selected count.
        capacities.write(
          new Float32Array(CANDIDATE_COUNT).fill((totalDemand / selectCount) * current.capacity)
        );
      }
    };
  },
  rebuildOn: ['operation', 'selectCount'],
  draw: ({outputs}) => [
    getRoadContext(outputs),
    {layer: 'dots', positions: outputs.demands, values: outputs.assignments, radius: 5},
    {
      layer: 'dots',
      positions: outputs.candidates,
      values: outputs.selected,
      ramp: 'sequential',
      range: [-1, 1],
      radius: 15
    }
  ]
});
