// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkFacilityOptimization (backup coverage): pick `selectCount` of the candidate sites so the
// most demand has a backup within `threshold`. Demand dots take the color of their assigned site.

import {GPUNetworkFacilityOptimization} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {addCostMatrix, createAccessScene, uploadSearch} from './shared/access';
import {getRoadContext, uploadRoads} from './shared/roads';
import {parameters} from './network-facility-optimization.parameters';

const DEMAND_COUNT = 56;
const CANDIDATE_COUNT = 12;

export default defineOperationRun({
  parameters,
  caption: 'Candidate sites chosen for backup coverage (bright) among the demand locations.',
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
    const threshold = builder.allocate('threshold', 'float32', 1);
    const selected = builder.allocate('selected', 'uint32', CANDIDATE_COUNT);
    const assignments = builder.allocate('assignments', 'uint32', DEMAND_COUNT);
    const objective = builder.allocate('objective', 'float32', 1);
    const status = builder.allocate('status', 'uint32', 4);
    builder.graph.add(
      new GPUNetworkFacilityOptimization({
        operation: 'backup-coverage',
        facilityCount: CANDIDATE_COUNT,
        selectCount: Math.round(values.selectCount),
        costs: matrix.view,
        demand: builder.upload('demand', 'float32', scene.population).view,
        capacities: builder.upload(
          'capacities',
          'float32',
          new Float32Array(CANDIDATE_COUNT).fill(1000)
        ).view,
        threshold: threshold.view,
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
        demands: builder.upload('demands', 'float32x2', scene.demandPositions),
        candidates: builder.upload('candidates', 'float32x2', scene.facilityPositions)
      },
      update: current => {
        search.costLimit.write(Float32Array.of(4));
        threshold.write(Float32Array.of(current.threshold));
      }
    };
  },
  rebuildOn: ['selectCount'],
  draw: ({outputs}) => [
    getRoadContext(outputs),
    {layer: 'dots', context: true, positions: outputs.demands, radius: 5},
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
