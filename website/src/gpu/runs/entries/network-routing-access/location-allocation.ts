// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULocationAllocation on network costs: choose which candidate facilities to open and assign every
// demand location to its facility. Demand dots take the color of their assigned facility.

import {GPULocationAllocation} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addCostMatrix, createAccessScene, uploadSearch} from './shared/access';
import {getRoadContext, uploadRoads} from './shared/roads';
import {parameters} from './location-allocation.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Selected facilities (bright) and the demand locations assigned to each.',
  createInputs: ({random, world}) => createAccessScene(random, world, 12, 120),
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
  draw: ({outputs}) => [
    getRoadContext(outputs),
    {layer: 'dots', positions: outputs.demands, values: outputs.assignments, radius: 4.5},
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
