// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkRationalAccess: demand shifts between facilities until congestion equalizes. Facility
// dots show the final score; demand dots show the demand left unserved.

import {GPUNetworkRationalAccess} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {addCostMatrix, createAccessScene, uploadSearch} from './shared/access';
import {getRoadContext, uploadRoads} from './shared/roads';
import {parameters} from './network-rational-access.parameters';

const ORIGIN_COUNT = 48;
const FACILITY_COUNT = 8;

export default defineOperationRun({
  parameters,
  caption: 'Facility scores after rational reallocation; demand dots show unserved demand.',
  createInputs: ({random, world}) => createAccessScene(random, world, FACILITY_COUNT, ORIGIN_COUNT),
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
    const parameterBuffer = builder.allocate('parameters', 'float32', 6);
    const assignments = builder.allocate('assignments', 'float32', ORIGIN_COUNT * FACILITY_COUNT);
    const scores = builder.allocate('scores', 'float32', ORIGIN_COUNT);
    const unserved = builder.allocate('unserved', 'float32', ORIGIN_COUNT);
    const status = builder.allocate('status', 'uint32', 4);
    builder.graph.add(
      new GPUNetworkRationalAccess({
        costs: matrix.view,
        demand: builder.upload(
          'demand',
          'float32',
          Float32Array.from(scene.population, value => Math.round(value))
        ).view,
        supply: builder.upload(
          'supply',
          'float32',
          Float32Array.from(scene.supply, value => 40 * value)
        ).view,
        parameters: parameterBuffer.view,
        maximumCycles: Math.round(values.maximumCycles),
        demandMode: 'fractional',
        output: {
          assignments: assignments.view,
          scores: scores.view,
          unservedDemand: unserved.view,
          status: status.view
        }
      })
    );
    return {
      outputs: {
        ...uploadRoads(builder, network),
        scores,
        unserved,
        demands: builder.upload('demands', 'float32x2', scene.demandPositions),
        facilities: builder.upload('facilities', 'float32x2', scene.facilityPositions)
      },
      update: current => {
        search.costLimit.write(Float32Array.of(4));
        // tau, rho, initial step, minimum step, half-life, gap tolerance.
        parameterBuffer.write(Float32Array.of(current.tau, current.rho, 0.5, 0.01, 8, 0.0001));
      }
    };
  },
  rebuildOn: ['maximumCycles'],
  draw: ({outputs}) => [
    getRoadContext(outputs),
    {layer: 'dots', positions: outputs.demands, values: outputs.scores, radius: 9},
    {layer: 'dots', positions: outputs.facilities, radius: 11, color: [1, 1, 1, 1]}
  ]
});
