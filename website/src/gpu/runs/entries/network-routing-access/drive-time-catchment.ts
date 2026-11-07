// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addDriveTimeCatchmentRecipe: snap facilities and demand points to the road network, search drive
// times from the facilities and band every demand point by travel time.

import {addDriveTimeCatchmentRecipe} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {uploadNetworkTopology} from '../../inputs';
import {createRoads, getRoadContext, gatherPositions, pickNodes, uploadRoads} from './shared/roads';
import {parameters} from './drive-time-catchment.parameters';

const DEMAND_COUNT = 400;
const BAND_COUNT = 4;

export default defineOperationRun({
  parameters,
  caption: 'Demand points banded by drive time to the nearest facility.',
  createInputs: ({random, world}) => {
    const network = createRoads(random, world, 14);
    const demand = Float32Array.from({length: DEMAND_COUNT * 2}, (_, index) =>
      index % 2 === 0 ? random() * world.width : random()
    );
    const facilityNodes = pickNodes(network, world, 6, 1);
    return {
      network,
      demand,
      facilityNodes,
      facilities: gatherPositions(network.nodes, facilityNodes)
    };
  },
  build(builder, {network, demand, facilities}, values) {
    const facilityCount = Math.round(values.facilityCount);
    const {topology} = uploadNetworkTopology(builder, network);
    const nodePositions = builder.upload('node-positions', 'float32x2', network.nodes);
    const breaks = builder.allocate('cost-breaks', 'float32', BAND_COUNT);
    const demandBands = builder.allocate('demand-bands', 'uint32', DEMAND_COUNT);
    addDriveTimeCatchmentRecipe(builder.graph, {
      topology: {...topology, nodePositions: nodePositions.view},
      facilities: builder.upload('facilities', 'float32x2', facilities.slice(0, facilityCount * 2))
        .view,
      demand: builder.upload('demand', 'float32x2', demand).view,
      costBreaks: breaks.view,
      bandCapacity: BAND_COUNT,
      maximumIterations: 64,
      outputs: {demandBands: demandBands.view}
    });
    return {
      outputs: {
        ...uploadRoads(builder, network),
        demandBands,
        demands: builder.upload('demands', 'float32x2', demand),
        facilities: builder.upload('facility-points', 'float32x2', facilities)
      },
      update: current =>
        breaks.write(
          Float32Array.from({length: BAND_COUNT}, (_, band) => (band + 1) * current.breakWidth)
        )
    };
  },
  rebuildOn: ['facilityCount'],
  draw: ({outputs, values}) => [
    getRoadContext(outputs),
    {
      layer: 'dots',
      positions: outputs.demands,
      values: outputs.demandBands,
      ramp: 'sequential',
      range: [-1, 4],
      radius: 5
    },
    {
      layer: 'dots',
      positions: outputs.facilities,
      count: Math.round(values.facilityCount),
      radius: 10,
      color: [1, 1, 1, 1]
    }
  ]
});
