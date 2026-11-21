// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addDriveTimeCatchmentRecipe: snap facilities and demand points to the road network, search drive
// times from the facilities and band every demand point by travel time.

import {addDriveTimeCatchmentRecipe} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {uploadNetworkTopology} from '../../inputs';
import {createFacilityDrag, getFacilityMarkers, moveFacility} from './shared/access';
import {createRoads, getRoadContext, gatherPositions, pickNodes, uploadRoads} from './shared/roads';
import {parameters} from './drive-time-catchment.parameters';

const DEMAND_COUNT = 200;
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
    const facilities = gatherPositions(network.nodes, facilityNodes);
    const scene = {network, facilityNodes, facilityPositions: facilities};
    return {
      network,
      demand,
      facilityNodes,
      facilities,
      drag: createFacilityDrag(scene),
      scene,
      breakWidth: 0.25
    };
  },
  build(builder, {network, demand, facilities}, values) {
    const facilityCount = Math.round(values.facilityCount);
    const facilityBuffer = builder.upload(
      'facilities',
      'float32x2',
      facilities.slice(0, facilityCount * 2)
    );
    const facilityPoints = builder.upload('facility-points', 'float32x2', facilities);
    const {topology} = uploadNetworkTopology(builder, network);
    const nodePositions = builder.upload('node-positions', 'float32x2', network.nodes);
    const breaks = builder.allocate('cost-breaks', 'float32', BAND_COUNT);
    const demandBands = builder.allocate('demand-bands', 'uint32', DEMAND_COUNT);
    addDriveTimeCatchmentRecipe(builder.graph, {
      topology: {...topology, nodePositions: nodePositions.view},
      facilities: facilityBuffer.view,
      demand: builder.upload('demand', 'float32x2', demand).view,
      costBreaks: breaks.view,
      bandCapacity: BAND_COUNT,
      maximumIterations: 64,
      output: {demandBands: demandBands.view}
    });
    return {
      outputs: {
        ...uploadRoads(builder, network),
        demandBands,
        demands: builder.upload('demands', 'float32x2', demand),
        facilities: facilityPoints
      },
      update: current => {
        facilityBuffer.write(facilities.subarray(0, Math.round(current.facilityCount) * 2));
        facilityPoints.write(facilities);
        breaks.write(
          Float32Array.from({length: BAND_COUNT}, (_, band) => (band + 1) * current.breakWidth)
        );
      }
    };
  },
  rebuildOn: ['facilityCount'],
  interaction: {
    kind: 'drag',
    hint: 'drag a facility',
    rest: ({inputs, toUnit}) =>
      toUnit([inputs.drag.defaultPositions[0], inputs.drag.defaultPositions[1]]),
    resolve(pointer, {inputs, values}) {
      const move = moveFacility(
        inputs.scene,
        inputs.drag,
        pointer.x,
        pointer.y,
        Math.round(values.facilityCount)
      );
      const at: [number, number] = [
        inputs.network.nodes[move.node * 2],
        inputs.network.nodes[move.node * 2 + 1]
      ];
      return {
        rerun: move.changed,
        glyphs: [1, 2, 3, 4].map(band => ({
          shape: 'ring' as const,
          at,
          radius: band * values.breakWidth,
          dashed: true
        })),
        readout: `facility ${move.facility + 1} · breaks every ${values.breakWidth.toFixed(2)}`
      };
    }
  },
  draw: ({outputs, values, theme}) => [
    getRoadContext(outputs),
    {
      layer: 'dots',
      positions: outputs.demands,
      values: outputs.demandBands,
      ramp: 'sequential',
      range: [-1, 4],
      radius: 5
    },
    ...getFacilityMarkers(outputs.facilities, theme, {
      count: Math.round(values.facilityCount),
      radius: 10
    })
  ]
});
