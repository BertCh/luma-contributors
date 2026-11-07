// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkIsochrones: contoured travel-time areas around an origin. Network costs are splatted to
// a raster and cut into bands by `costBreaks`; each band polygon is drawn from the triangles the
// operation wrote.

import {
  GPUNetworkIsochrones,
  getGPUNetworkIsochronesParameterValues
} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {uploadNetworkTopology} from '../../inputs';
import {createRoads, getRoadContext, uploadRoads} from './shared/roads';
import {parameters} from './network-isochrones.parameters';

const BREAK_COUNT = 4;
const TRIANGLE_CAPACITY = 20000;
const WIDTH = 128;
const HEIGHT = 72;

export default defineOperationRun({
  parameters,
  caption: 'Travel-time bands around the origin, from the isoband triangles.',
  createInputs: ({random, world}) => {
    const network = createRoads(random, world, 16);
    return {
      network,
      world,
      origin: network.findNearestNode(world.width * 0.5, 0.5),
      costs: new Float32Array(network.slotCount)
    };
  },
  build(builder, inputs) {
    const {network, world} = inputs;
    const {topology, weights} = uploadNetworkTopology(builder, network);
    const nodePositions = builder.upload('node-positions', 'float32x2', network.nodes);
    const breaks = builder.allocate('cost-breaks', 'float32', BREAK_COUNT);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const nodeCosts = builder.allocate('node-costs', 'float32', network.nodeCount);
    const triangles = builder.allocate('triangles', 'float32x2', TRIANGLE_CAPACITY * 3);
    const triangleBands = builder.allocate('triangle-bands', 'uint32', TRIANGLE_CAPACITY);
    const count = builder.allocate('count', 'uint32', 1);
    const requiredCount = builder.allocate('required-count', 'uint32', 1);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    const converged = builder.allocate('converged', 'uint32', 1);
    builder.graph.add(
      new GPUNetworkIsochrones({
        topology: {...topology, nodePositions: nodePositions.view},
        costs: nodeCosts.view,
        startNodes: builder.upload('start-nodes', 'uint32', Uint32Array.of(inputs.origin)).view,
        maximumIterations: 64,
        localIterations: 16,
        costBreaks: breaks.view,
        parameters: parameterBuffer.view,
        output: {status: {converged: converged.view}},
        raster: {
          width: WIDTH,
          height: HEIGHT,
          output: {
            triangles: triangles.view,
            triangleBands: triangleBands.view,
            status: {count: count.view, requiredCount: requiredCount.view, overflow: overflow.view}
          }
        }
      })
    );
    // Each triangle is its own polygon: vertex ids 0..3n and owner row = triangle index.
    const indices = Uint32Array.from({length: TRIANGLE_CAPACITY * 3}, (_, index) => index);
    const owners = Uint32Array.from({length: TRIANGLE_CAPACITY}, (_, index) => index);
    return {
      outputs: {
        ...uploadRoads(builder, network),
        triangles,
        triangleBands,
        indices: builder.upload('indices', 'uint32', indices),
        owners: builder.upload('owners', 'uint32', owners)
      },
      update: values => {
        network.fillWeights(inputs.costs, values.congestion);
        weights.write(inputs.costs);
        triangles.write(new Float32Array(TRIANGLE_CAPACITY * 6));
        breaks.write(
          Float32Array.from({length: BREAK_COUNT}, (_, band) => (band + 1) * values.breakWidth)
        );
        parameterBuffer.write(
          getGPUNetworkIsochronesParameterValues({
            breakCount: BREAK_COUNT,
            extent: [0, 0, world.width, 1],
            bufferRadius: values.bufferRadius
          })
        );
      }
    };
  },
  draw: ({outputs}) => [
    getRoadContext(outputs),
    {
      layer: 'polygons',
      positions: outputs.triangles,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.triangleBands,
      triangleCount: TRIANGLE_CAPACITY,
      ramp: 'sequential',
      range: [-1, 5],
      alpha: 0.8
    }
  ]
});
