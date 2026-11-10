// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addNetworkDisruptionRecipe: close roads (edgeMask) and compare origin-destination costs and
// accessibility before and after. Closed roads are drawn in the second color; origins show how
// much reachable opportunity each one lost.

import {
  addNetworkDisruptionRecipe,
  getGPUNetworkAccessibilityParameterValues
} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {uploadNetworkTopology} from '../../inputs';
import {
  gatherPositions,
  createRoadNetwork,
  createRoads,
  getMaskLook,
  getSlotValues,
  pickNodes,
  uploadSlotLines
} from './shared/roads';

const ORIGIN_COUNT = 40;

import {parameters} from './network-disruption.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Closed roads, and the accessibility each origin loses (diverging).',
  createInputs: ({random, world}) => {
    const network = createRoadNetwork(createRoads(random, world, 12), 0);
    const origins = pickNodes(network, world, ORIGIN_COUNT, 5);
    const jobs = Float32Array.from({length: network.nodeCount}, () =>
      random() < 0.15 ? 1 + random() * 4 : 0
    );
    const middles = new Float32Array(network.slotCount * 2);
    for (let slot = 0; slot < network.slotCount; slot++) {
      const source = network.slotSources[slot];
      const target = network.neighbors[slot];
      middles[slot * 2] = (network.nodes[source * 2] + network.nodes[target * 2]) / 2;
      middles[slot * 2 + 1] = (network.nodes[source * 2 + 1] + network.nodes[target * 2 + 1]) / 2;
    }
    return {
      network,
      origins,
      jobs,
      middles,
      mask: new Uint32Array(network.slotCount),
      center: [world.width / 2, 0.5] as [number, number],
      closedSlots: 0
    };
  },
  build(builder, inputs) {
    const {network, origins, jobs, middles, mask} = inputs;
    const {topology} = uploadNetworkTopology(builder, network, network.lengths);
    const edgeMask = builder.allocate('edge-mask', 'uint32', network.slotCount);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const accessibilityDifferences = builder.allocate(
      'accessibility-differences',
      'float32',
      ORIGIN_COUNT
    );
    addNetworkDisruptionRecipe(builder.graph, {
      search: {
        topology,
        startNodes: builder.upload('origins', 'uint32', origins).view,
        maximumIterations: 64
      },
      rowCount: ORIGIN_COUNT,
      edgeMask: edgeMask.view,
      opportunityWeights: builder.upload('jobs', 'float32', jobs).view,
      parameters: parameterBuffer.view,
      outputs: {accessibilityDifferences: accessibilityDifferences.view}
    });
    return {
      outputs: {
        slots: uploadSlotLines(builder, network),
        edgeMask,
        accessibilityDifferences,
        origins: builder.upload(
          'origin-positions',
          'float32x2',
          gatherPositions(network.nodes, origins)
        )
      },
      update: values => {
        // Every road whose middle lies inside the closure disc is closed (mask 0), both directions.
        inputs.closedSlots = 0;
        for (let slot = 0; slot < network.slotCount; slot++) {
          const isClosed =
            Math.hypot(
              middles[slot * 2] - inputs.center[0],
              middles[slot * 2 + 1] - inputs.center[1]
            ) < values.closureRadius;
          mask[slot] = isClosed ? 0 : 1;
          if (isClosed) inputs.closedSlots++;
        }
        edgeMask.write(mask);
        parameterBuffer.write(
          getGPUNetworkAccessibilityParameterValues({threshold: values.threshold})
        );
      }
    };
  },
  // The pointer places the closure: every road inside the disc is closed.
  interaction: {
    kind: 'place',
    hint: 'close the roads here',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const changed = pointer.x !== inputs.center[0] || pointer.y !== inputs.center[1];
      inputs.center = [pointer.x, pointer.y];
      const differences = peek(outputs.accessibilityDifferences);
      let total = 0;
      if (differences) for (const value of differences) total += Number.isFinite(value) ? value : 0;
      const mean = differences ? total / differences.length : 0;
      return {
        rerun: changed,
        glyphs: [{shape: 'ring', at: inputs.center, radius: values.closureRadius, dashed: true}],
        readout: `closed ${inputs.closedSlots / 2} roads · mean access ${mean.toFixed(2)}`
      };
    }
  },
  draw: ({outputs, theme}) => [
    getSlotValues(outputs.slots, outputs.edgeMask, {
      radius: 1.8,
      ...getMaskLook(theme, {danger: true})
    }),
    {
      layer: 'dots',
      positions: outputs.origins,
      values: outputs.accessibilityDifferences,
      ramp: 'diverging',
      range: [-12, 12],
      radius: 10
    }
  ]
});
