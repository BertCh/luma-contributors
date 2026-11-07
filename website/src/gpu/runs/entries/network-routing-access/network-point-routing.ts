// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkPointRouting: routing between positions along directed roads, with explicit turns
// (no U-turns here). One query per directed road; roads are colored by the cost of reaching them
// from a point partway along the source road.

import {GPUNetworkPointRouting} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {createGridRoadNetwork, getSlotValues, uploadSlotLines} from './shared/roads';
import {parameters} from './network-point-routing.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Cost to reach each directed road from a point on the source road, turns included.',
  createInputs: ({random, world}) => {
    const network = createGridRoadNetwork(random, world, 7, 5);
    const edgeCount = network.slotCount;
    const turns: number[] = [];
    for (let slot = 0; slot < edgeCount; slot++) {
      const head = network.neighbors[slot];
      for (let next = network.offsets[head]; next < network.offsets[head + 1]; next++) {
        if (next !== network.reverseSlots[slot]) turns.push(slot, next);
      }
    }
    // Two copies of one query per road: the second copy fills the second half of the slot lines.
    const queries = new Uint32Array(edgeCount * 4);
    const sourceRoad = network.offsets[network.findNearestNode(0.3, 0.5)];
    for (let query = 0; query < edgeCount * 2; query++) {
      queries[query * 2] = sourceRoad;
      queries[query * 2 + 1] = query % edgeCount;
    }
    return {
      network,
      turns: Uint32Array.from(turns),
      queries,
      edgeCosts: new Float32Array(edgeCount)
    };
  },
  build(builder, inputs) {
    const {network, queries} = inputs;
    const queryCount = network.slotCount * 2;
    const costs = builder.allocate('edge-costs', 'float32', network.slotCount);
    const fractions = builder.allocate('fractions', 'float32x2', queryCount);
    const queryCosts = builder.allocate('query-costs', 'float32', queryCount);
    const status = builder.allocate('status', 'uint32', queryCount);
    const predecessors = builder.allocate('predecessors', 'uint32', queryCount * network.slotCount);
    builder.graph.add(
      new GPUNetworkPointRouting({
        turns: builder.upload('turns', 'uint32x2', inputs.turns).view,
        costs: costs.view,
        queries: builder.upload('queries', 'uint32x2', queries).view,
        fractions: fractions.view,
        output: {costs: queryCosts.view, status: status.view, predecessorEdges: predecessors.view}
      })
    );
    return {
      outputs: {slots: uploadSlotLines(builder, network, 0.006), queryCosts},
      update: values => {
        network.fillWeights(inputs.edgeCosts, values.congestion);
        costs.write(inputs.edgeCosts);
        fractions.write(
          Float32Array.from({length: queryCount * 2}, (_, index) =>
            index % 2 === 0 ? values.sourceFraction : values.targetFraction
          )
        );
      }
    };
  },
  draw: ({outputs}) => [getSlotValues(outputs.slots, outputs.queryCosts, {radius: 3})]
});
