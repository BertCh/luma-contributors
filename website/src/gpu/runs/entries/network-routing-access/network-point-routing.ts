// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkPointRouting: routing between positions along directed roads, with explicit turns
// (no U-turns here). One query per directed road; roads are colored by the cost of reaching them
// from a point partway along the source road.

import {GPUNetworkPointRouting} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {
  createSlotSegments,
  createSmallRoadNetwork,
  findNearestSlot,
  getEndpointMarkers,
  getSlotValues,
  uploadSlotLines
} from './shared/roads';
import {parameters} from './network-point-routing.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Cost to reach each directed road from a point on the source road, turns included.',
  createInputs: ({random, world}) => {
    const network = createSmallRoadNetwork(random, world, 35);
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
    const defaultRoad = network.offsets[network.findNearestNode(0.3, 0.5)];
    for (let query = 0; query < edgeCount * 2; query++) {
      queries[query * 2] = defaultRoad;
      queries[query * 2 + 1] = query % edgeCount;
    }
    return {
      network,
      defaultRoad,
      sourceRoad: defaultRoad,
      slotSegments: createSlotSegments(network),
      turns: Uint32Array.from(turns),
      queries,
      edgeCosts: new Float32Array(edgeCount)
    };
  },
  build(builder, inputs) {
    const {network, queries, slotSegments} = inputs;
    const queryCount = network.slotCount * 2;
    const costs = builder.allocate('edge-costs', 'float32', network.slotCount);
    const fractions = builder.allocate('fractions', 'float32x2', queryCount);
    const queryCosts = builder.allocate('query-costs', 'float32', queryCount);
    const queryBuffer = builder.upload('queries', 'uint32x2', queries);
    const sourcePosition = builder.allocate('source-position', 'float32x2', 1);
    const status = builder.allocate('status', 'uint32', queryCount);
    const predecessors = builder.allocate('predecessors', 'uint32', queryCount * network.slotCount);
    builder.graph.add(
      new GPUNetworkPointRouting({
        turns: builder.upload('turns', 'uint32x2', inputs.turns).view,
        costs: costs.view,
        queries: queryBuffer.view,
        fractions: fractions.view,
        output: {costs: queryCosts.view, status: status.view, predecessorEdges: predecessors.view}
      })
    );
    return {
      outputs: {slots: uploadSlotLines(builder, network, 0.012), queryCosts, sourcePosition},
      update: values => {
        for (let query = 0; query < queryCount; query++) queries[query * 2] = inputs.sourceRoad;
        queryBuffer.write(queries);
        const slot = inputs.sourceRoad * 4;
        sourcePosition.write(
          Float32Array.of(
            slotSegments[slot] +
              (slotSegments[slot + 2] - slotSegments[slot]) * values.sourceFraction,
            slotSegments[slot + 1] +
              (slotSegments[slot + 3] - slotSegments[slot + 1]) * values.sourceFraction
          )
        );
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
  // The pointer picks the directed source road and the point along it that the costs start from.
  interaction: {
    kind: 'place',
    hint: 'place the source',
    rest: ({inputs, toUnit}) => {
      const slot = inputs.defaultRoad * 4;
      const segments = inputs.slotSegments;
      const deltaX = segments[slot + 2] - segments[slot];
      const deltaY = segments[slot + 3] - segments[slot + 1];
      const length = Math.hypot(deltaX, deltaY) || 1;
      // Start a hair to the right of the direction of travel, where the pick prefers this slot.
      return toUnit([
        segments[slot] + deltaX * parameters[0].default + (deltaY / length) * 0.004,
        segments[slot + 1] + deltaY * parameters[0].default - (deltaX / length) * 0.004
      ]);
    },
    resolve(pointer, {inputs, outputs, peek}) {
      const hit = findNearestSlot(inputs.slotSegments, pointer.x, pointer.y);
      const changed = hit.slot !== inputs.sourceRoad;
      inputs.sourceRoad = hit.slot;
      const fraction = Math.round(hit.fraction * 20) / 20;
      const costs = peek(outputs.queryCosts);
      let maximumCost = 0;
      if (costs) {
        for (const cost of costs)
          if (Number.isFinite(cost) && cost < 1e20) maximumCost = Math.max(maximumCost, cost);
      }
      return {
        values: {sourceFraction: fraction},
        rerun: changed,
        glyphs: [{shape: 'crosshair', at: hit.point, size: 12}],
        readout: `source road ${hit.slot} at ${fraction.toFixed(2)} · max cost ${maximumCost.toFixed(2)}`
      };
    }
  },
  draw: ({outputs, theme}) => [
    getSlotValues(outputs.slots, outputs.queryCosts, {radius: 3, range: [0, 3], bands: 12}),
    ...getEndpointMarkers(outputs.sourcePosition, theme, {colorIndex: 4})
  ]
});
