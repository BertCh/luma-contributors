// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkStructure: bridges, articulation points and strongly connected components of a road
// network with closures. A bridge is a road whose loss disconnects the network.

import {GPUNetworkStructure} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {
  createRoadNetwork,
  createRoads,
  createSlotSegments,
  getMaskLook,
  getRoadContext,
  getRoadValues,
  getSlotValues,
  uploadRoadTopology,
  uploadRoads,
  uploadSlotLines
} from './shared/roads';
import {parameters} from './network-structure.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Bridges (roads whose loss disconnects), articulation points, strong components.',
  createInputs: ({random, world}) => ({
    base: createRoads(random, world),
    view: null as {
      network: ReturnType<typeof createRoadNetwork>;
      slotSegments: Float32Array;
      cache: Map<number, {cut: Uint32Array; size: number}>;
    } | null
  }),
  build(builder, inputs, values) {
    const network = createRoadNetwork(inputs.base, values.closures);
    inputs.view = {network, slotSegments: createSlotSegments(network), cache: new Map()};
    const {topology} = uploadRoadTopology(builder, network);
    const bridges = builder.allocate('bridges', 'uint32', network.slotCount);
    const articulationPoints = builder.allocate('articulation-points', 'uint32', network.nodeCount);
    const strongComponents = builder.allocate('strong-components', 'uint32', network.nodeCount);
    builder.graph.add(
      new GPUNetworkStructure({
        topology,
        reverseEdgeSlots: builder.upload('reverse-slots', 'uint32', network.reverseSlots).view,
        output: {
          bridges: bridges.view,
          articulationPoints: articulationPoints.view,
          strongComponents: strongComponents.view
        }
      })
    );
    return {
      outputs: {
        ...uploadRoads(builder, network),
        slots: uploadSlotLines(builder, network),
        bridges,
        articulationPoints,
        strongComponents,
        cutSide: builder.allocate('cut-side', 'uint32', network.nodeCount)
      }
    };
  },
  // Probe: the bridge nearest the pointer, and the part of the network its loss cuts off.
  interaction: {
    kind: 'probe',
    hint: 'hover a bridge',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, peek}) {
      const bridges = peek(outputs.bridges);
      const view = inputs.view;
      if (!bridges || !view) return {readout: 'bridges'};
      const {network, slotSegments, cache} = view;
      let best = -1;
      let bestDistance = Infinity;
      for (let slot = 0; slot < network.slotCount; slot++) {
        if (bridges[slot] !== 1) continue;
        const x0 = slotSegments[slot * 4];
        const y0 = slotSegments[slot * 4 + 1];
        const deltaX = slotSegments[slot * 4 + 2] - x0;
        const deltaY = slotSegments[slot * 4 + 3] - y0;
        const lengthSquared = deltaX * deltaX + deltaY * deltaY || 1;
        const fraction = Math.min(
          1,
          Math.max(0, ((pointer.x - x0) * deltaX + (pointer.y - y0) * deltaY) / lengthSquared)
        );
        const distance = Math.hypot(
          pointer.x - x0 - deltaX * fraction,
          pointer.y - y0 - deltaY * fraction
        );
        if (distance < bestDistance) {
          bestDistance = distance;
          best = slot;
        }
      }
      if (best < 0) return {readout: 'no bridges'};
      const road = Math.min(best, network.reverseSlots[best]);
      let result = cache.get(road);
      if (!result) {
        // Flood fill from both ends without the road; the smaller side is what the road keeps connected.
        const floodFill = (start: number, blocked: number) => {
          const seen = new Uint8Array(network.nodeCount);
          const queue = [start];
          seen[start] = 1;
          for (let head = 0; head < queue.length; head++) {
            const node = queue[head];
            for (let slot = network.offsets[node]; slot < network.offsets[node + 1]; slot++) {
              if (slot === road || slot === network.reverseSlots[road]) continue;
              const next = network.neighbors[slot];
              if (seen[next]) continue;
              seen[next] = 1;
              queue.push(next);
            }
          }
          return {seen, size: queue.length, blocked};
        };
        const first = floodFill(network.slotSources[road], road);
        const second = floodFill(network.neighbors[road], road);
        const smaller = first.size <= second.size ? first : second;
        const cut = new Uint32Array(network.nodeCount).fill(0xffffffff);
        for (let node = 0; node < cut.length; node++) if (smaller.seen[node]) cut[node] = 0;
        result = {cut, size: smaller.size};
        cache.set(road, result);
      }
      outputs.cutSide.write(result.cut);
      return {
        glyphs: [
          {
            shape: 'line',
            from: [slotSegments[best * 4], slotSegments[best * 4 + 1]],
            to: [slotSegments[best * 4 + 2], slotSegments[best * 4 + 3]]
          }
        ],
        readout: `bridge ${road} · loses ${result.size} intersections if closed`
      };
    }
  },
  draw: ({outputs, values, theme}) => [
    getRoadContext(outputs),
    values.output === 0
      ? getSlotValues(outputs.slots, outputs.bridges, {radius: 2.4, ...getMaskLook(theme)})
      : getRoadValues(
          outputs,
          values.output === 1 ? outputs.articulationPoints : outputs.strongComponents,
          {radius: 2.4}
        ),
    {
      layer: 'dots',
      positions: outputs.nodes,
      values: outputs.cutSide,
      ramp: 'categorical',
      categories: [[...theme.categories[2], 1]],
      radius: 6
    }
  ]
});
