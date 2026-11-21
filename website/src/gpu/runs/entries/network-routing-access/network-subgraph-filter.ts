// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkSubgraphFilter: keeps the roads whose speed lies in a range. Surviving roads are
// colored, the rest stay muted; intersections left without a road drop out.

import {
  GPUNetworkSubgraphFilter,
  getGPUNetworkSubgraphFilterParameterValues
} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {clamp, snapToParameter, findParameter} from '../../interaction';
import {
  createRoadNetwork,
  createRoads,
  getMaskLook,
  getRoadContext,
  getSlotValues,
  uploadRoadTopology,
  uploadRoads,
  uploadSlotLines
} from './shared/roads';
import {parameters} from './network-subgraph-filter.parameters';

const MINIMUM = findParameter(parameters, 'minimumSpeed');
const MAXIMUM = findParameter(parameters, 'maximumSpeed');
const CENTER = (MINIMUM.default + MAXIMUM.default) / 2;
const WIDTH = MAXIMUM.default - MINIMUM.default;

const LAYOUT = {nodeColumnCount: 0, edgeColumnCount: 1, hasEdgeTimes: false} as const;

export default defineOperationRun({
  parameters,
  caption: 'Roads kept by the speed filter, colored; dropped roads are muted.',
  createInputs: ({random, world}) => {
    const network = createRoadNetwork(createRoads(random, world), 0);
    // One speed per road, shared by both directed slots: longer roads are faster (arterials), with
    // a little noise so classes overlap.
    const roadLengths = Float32Array.from({length: network.slotCount}, (_, slot) =>
      Math.min(network.lengths[slot], network.lengths[network.reverseSlots[slot]])
    );
    const sorted = Float32Array.from(roadLengths).sort();
    const speeds = Float32Array.from({length: network.slotCount}, (_, slot) => {
      const road = Math.min(slot, network.reverseSlots[slot]);
      const rank = sorted.indexOf(roadLengths[slot]) / Math.max(sorted.length - 1, 1);
      const noise = (Math.imul(road + 1, 2654435761) >>> 0) / 4294967296;
      return clamp(rank * 0.85 + noise * 0.15, 0, 1);
    });
    return {network, speeds};
  },
  build(builder, {network, speeds}) {
    const {topology} = uploadRoadTopology(builder, network);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const nodeMask = builder.allocate('node-mask', 'uint32', network.nodeCount);
    const edgeMask = builder.allocate('edge-mask', 'uint32', network.slotCount);
    builder.graph.add(
      new GPUNetworkSubgraphFilter({
        topology,
        edgeColumns: [builder.upload('speed', 'float32', speeds).view],
        parameters: parameterBuffer.view,
        dropIsolated: true,
        output: {nodeMask: nodeMask.view, edgeMask: edgeMask.view}
      })
    );
    return {
      outputs: {
        ...uploadRoads(builder, network),
        slots: uploadSlotLines(builder, network),
        edgeMask,
        nodeMask
      },
      update: values =>
        parameterBuffer.write(
          getGPUNetworkSubgraphFilterParameterValues(LAYOUT, {
            edgeAttributeRanges: [[values.minimumSpeed, values.maximumSpeed]]
          })
        )
    };
  },
  // Window: x is the centre of the speed range the filter keeps, y its width.
  interaction: {
    kind: 'window',
    hint: 'slide the speed window',
    rest: [0.05 + 0.9 * CENTER, 1 - (0.05 + 0.9 * ((WIDTH - 0.1) / 0.9))],
    resolve(pointer, {outputs, world, peek}) {
      const center = clamp((pointer.unitX - 0.05) / 0.9, 0, 1);
      const width = 0.1 + 0.9 * clamp((1 - pointer.unitY - 0.05) / 0.9, 0, 1);
      const minimumSpeed = snapToParameter(MINIMUM, center - width / 2);
      const maximumSpeed = snapToParameter(MAXIMUM, center + width / 2);
      const mask = peek(outputs.edgeMask);
      let kept = 0;
      if (mask) for (const value of mask) kept += value === 1 ? 1 : 0;
      const left = world.width * 0.06;
      const scale = world.width * 0.88;
      return {
        values: {minimumSpeed, maximumSpeed},
        glyphs: [
          {shape: 'line', from: [left, 0.04], to: [left + scale, 0.04]},
          {
            shape: 'box',
            minimum: [left + scale * minimumSpeed, 0.02],
            maximum: [left + scale * maximumSpeed, 0.06]
          }
        ],
        readout: `speed ${minimumSpeed.toFixed(2)}-${maximumSpeed.toFixed(2)} · ${mask ? Math.round((100 * kept) / mask.length) : 0}% of roads kept`
      };
    }
  },
  draw: ({outputs, theme}) => [
    getRoadContext(outputs),
    getSlotValues(outputs.slots, outputs.edgeMask, {radius: 2.2, ...getMaskLook(theme)})
  ]
});
