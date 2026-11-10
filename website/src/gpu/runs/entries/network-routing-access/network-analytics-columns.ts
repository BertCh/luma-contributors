// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkAnalyticsColumns: degree, normalized degree and connected components as per-node
// columns. Closing roads splits the network into components.

import {GPUNetworkAnalyticsColumns} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {chooseSticky} from '../../pick';
import {
  createNodeIndex,
  createRoadNetwork,
  createRoads,
  getRoadContext,
  getRoadValues,
  uploadRoadTopology,
  uploadRoads
} from './shared/roads';
import {parameters} from './network-analytics-columns.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Per-node analytics columns: component label, degree, normalized degree.',
  createInputs: ({random, world}) => {
    const base = createRoads(random, world);
    return {
      base,
      index: createNodeIndex(base),
      node: base.findNearestNode(world.width / 2, 0.5),
      defaultNode: base.findNearestNode(world.width / 2, 0.5)
    };
  },
  build(builder, {base}, values) {
    const network = createRoadNetwork(base, values.closures);
    const {topology} = uploadRoadTopology(builder, network);
    const degree = builder.allocate('degree', 'uint32', network.nodeCount);
    const normalized = builder.allocate('degree-normalized', 'float32', network.nodeCount);
    const components = builder.allocate('components', 'uint32', network.nodeCount);
    builder.graph.add(
      new GPUNetworkAnalyticsColumns({
        topology,
        reverseTopology: topology,
        degree: {output: degree.view, normalized: normalized.view},
        components: {output: components.view, maximumIterations: 64}
      })
    );
    return {
      outputs: {
        ...uploadRoads(builder, network),
        degree,
        normalized,
        components,
        selection: builder.allocate('selection', 'uint32', network.nodeCount)
      }
    };
  },
  // Probe: the hovered intersection's component, every intersection that can reach it.
  interaction: {
    kind: 'probe',
    hint: 'hover an intersection',
    rest: ({inputs, toUnit}) =>
      toUnit([
        inputs.base.nodes[inputs.defaultNode * 2],
        inputs.base.nodes[inputs.defaultNode * 2 + 1]
      ]),
    resolve(pointer, {inputs, outputs, peek}) {
      inputs.node = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.node);
      const node = inputs.node;
      const components = peek(outputs.components);
      const degrees = peek(outputs.degree);
      const selection = new Uint32Array(inputs.base.nodeCount).fill(0xffffffff);
      let size = 0;
      if (components) {
        for (let other = 0; other < selection.length; other++) {
          if (components[other] === components[node]) {
            selection[other] = 0;
            size++;
          }
        }
      }
      outputs.selection.write(selection);
      return {
        glyphs: [
          {
            shape: 'crosshair',
            at: [inputs.base.nodes[node * 2], inputs.base.nodes[node * 2 + 1]],
            size: 14
          }
        ],
        readout: `component ${components ? components[node] : '-'} · ${size} nodes · degree ${degrees ? degrees[node] : '-'}`
      };
    }
  },
  draw: ({outputs, values, theme}) => {
    const columns = [outputs.components, outputs.degree, outputs.normalized];
    return [
      getRoadContext(outputs),
      getRoadValues(outputs, columns[values.column], {
        radius: 2.2,
        ramp: values.column === 0 ? 'categorical' : 'sequential',
        range: values.column === 1 ? [0, 5] : values.column === 2 ? [0, 1] : undefined
      }),
      {
        layer: 'dots',
        positions: outputs.nodes,
        values: outputs.selection,
        ramp: 'categorical',
        categories: [[...theme.ink, 1]],
        radius: 3
      }
    ];
  }
});
