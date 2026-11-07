// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUDorlingCartogram turns centroids into non-overlapping circles whose area follows a value. The
// muted dots are the original centroids; each circle is drawn from the relaxed positions and radii
// the operation wrote, colored by value. `maximumIterations` shows the relaxation progress.

import {GPUDorlingCartogram} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts} from '../../inputs';
import {addKernelPass} from './shared/kernel';
import {parameters} from './dorling-cartogram.parameters';

const CIRCLE_SEGMENTS = 28;

export default defineOperationRun({
  parameters,
  caption: 'Centroids relaxed into circles sized by value; muted dots are where they started.',
  createInputs: ({random, world}) => {
    const tracts = createTracts(random, world, 6);
    // Circle areas in world units squared: 0.012 to 0.042.
    const areas = Float32Array.from(tracts.values, value => 0.012 + 0.03 * value);
    return {count: tracts.count, centers: tracts.centers, areas};
  },
  build(builder, inputs, values) {
    const positions = builder.upload('centroids', 'float32x2', inputs.centers);
    const areas = builder.upload('areas', 'float32', inputs.areas);
    const outputPositions = builder.allocate('circle-centers', 'float32x2', inputs.count);
    const radii = builder.allocate('circle-radii', 'float32', inputs.count);
    const sourceIds = builder.allocate('circle-sources', 'uint32', inputs.count);
    const metrics = builder.allocate('circle-metrics', 'float32x2', 1);
    const status = builder.allocate('circle-status', 'uint32', 4);
    builder.graph.add(
      new GPUDorlingCartogram({
        positions: positions.view,
        values: areas.view,
        maximumIterations: values.maximumIterations,
        areaScale: values.areaScale,
        attraction: values.attraction,
        output: {
          positions: outputPositions.view,
          radii: radii.view,
          sourceIds: sourceIds.view,
          metrics: metrics.view,
          status: status.view
        }
      })
    );
    const nodes = builder.allocate('circle-nodes', 'float32x2', inputs.count * CIRCLE_SEGMENTS);
    const nodeValues = builder.allocate(
      'circle-node-values',
      'float32',
      inputs.count * CIRCLE_SEGMENTS
    );
    const edges = builder.upload(
      'circle-edges',
      'uint32x2',
      Uint32Array.from({length: inputs.count * CIRCLE_SEGMENTS * 2}, (_, slot) => {
        const edge = Math.floor(slot / 2);
        const circle = Math.floor(edge / CIRCLE_SEGMENTS);
        const step = edge % CIRCLE_SEGMENTS;
        return circle * CIRCLE_SEGMENTS + (slot % 2 === 0 ? step : (step + 1) % CIRCLE_SEGMENTS);
      })
    );
    addKernelPass(builder, {
      id: 'circle-pass',
      invocationCount: inputs.count * CIRCLE_SEGMENTS,
      bindings: [
        {name: 'centers', buffer: outputPositions, access: 'read'},
        {name: 'radii', buffer: radii, access: 'read'},
        {name: 'nodes', buffer: nodes, access: 'read_write'},
        {name: 'nodeValues', buffer: nodeValues, access: 'read_write'}
      ],
      body: /* wgsl */ `
  let circle = index / ${CIRCLE_SEGMENTS}u;
  let angle = 6.2831853 * f32(index % ${CIRCLE_SEGMENTS}u) / ${CIRCLE_SEGMENTS}.0;
  let radius = radii[radiiOffset + circle];
  nodes[nodesOffset + index] = centers[centersOffset + circle] + radius * vec2f(cos(angle), sin(angle));
  nodeValues[nodeValuesOffset + index] = radius;`
    });
    return {outputs: {positions, nodes, nodeValues, edges}};
  },
  rebuildOn: ['maximumIterations', 'areaScale', 'attraction'],
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.positions, radius: 3},
    {
      layer: 'segments',
      nodes: outputs.nodes,
      edges: outputs.edges,
      values: outputs.nodeValues,
      radius: 2.2
    }
  ]
});
