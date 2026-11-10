// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUDorlingCartogram turns centroids into non-overlapping circles whose area follows a value. The
// muted dots are the original centroids; each circle is drawn from the relaxed positions and radii
// the operation wrote, colored by value. `maximumIterations` shows the relaxation progress.

import {GPUDorlingCartogram} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts} from '../../inputs';
import {clamp} from '../../interaction';
import {addKernelPass} from './shared/kernel';
import {parameters} from './dorling-cartogram.parameters';

const CIRCLE_SEGMENTS = 28;
// The playhead snaps to these iteration counts; each is a pre-built graph (buildCache).
const ITERATION_STEPS = [0, 5, 20, 50, 100];

export default defineOperationRun({
  parameters,
  caption: 'Centroids relaxed into circles sized by value; muted dots are where they started.',
  createInputs: ({random, world}) => {
    const tracts = createTracts(random, world, 6);
    // Circle areas in world units squared: 0.012 to 0.042.
    const areas = Float32Array.from(tracts.values, value => 0.012 + 0.03 * value);
    return {
      count: tracts.count,
      centers: tracts.centers,
      areas,
      outline: tracts.outline,
      outlineCount: tracts.outline.length / 4
    };
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
    // One displacement line from every centroid to its relaxed circle center.
    const links = builder.allocate('links', 'float32x4', inputs.count);
    addKernelPass(builder, {
      id: 'link-pass',
      invocationCount: inputs.count,
      bindings: [
        {name: 'start', buffer: positions, access: 'read'},
        {name: 'end', buffer: outputPositions, access: 'read'},
        {name: 'links', buffer: links, access: 'read_write'}
      ],
      body: /* wgsl */ `
  links[linksOffset + index] = vec4f(start[startOffset + index], end[endOffset + index]);`
    });
    const outline = builder.upload('outline', 'float32x4', inputs.outline);
    return {outputs: {positions, nodes, nodeValues, edges, links, outline}};
  },
  rebuildOn: ['maximumIterations', 'areaScale', 'attraction'],
  buildCache: 5,
  // The pointer's x is a playhead over the relaxation: it settles on one of five pre-built
  // iteration counts, and the dashed links show how far each circle has moved from its centroid.
  interaction: {
    kind: 'time',
    hint: 'scrub the relaxation',
    rest: [0.9, 0.5],
    resolve(pointer, {inputs}) {
      const step = clamp(
        Math.floor(pointer.unitX * ITERATION_STEPS.length),
        0,
        ITERATION_STEPS.length - 1
      );
      const maximumIterations = ITERATION_STEPS[step];
      return {
        settle: {maximumIterations},
        glyphs: [{shape: 'curtain', x: pointer.x}],
        readout: `maximumIterations ${maximumIterations} · ${inputs.count} circles`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'lines',
      context: true,
      segments: outputs.outline,
      count: inputs.outlineCount,
      radius: 1,
      alpha: 0.5
    },
    {layer: 'lines', segments: outputs.links, color: [0.8, 0.8, 0.85, 0.35], radius: 1},
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
