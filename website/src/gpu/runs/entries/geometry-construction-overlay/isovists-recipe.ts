// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addIsovistsRecipe casts rays from viewpoints against building walls and reports where each ray
// stops: the visible area (isovist) from every viewpoint. Walls are muted; each viewpoint's visible
// polygon is drawn from the ray end points the recipe wrote, in the viewpoint's color.

import {addIsovistsRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addKernelPass, getViewBuffer} from './shared/kernel';
import {createCityLayout, getBuildings, sampleStreet} from './shared/urban';
import {parameters} from './isovists-recipe.parameters';

// Viewpoint 0 follows the pointer; viewpoint 1 stays put for comparison.
const VIEWPOINT_COUNT = 2;

export default defineOperationRun({
  parameters,
  caption:
    'Visible area from two viewpoints, one under the pointer: rays hit a wall at the first building wall.',
  createInputs: ({world}) => {
    const layout = createCityLayout(world, 24);
    const viewpoints = sampleStreet(layout, VIEWPOINT_COUNT);
    return {layout, viewpoints, initial: [viewpoints[0], viewpoints[1]] as const};
  },
  build(builder, inputs, values) {
    const buildings = getBuildings(inputs.layout, values.fill);
    const rayCount = values.rayCount;
    const positions = builder.upload('viewpoints', 'float32x2', inputs.viewpoints);
    const segments = builder.upload('walls', 'float32x4', buildings.segments);
    const result = addIsovistsRecipe(builder.graph, {
      positions: positions.view,
      segments: segments.view,
      rayCount,
      maximumDistance: values.maximumDistance,
      sampling: 'uniform'
    });
    // Ray end points (missing rays are NaN) become drawable nodes colored by viewpoint.
    const nodeCount = VIEWPOINT_COUNT * rayCount;
    const nodes = builder.allocate('ray-nodes', 'float32x2', nodeCount);
    addKernelPass(builder, {
      id: 'isovist-publish-pass',
      invocationCount: nodeCount,
      bindings: [
        {name: 'endpoints', buffer: getViewBuffer(result.outputs.endpoints), access: 'read'},
        {name: 'nodes', buffer: nodes, access: 'read_write'}
      ],
      body: /* wgsl */ `
  let point = endpoints[endpointsOffset + index];
  nodes[nodesOffset + index] = select(vec2f(-9.0), point, point.x == point.x && point.y == point.y);`
    });
    const edges = builder.upload(
      'ray-edges',
      'uint32x2',
      Uint32Array.from({length: nodeCount * 2}, (_, slot) => {
        const edge = Math.floor(slot / 2);
        const viewpoint = Math.floor(edge / rayCount);
        const ray = edge % rayCount;
        return viewpoint * rayCount + (slot % 2 === 0 ? ray : (ray + 1) % rayCount);
      })
    );
    const owners = builder.upload(
      'ray-owners',
      'uint32',
      Uint32Array.from({length: nodeCount}, (_, node) => Math.floor(node / rayCount))
    );
    const rays = builder.allocate('rays', 'float32x4', nodeCount);
    addKernelPass(builder, {
      id: 'isovist-ray-pass',
      invocationCount: nodeCount,
      bindings: [
        {name: 'viewpoints', buffer: positions, access: 'read'},
        {name: 'nodes', buffer: nodes, access: 'read'},
        {name: 'rays', buffer: rays, access: 'read_write'}
      ],
      body: /* wgsl */ `
  let end = nodes[nodesOffset + index];
  rays[raysOffset + index] = select(vec4f(-9.0), vec4f(viewpoints[viewpointsOffset + index / ${rayCount}u], end), end.x > -8.0);`
    });
    const walls = builder.upload('wall-segments', 'float32x4', buildings.segments);
    return {
      outputs: {walls, nodes, edges, owners, rays, positions},
      update: () => positions.write(inputs.viewpoints)
    };
  },
  rebuildOn: ['maximumDistance', 'rayCount', 'fill'],
  // The pointer is the first viewpoint: the rays are recast from it and stop at the first wall.
  interaction: {
    kind: 'place',
    hint: 'move the viewpoint',
    rest: ({inputs, toUnit}) => toUnit(inputs.initial),
    resolve(pointer, {inputs, outputs, values, peek}) {
      inputs.viewpoints[0] = pointer.x;
      inputs.viewpoints[1] = pointer.y;
      const nodes = peek(outputs.nodes);
      let stopped = 0;
      if (nodes) {
        for (let ray = 0; ray < values.rayCount; ray++) {
          const length = Math.hypot(nodes[ray * 2] - pointer.x, nodes[ray * 2 + 1] - pointer.y);
          if (nodes[ray * 2] > -8 && length < values.maximumDistance - 1e-3) stopped++;
        }
      }
      return {
        rerun: true,
        glyphs: [
          {shape: 'ring', at: [pointer.x, pointer.y], radius: values.maximumDistance, dashed: true},
          {shape: 'crosshair', at: [pointer.x, pointer.y]}
        ],
        readout: `maximumDistance ${values.maximumDistance.toFixed(2)}${nodes ? ` · ${stopped} of ${values.rayCount} rays stop` : ''}`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.walls, radius: 2.5},
    {
      layer: 'segments',
      nodes: outputs.nodes,
      edges: outputs.edges,
      values: outputs.owners,
      radius: 2
    },
    {layer: 'lines', segments: outputs.rays, alpha: 0.2, radius: 0.8},
    {layer: 'dots', positions: outputs.positions, values: undefined, radius: 6, color: [1, 1, 1, 1]}
  ]
});
