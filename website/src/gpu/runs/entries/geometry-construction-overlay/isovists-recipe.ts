// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addIsovistsRecipe casts rays from viewpoints against building walls and reports where each ray
// stops: the visible area (isovist) from every viewpoint. Walls are muted; each viewpoint's visible
// polygon is drawn from the ray end points the recipe wrote, in the viewpoint's color.

import {addIsovistsRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addKernelPass, getViewBuffer} from './shared/kernel';
import {createCityLayout, getBuildings, getIntersections} from './shared/urban';
import {parameters} from './isovists-recipe.parameters';

const VIEWPOINT_COUNT = 4;

export default defineOperationRun({
  parameters,
  caption: 'Visible area from four viewpoints: rays stop at the first building wall.',
  createInputs: ({random, world}) => {
    const layout = createCityLayout(random, world, 7, 4);
    const intersections = getIntersections(layout);
    const viewpoints = Float32Array.from(
      [3, 8, 13, 17].flatMap(index => [intersections[index * 2], intersections[index * 2 + 1]])
    );
    return {layout, viewpoints};
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
    return {outputs: {walls, nodes, edges, owners, rays, positions}};
  },
  rebuildOn: ['maximumDistance', 'rayCount', 'fill'],
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.walls, radius: 2.5},
    {
      layer: 'segments',
      nodes: outputs.nodes,
      edges: outputs.edges,
      values: outputs.owners,
      radius: 2
    },
    {layer: 'lines', segments: outputs.rays, alpha: 0.12, radius: 0.8},
    {layer: 'dots', positions: outputs.positions, values: undefined, radius: 6, color: [1, 1, 1, 1]}
  ]
});
