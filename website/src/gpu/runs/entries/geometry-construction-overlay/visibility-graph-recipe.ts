// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addVisibilityGraphRecipe links every pair of street points that can see each other (a clear line
// within maximumDistance) and scores each point. Edges are the visible pairs the recipe wrote; point
// color is the normalized choice (through-movement) score: bright points lie on the most routes.

import {addVisibilityGraphRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addKernelPass, getViewBuffer} from './shared/kernel';
import {createCityLayout, getBuildings} from './shared/urban';
import {parameters} from './visibility-graph-recipe.parameters';

const EDGE_CAPACITY = 6000;
const MAXIMUM_POINTS = 256;

export default defineOperationRun({
  parameters,
  caption: 'Visible pairs of street points, colored by how many shortest routes pass through them.',
  createInputs: ({random, world}) => {
    const layout = createCityLayout(random, world, 6, 3);
    // Street points: a fine grid, kept only where it lies outside the largest possible buildings.
    const largest = getBuildings(layout, 0.9);
    const points: number[] = [];
    const columns = 32;
    const rows = 18;
    for (let row = 0; row < rows; row++) {
      for (let column = 0; column < columns; column++) {
        const x = ((column + 0.5) / columns) * world.width;
        const y = (row + 0.5) / rows;
        const isFree = Array.from({length: largest.count}, (_, building) => building).every(
          building => {
            const [x0, y0, x1, y1] = largest.rectangles.subarray(building * 4, building * 4 + 4);
            return x < x0 - 0.01 || x > x1 + 0.01 || y < y0 - 0.01 || y > y1 + 0.01;
          }
        );
        if (isFree && points.length < MAXIMUM_POINTS * 2) points.push(x, y);
      }
    }
    return {layout, points: Float32Array.from(points)};
  },
  build(builder, inputs, values) {
    const buildings = getBuildings(inputs.layout, values.fill);
    const pointCount = inputs.points.length / 2;
    const positions = builder.upload('points', 'float32x2', inputs.points);
    const segments = builder.upload('walls', 'float32x4', buildings.segments);
    const result = addVisibilityGraphRecipe(builder.graph, {
      positions: positions.view,
      segments: segments.view,
      maximumDistance: values.maximumDistance,
      edgeCapacity: EDGE_CAPACITY,
      tolerance: 0.002
    });
    // Own copies the layers can draw: nodes (plus one off-screen sentinel), edges, node scores.
    const nodes = builder.allocate('graph-nodes', 'float32x2', pointCount + 1);
    const nodeValues = builder.allocate('graph-node-values', 'float32', pointCount + 1);
    const edges = builder.allocate('graph-edges', 'uint32x2', EDGE_CAPACITY);
    addKernelPass(builder, {
      id: 'visibility-node-pass',
      invocationCount: pointCount + 1,
      bindings: [
        {name: 'positions', buffer: positions, access: 'read'},
        {name: 'choice', buffer: getViewBuffer(result.outputs.normalizedChoice), access: 'read'},
        {name: 'nodes', buffer: nodes, access: 'read_write'},
        {name: 'nodeValues', buffer: nodeValues, access: 'read_write'}
      ],
      body: /* wgsl */ `
  let isPoint = index < ${pointCount}u;
  nodes[nodesOffset + index] = select(vec2f(-9.0), positions[positionsOffset + min(index, ${pointCount - 1}u)], isPoint);
  nodeValues[nodeValuesOffset + index] = select(0.0, choice[choiceOffset + min(index, ${pointCount - 1}u)], isPoint);`
    });
    addKernelPass(builder, {
      id: 'visibility-edge-pass',
      invocationCount: EDGE_CAPACITY,
      bindings: [
        {name: 'sourceEdges', buffer: getViewBuffer(result.outputs.edges), access: 'read'},
        {name: 'edges', buffer: edges, access: 'read_write'}
      ],
      body: /* wgsl */ `
  let pair = sourceEdges[sourceEdgesOffset + index];
  let isEdge = pair.x < ${pointCount}u && pair.y < ${pointCount}u;
  edges[edgesOffset + index] = select(vec2u(${pointCount}u, ${pointCount}u), pair, isEdge);`
    });
    const walls = builder.upload('wall-segments', 'float32x4', buildings.segments);
    return {outputs: {walls, nodes, nodeValues, edges}};
  },
  rebuildOn: ['maximumDistance', 'fill'],
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.walls, radius: 2.5},
    {
      layer: 'segments',
      nodes: outputs.nodes,
      edges: outputs.edges,
      alpha: 0.35,
      color: [0.38, 0.68, 1, 1],
      radius: 0.8
    },
    {layer: 'dots', positions: outputs.nodes, values: outputs.nodeValues, radius: 4, range: 'auto'}
  ]
});
