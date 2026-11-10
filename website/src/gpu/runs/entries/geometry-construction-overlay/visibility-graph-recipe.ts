// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addVisibilityGraphRecipe links every pair of street points that can see each other (a clear line
// within maximumDistance) and scores each point. Edges are the visible pairs the recipe wrote; point
// color is the normalized choice (through-movement) score: bright points lie on the most routes.

import {addVisibilityGraphRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {chooseSticky, createPointIndex} from '../../pick';
import {addKernelPass, getViewBuffer} from './shared/kernel';
import {createCityLayout, getBuildings, sampleOpenSpace} from './shared/urban';
import {parameters} from './visibility-graph-recipe.parameters';

const EDGE_CAPACITY = 6000;

export default defineOperationRun({
  parameters,
  caption: 'Visible pairs of street points, colored by how many shortest routes pass through them.',
  createInputs: ({world}) => {
    const layout = createCityLayout(world, 22);
    const points = sampleOpenSpace(layout, world);
    return {layout, points, index: createPointIndex(points), focus: {row: -1}};
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
  // The pointer picks a node; the visible pairs the recipe wrote for it are drawn bright.
  interaction: {
    kind: 'probe',
    hint: 'hover a point',
    resolve(pointer, {inputs, outputs, values, peek}) {
      const row = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.focus.row);
      inputs.focus.row = row;
      if (row < 0) return {};
      const x = inputs.points[row * 2];
      const y = inputs.points[row * 2 + 1];
      const pairs = peek(outputs.edges);
      const choice = peek(outputs.nodeValues);
      const pointCount = inputs.points.length / 2;
      const glyphs: RunGlyph[] = [
        {shape: 'ring', at: [x, y], radius: values.maximumDistance, dashed: true},
        {shape: 'dot', at: [x, y]}
      ];
      let visibleCount = 0;
      if (pairs) {
        for (let slot = 0; slot < pairs.length; slot += 2) {
          const other =
            pairs[slot] === row ? pairs[slot + 1] : pairs[slot + 1] === row ? pairs[slot] : -1;
          if (other < 0 || other >= pointCount) continue;
          visibleCount++;
          glyphs.push({
            shape: 'line',
            from: [x, y],
            to: [inputs.points[other * 2], inputs.points[other * 2 + 1]]
          });
        }
      }
      return {
        focus: row,
        glyphs,
        readout: `node ${row} · ${visibleCount} visible${choice && Number.isFinite(choice[row]) ? ` · choice ${choice[row].toFixed(2)}` : ''}`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.walls, radius: 2.5},
    {
      layer: 'segments',
      nodes: outputs.nodes,
      edges: outputs.edges,
      alpha: 0.12,
      color: [0.38, 0.68, 1, 1],
      radius: 0.7
    },
    {layer: 'dots', positions: outputs.nodes, values: outputs.nodeValues, radius: 4, range: 'auto'}
  ]
});
