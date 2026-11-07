// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkNoding: build a routable network from raw linestrings. Lines are cut at every crossing
// (muted: the input lines); the bright edges are the noded pieces and the dots the nodes, with end
// points closer than `tolerance` snapped together.

import {GPUNetworkNoding} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {addEdgePairs} from './shared/display';
import {parameters} from './network-noding.parameters';

const NODE_CAPACITY = 512;
const PIECE_CAPACITY = 512;
const PIECE_VERTEX_CAPACITY = 4096;
const VERTEX_COUNT = 9;

export default defineOperationRun({
  parameters,
  caption: 'Lines cut at crossings: noded edges and the nodes they snap to.',
  createInputs: ({random, world}) => {
    // Streets from one random point to another, with a little wobble, so they cross.
    const vertices = new Float32Array(16 * VERTEX_COUNT * 2);
    for (let line = 0; line < 16; line++) {
      const startX = random() * world.width;
      const startY = random();
      const endX = random() * world.width;
      const endY = random();
      for (let vertex = 0; vertex < VERTEX_COUNT; vertex++) {
        const amount = vertex / (VERTEX_COUNT - 1);
        const index = (line * VERTEX_COUNT + vertex) * 2;
        vertices[index] = startX + (endX - startX) * amount + (random() - 0.5) * 0.04;
        vertices[index + 1] = startY + (endY - startY) * amount + (random() - 0.5) * 0.04;
      }
    }
    return {vertices};
  },
  build(builder, lines, values) {
    const lineCount = Math.round(values.lineCount);
    const vertexCount = lineCount * VERTEX_COUNT;
    const tolerance = builder.allocate('tolerance', 'float32', 1);
    const pieceOffsets = builder.allocate('piece-offsets', 'uint32', PIECE_CAPACITY + 1);
    const piecePositions = builder.allocate('piece-positions', 'float32x2', PIECE_VERTEX_CAPACITY);
    const pieceSources = builder.allocate('piece-sources', 'uint32', PIECE_CAPACITY);
    const pieceCount = builder.allocate('piece-count', 'uint32', 1);
    const pieceOverflow = builder.allocate('piece-overflow', 'uint32', 1);
    const nodePositions = builder.allocate('node-positions', 'float32x2', NODE_CAPACITY);
    const nodeCount = builder.allocate('node-count', 'uint32', 1);
    const fromNodes = builder.allocate('from-nodes', 'uint32', PIECE_CAPACITY);
    const toNodes = builder.allocate('to-nodes', 'uint32', PIECE_CAPACITY);
    const lengths = builder.allocate('lengths', 'float32', PIECE_CAPACITY);
    const offsets = builder.allocate('offsets', 'uint32', NODE_CAPACITY + 1);
    const neighbors = builder.allocate('neighbors', 'uint32', PIECE_CAPACITY * 2);
    const weights = builder.allocate('weights', 'float32', PIECE_CAPACITY * 2);
    builder.graph.add(
      new GPUNetworkNoding({
        lines: {
          kind: 'lines',
          positions: builder.upload(
            'line-positions',
            'float32x2',
            lines.vertices.slice(0, vertexCount * 2)
          ).view,
          lineOffsets: builder.upload(
            'line-offsets',
            'uint32',
            Uint32Array.from({length: lineCount + 1}, (_, index) => index * VERTEX_COUNT)
          ).view
        },
        intersectionCapacity: 1024,
        tolerance: tolerance.view,
        output: {
          pieces: {
            geometry: {
              kind: 'lines',
              positions: piecePositions.view,
              lineOffsets: pieceOffsets.view
            },
            sourceIds: pieceSources.view,
            status: {count: pieceCount.view, overflow: pieceOverflow.view}
          },
          nodes: {positions: nodePositions.view, status: {count: nodeCount.view}},
          edges: {fromNodes: fromNodes.view, toNodes: toNodes.view, lengths: lengths.view},
          topology: {offsets: offsets.view, neighbors: neighbors.view, weights: weights.view}
        }
      })
    );
    const segments = new Float32Array((VERTEX_COUNT - 1) * lineCount * 4);
    let segment = 0;
    for (let line = 0; line < lineCount; line++) {
      for (let vertex = 0; vertex < VERTEX_COUNT - 1; vertex++) {
        const base = (line * VERTEX_COUNT + vertex) * 2;
        segments.set(lines.vertices.subarray(base, base + 4), segment++ * 4);
      }
    }
    return {
      outputs: {
        nodePositions,
        edges: addEdgePairs(builder, {
          id: 'noded',
          from: fromNodes,
          to: toNodes,
          fallbackNode: NODE_CAPACITY - 1
        }),
        inputLines: builder.upload('input-lines', 'float32x4', segments)
      },
      update: current => {
        tolerance.write(Float32Array.of(current.tolerance));
        // Unused node rows sit offscreen; the operation overwrites the nodes it finds.
        nodePositions.write(new Float32Array(NODE_CAPACITY * 2).fill(-100));
      }
    };
  },
  rebuildOn: ['lineCount'],
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.inputLines, radius: 1.2},
    {layer: 'segments', nodes: outputs.nodePositions, edges: outputs.edges, radius: 2},
    {layer: 'dots', positions: outputs.nodePositions, radius: 4.5, color: [1, 1, 1, 1]}
  ]
});
