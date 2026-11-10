// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkNoding: build a routable network from raw linestrings. Lines are cut at every crossing
// (muted: the input lines); the bright edges are the noded pieces and the dots the nodes, with end
// points closer than `tolerance` snapped together.

import {GPUNetworkNoding} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {addEdgePairs} from './shared/display';
import {findParameter, snapToParameter} from '../../interaction';
import {findReachPoint} from './shared/roads';
import {parameters} from './network-noding.parameters';

const TOLERANCE = findParameter(parameters, 'tolerance');
const DEFAULT_LINE_COUNT = findParameter(parameters, 'lineCount').default;
import streets from '../../../data/sf-market-streets.json';
import {fitLocalMap} from '../../../data/fit-local-map';

const NODE_CAPACITY = 512;
const PIECE_CAPACITY = 512;
const PIECE_VERTEX_CAPACITY = 4096;
const VERTEX_COUNT = 9;

export default defineOperationRun({
  parameters,
  caption: 'Lines cut at crossings: noded edges and the nodes they snap to.',
  createInputs: ({world}) => {
    const points = streets.lines.flat();
    const project = fitLocalMap(points, world);
    return {vertices: Float32Array.from(points.flatMap(([x, y]) => project(x, y)))};
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
        nodeCount,
        pieceCount,
        nodeIds: builder.upload(
          'node-ids',
          'uint32',
          Uint32Array.from({length: NODE_CAPACITY}, (_, index) => index)
        ),
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
  // Reach: the distance from the nearest input vertex to the pointer is the snapping tolerance.
  interaction: {
    kind: 'reach',
    hint: 'reach to merge nodes',
    rest: ({inputs, world, toUnit}) =>
      toUnit(
        findReachPoint(
          inputs.vertices.subarray(0, DEFAULT_LINE_COUNT * VERTEX_COUNT * 2),
          TOLERANCE.default,
          world
        )
      ),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const vertexCount = Math.round(values.lineCount) * VERTEX_COUNT;
      let nearest = 0;
      let nearestDistance = Infinity;
      for (let vertex = 0; vertex < vertexCount; vertex++) {
        const distance = Math.hypot(
          inputs.vertices[vertex * 2] - pointer.x,
          inputs.vertices[vertex * 2 + 1] - pointer.y
        );
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearest = vertex;
        }
      }
      const tolerance = snapToParameter(TOLERANCE, nearestDistance);
      const nodes = peek(outputs.nodeCount);
      const pieces = peek(outputs.pieceCount);
      return {
        values: {tolerance},
        glyphs: [
          {
            shape: 'ring',
            at: [inputs.vertices[nearest * 2], inputs.vertices[nearest * 2 + 1]],
            radius: tolerance,
            dashed: true
          }
        ],
        readout: `tolerance ${tolerance.toFixed(2)} · ${nodes ? nodes[0] : 0} nodes · ${pieces ? pieces[0] : 0} pieces`
      };
    }
  },
  draw: ({outputs, theme}) => [
    {layer: 'lines', context: true, segments: outputs.inputLines, radius: 1.2},
    {
      layer: 'segments',
      nodes: outputs.nodePositions,
      edges: outputs.edges,
      values: outputs.nodeIds,
      radius: 3
    },
    {layer: 'dots', positions: outputs.nodePositions, radius: 6, color: [...theme.ink, 1]}
  ]
});
