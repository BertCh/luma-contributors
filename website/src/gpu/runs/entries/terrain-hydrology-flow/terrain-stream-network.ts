// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainStreamNetwork on the D8 directions and stream mask of GPUTerrainFlow: reach polylines
// between headwaters, confluences and outlets, drawn from the vertex positions it wrote, one color
// per reach.

import {GPUTerrainStreamNetwork} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {findNearestOnPolyline} from '../../pick';
import {parameters} from './terrain-stream-network.parameters';
import {addTerrainFlow, createTerrainInputs, getTerrainContextLayer} from './shared/terrain';

const REACH_CAPACITY = 1024;
const VERTEX_CAPACITY = 16384;

export default defineOperationRun({
  parameters,
  caption: 'Reach polylines extracted from the stream mask; they split at every confluence.',
  createInputs: context => createTerrainInputs(context),
  build(builder, terrain) {
    const flow = addTerrainFlow(builder, terrain);
    const sourceCells = builder.allocate('source-cells', 'uint32', REACH_CAPACITY);
    const targetCells = builder.allocate('target-cells', 'uint32', REACH_CAPACITY);
    const vertexOffsets = builder.allocate('vertex-offsets', 'uint32', REACH_CAPACITY + 1);
    const positions = builder.allocate('positions', 'float32x2', VERTEX_CAPACITY);
    const lengths = builder.allocate('lengths', 'float32', REACH_CAPACITY);
    const reachCount = builder.allocate('reach-count', 'uint32', 1);
    const vertexCount = builder.allocate('vertex-count', 'uint32', 1);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    const unresolvedCount = builder.allocate('unresolved-count', 'uint32', 1);
    builder.graph.add(
      new GPUTerrainStreamNetwork({
        grid: terrain.grid,
        flowDirections: flow.flowDirections.view,
        streams: flow.streams.view,
        output: {
          sourceCells: sourceCells.view,
          targetCells: targetCells.view,
          vertexOffsets: vertexOffsets.view,
          positions: positions.view,
          lengths: lengths.view,
          reachCount: reachCount.view,
          vertexCount: vertexCount.view,
          status: {overflow: overflow.view},
          unresolvedCount: unresolvedCount.view
        }
      })
    );
    return {
      outputs: {
        filledElevation: flow.filledElevation,
        positions,
        vertexOffsets,
        reachCount,
        lengths
      },
      update: values => {
        flow.writeStreamThreshold(values.streamThreshold);
        // Rows past the vertex count are never written; keep them out of the picture.
        positions.write(new Float32Array(VERTEX_CAPACITY * 2).fill(Number.NaN));
      }
    };
  },
  async afterRun({outputs, upload}) {
    // Edges between consecutive vertices of a reach, and the reach each vertex belongs to.
    const [offsetBytes, countBytes] = await Promise.all([
      outputs.vertexOffsets.buffer.readAsync(),
      outputs.reachCount.buffer.readAsync()
    ]);
    const offsets = new Uint32Array(
      offsetBytes.buffer,
      offsetBytes.byteOffset,
      offsetBytes.byteLength >> 2
    );
    const reachCount = Math.min(
      new Uint32Array(countBytes.buffer, countBytes.byteOffset, 1)[0],
      REACH_CAPACITY
    );
    const edges: number[] = [];
    const reachOfVertex = new Uint32Array(VERTEX_CAPACITY);
    for (let reach = 0; reach < reachCount; reach++) {
      for (let vertex = offsets[reach]; vertex < offsets[reach + 1]; vertex++) {
        reachOfVertex[vertex] = reach;
        if (vertex + 1 < offsets[reach + 1]) edges.push(vertex, vertex + 1);
      }
    }
    return {
      edges: upload('reach-edges', 'uint32x2', Uint32Array.from(edges.length > 0 ? edges : [0, 0])),
      reachOfVertex: upload('reach-of-vertex', 'uint32', reachOfVertex)
    };
  },
  // The pointer picks the nearest reach: it is outlined from its first to its last vertex, with
  // its ends marked, and the readout says what it joins.
  interaction: {
    kind: 'probe',
    hint: 'hover a reach',
    rest: [0.5, 0.55],
    resolve(pointer, {outputs, inputs, peek}) {
      const positions = peek(outputs.positions) as Float32Array | null;
      const offsets = peek(outputs.vertexOffsets) as Uint32Array | null;
      const counts = peek(outputs.reachCount) as Uint32Array | null;
      const lengths = peek(outputs.lengths) as Float32Array | null;
      if (!positions || !offsets || !counts || !lengths) return {};
      const reachCount = Math.min(counts[0], REACH_CAPACITY);
      let best = -1;
      let bestDistance = inputs.cellSize * 8;
      for (let reach = 0; reach < reachCount; reach++) {
        const hit = findNearestOnPolyline(
          positions,
          pointer.x,
          pointer.y,
          offsets[reach],
          offsets[reach + 1] - offsets[reach]
        );
        if (hit && hit.distance < bestDistance) {
          bestDistance = hit.distance;
          best = reach;
        }
      }
      if (best < 0) return {readout: `${reachCount} reaches`};
      const first = offsets[best];
      const last = offsets[best + 1] - 1;
      const start = [positions[first * 2], positions[first * 2 + 1]] as const;
      const end = [positions[last * 2], positions[last * 2 + 1]] as const;
      // A reach starts at a confluence when another reach ends in the same cell, and ends at one
      // when another reach starts there; otherwise it is a headwater or reaches the outlet.
      let startsAtConfluence = false;
      let endsAtConfluence = false;
      for (let other = 0; other < reachCount; other++) {
        if (other === best) continue;
        const otherLast = offsets[other + 1] - 1;
        if (
          Math.hypot(positions[otherLast * 2] - start[0], positions[otherLast * 2 + 1] - start[1]) <
          inputs.cellSize * 0.5
        ) {
          startsAtConfluence = true;
        }
        if (
          Math.hypot(
            positions[offsets[other] * 2] - end[0],
            positions[offsets[other] * 2 + 1] - end[1]
          ) <
          inputs.cellSize * 0.5
        ) {
          endsAtConfluence = true;
        }
      }
      const glyphs: RunGlyph[] = [
        {
          shape: 'polyline',
          points: positions.subarray(first * 2, (last + 1) * 2)
        },
        {shape: 'dot', at: start},
        {shape: 'dot', at: end}
      ];
      return {
        glyphs,
        readout: `reach ${best + 1}/${reachCount} · length ${lengths[best].toFixed(2)} · ${startsAtConfluence ? 'confluence' : 'headwater'} to ${endsAtConfluence ? 'confluence' : 'outlet'}`
      };
    }
  },
  draw: ({inputs, outputs, extra}) => [
    getTerrainContextLayer(inputs, outputs.filledElevation),
    {
      layer: 'segments',
      nodes: outputs.positions,
      edges: extra.edges,
      values: extra.reachOfVertex,
      radius: 1.8
    }
  ]
});
