// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUBufferSelection keeps the points within `distance` of a river (a chain of segments). Selected
// points are bright, the rest muted; drag the distance and the selection follows with no recompile.

import {createRiverLine} from '../../../data/hydrography';
import {GPUBufferSelection} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findParameter, snapToParameter, type RunGlyph} from '../../interaction';
import {findNearestOnPolyline} from '../../pick';
import {createRandomPoints} from './support/join';
import {parameters} from './buffer-selection.parameters';

const POINT_COUNT = 900;
const SEGMENT_COUNT = 40;
const DISTANCE_PARAMETER = findParameter(parameters, 'distance');

/** The river at a given bend: the default bend leaves the projected Missouri reach unchanged. */
function bendRiver(reach: number[][], bend: number): number[][] {
  return reach.map(([x, y]) => [x, 0.5 + (y - 0.5) * (bend / 0.18)]);
}

/** The river's two edges of a corridor `distance` wide, as open polylines. */
function getCorridorEdges(vertices: number[][], distance: number): number[][] {
  return [-1, 1].map(side =>
    vertices.flatMap(([x, y], index) => {
      const before = vertices[Math.max(0, index - 1)];
      const after = vertices[Math.min(vertices.length - 1, index + 1)];
      const length = Math.hypot(after[0] - before[0], after[1] - before[1]) || 1;
      return [
        x - (side * distance * (after[1] - before[1])) / length,
        y + (side * distance * (after[0] - before[0])) / length
      ];
    })
  );
}

export default defineOperationRun({
  parameters,
  caption: 'Points within a distance of the river stay selected.',
  createInputs: ({random, world}) => ({
    points: createRandomPoints(random, world, POINT_COUNT),
    reach: createRiverLine(0, SEGMENT_COUNT + 1, world)
  }),
  build(builder, inputs) {
    const points = builder.upload('points', 'float32x2', inputs.points);
    const starts = builder.allocate('river-starts', 'float32x2', SEGMENT_COUNT);
    const ends = builder.allocate('river-ends', 'float32x2', SEGMENT_COUNT);
    const river = builder.allocate('river', 'float32x4', SEGMENT_COUNT);
    const distance = builder.allocate('distance', 'float32', 1);
    const outputMask = builder.allocate('output-mask', 'uint32', POINT_COUNT);
    const distances = builder.allocate('distances', 'float32', POINT_COUNT);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPUBufferSelection({
        points: {kind: 'points', positions: points.view},
        features: {kind: 'segments', starts: starts.view, ends: ends.view},
        distance: distance.view,
        candidateCapacity: 1 << 16,
        output: {
          outputMask: outputMask.view,
          distances: distances.view,
          status: {overflow: overflow.view}
        }
      })
    );
    return {
      outputs: {points, river, outputMask},
      update: current => {
        // At the default bend the projected Missouri reach is unchanged.
        const vertices = bendRiver(inputs.reach, current.bend);
        starts.write(Float32Array.from(vertices.slice(0, -1).flat()));
        ends.write(Float32Array.from(vertices.slice(1).flat()));
        river.write(
          Float32Array.from(
            vertices.slice(0, -1).flatMap((vertex, index) => [...vertex, ...vertices[index + 1]])
          )
        );
        distance.write(Float32Array.of(current.distance));
      }
    };
  },
  // Reach: the distance from the pointer to the nearest river segment is the selection distance.
  interaction: {
    kind: 'reach',
    hint: 'measure from the river',
    rest: ({toUnit, inputs, values}) => {
      const vertices = bendRiver(inputs.reach, values.bend);
      const middle = SEGMENT_COUNT / 2;
      const [x0, y0] = vertices[middle];
      const [x1, y1] = vertices[middle + 1];
      const length = Math.hypot(x1 - x0, y1 - y0);
      const normal = [-(y1 - y0) / length, (x1 - x0) / length];
      const middleX = (x0 + x1) / 2;
      const middleY = (y0 + y1) / 2;
      // Bisect along the normal for the point whose nearest river point is the default distance away.
      let near = 0;
      let far = 0.5;
      for (let step = 0; step < 30; step++) {
        const reach = (near + far) / 2;
        const hit = findNearestOnPolyline(
          vertices.flat(),
          middleX + normal[0] * reach,
          middleY + normal[1] * reach
        );
        if ((hit?.distance ?? 0) < DISTANCE_PARAMETER.default) near = reach;
        else far = reach;
      }
      const reach = (near + far) / 2;
      return toUnit([middleX + normal[0] * reach, middleY + normal[1] * reach]);
    },
    resolve(pointer, {inputs, outputs, values, peek}) {
      const vertices = bendRiver(inputs.reach, values.bend);
      const hit = findNearestOnPolyline(vertices.flat(), pointer.x, pointer.y);
      if (!hit) return {};
      const distance = snapToParameter(DISTANCE_PARAMETER, hit.distance);
      const mask = peek(outputs.outputMask);
      let selected = 0;
      if (mask) for (const bit of mask) selected += bit > 0 ? 1 : 0;
      const glyphs: RunGlyph[] = [
        {shape: 'line', from: hit.point, to: [pointer.x, pointer.y]},
        {shape: 'dot', at: hit.point},
        ...getCorridorEdges(vertices, distance).map(
          (points): RunGlyph => ({shape: 'polyline', points, dashed: true})
        )
      ];
      return {
        values: {distance},
        glyphs,
        readout: `distance ${distance.toFixed(3)} · ${mask ? selected : '…'} of ${POINT_COUNT} selected`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'dots', positions: outputs.points, values: outputs.outputMask, radius: 3},
    {layer: 'lines', context: true, segments: outputs.river, radius: 2, color: [1, 1, 1, 0.9]}
  ]
});
