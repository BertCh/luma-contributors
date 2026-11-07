// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURegionMask writes a 0/1 mask of the points inside a polygon (here a star-shaped lasso with
// concave notches). The bright dots are the rows the operation marked.

import {GPURegionMask} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createUniformPoints} from '../../inputs';
import {parameters} from './region-mask.parameters';

const POINT_COUNT = 5;

export default defineOperationRun({
  parameters,
  caption: 'Points inside the lasso polygon, marked by the mask output.',
  createInputs: ({random, world}) => ({
    world,
    points: createUniformPoints(random, world, 3000, 0.2)
  }),
  build(builder, {world, points}) {
    const vertices = builder.allocate('vertices', 'float32x2', POINT_COUNT * 2);
    const mask = builder.allocate('mask', 'uint32', points.count);
    builder.graph.add(
      new GPURegionMask({
        positions: builder.upload('positions', 'float32x2', points.positions).view,
        region: {
          kind: 'polygon',
          vertices: vertices.view,
          vertexCount: builder.upload('vertex-count', 'uint32', Uint32Array.of(POINT_COUNT * 2))
            .view
        },
        output: {
          mask: mask.view,
          status: {overflow: builder.allocate('overflow', 'uint32', 1).view}
        }
      })
    );
    const outline = builder.allocate('outline', 'float32x4', POINT_COUNT * 2);
    return {
      outputs: {mask, positions: builder.upload('points', 'float32x2', points.positions), outline},
      update: values => {
        const ring = new Float32Array(POINT_COUNT * 4);
        for (let index = 0; index < POINT_COUNT * 2; index++) {
          const angle = values.rotation + (index / (POINT_COUNT * 2)) * Math.PI * 2;
          const radius = values.size * (index % 2 === 0 ? 1 : values.notch);
          ring.set(
            [0.5 * world.width + Math.cos(angle) * radius, 0.5 + Math.sin(angle) * radius],
            index * 2
          );
        }
        vertices.write(ring);
        const segments = new Float32Array(POINT_COUNT * 8);
        for (let index = 0; index < POINT_COUNT * 2; index++) {
          const next = (index + 1) % (POINT_COUNT * 2);
          segments.set(
            [ring[index * 2], ring[index * 2 + 1], ring[next * 2], ring[next * 2 + 1]],
            index * 4
          );
        }
        outline.write(segments);
      }
    };
  },
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.positions, radius: 1.6, alpha: 0.5},
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.mask,
      range: [0, 1],
      ramp: 'sequential',
      fade: true,
      radius: 3
    },
    {layer: 'lines', context: true, segments: outputs.outline, color: [1, 1, 1, 1]}
  ]
});
