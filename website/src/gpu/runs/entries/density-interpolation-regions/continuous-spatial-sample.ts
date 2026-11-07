// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUContinuousSpatialSample lays a randomly shifted square lattice over an area frame (here a
// set of tracts with a lake cut out), keeping the lattice sites that fall inside the frame.

import {GPUContinuousSpatialSample} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts} from '../../inputs';
import {parameters} from './continuous-spatial-sample.parameters';

const CAPACITY = 4000;

export default defineOperationRun({
  parameters,
  caption: 'Sites of a random-shift lattice inside the frame; the lake is outside the frame.',
  createInputs: ({world}) => {
    const tracts = createTracts(() => 0.5, world, 6, 0);
    // Frame triangles (center, corner, next corner) of tracts away from the lake.
    const frame: number[] = [];
    const outline: number[] = [];
    for (let triangle = 0; triangle < tracts.triangleCount; triangle++) {
      const tract = tracts.triangleOwners[triangle];
      const x = tracts.centers[tract * 2];
      const y = tracts.centers[tract * 2 + 1];
      if (Math.hypot(x - 0.5 * world.width, y - 0.5) < 0.25) continue;
      for (let corner = 0; corner < 3; corner++) {
        const vertex = tracts.triangleIndices[triangle * 3 + corner];
        frame.push(tracts.vertices[vertex * 2], tracts.vertices[vertex * 2 + 1]);
      }
      outline.push(...tracts.outline.slice(triangle * 4, triangle * 4 + 4));
    }
    let [minX, minY, maxX, maxY] = [Infinity, Infinity, -Infinity, -Infinity];
    for (let index = 0; index < frame.length; index += 2) {
      minX = Math.min(minX, frame[index]);
      maxX = Math.max(maxX, frame[index]);
      minY = Math.min(minY, frame[index + 1]);
      maxY = Math.max(maxY, frame[index + 1]);
    }
    return {
      frame: Float32Array.from(frame),
      outline: Float32Array.from(outline),
      bounds: [minX - 0.01, minY - 0.01, maxX + 0.01, maxY + 0.01] as const
    };
  },
  build(builder, {frame, outline, bounds}, values) {
    // Unwritten slots stay NaN so only the sites the operation wrote are drawn.
    const sites = builder.upload('sites', 'float32x2', new Float32Array(CAPACITY * 2).fill(NaN));
    builder.graph.add(
      new GPUContinuousSpatialSample({
        frame: {type: 'area', positions: builder.upload('frame', 'float32x2', frame).view, bounds},
        spacing: values.spacing,
        seed: values.seed,
        capacity: CAPACITY,
        output: {
          positions: sites.view,
          designWeights: builder.allocate('design-weights', 'float32', CAPACITY).view,
          provenance: builder.allocate('provenance', 'uint32x4', CAPACITY).view,
          status: builder.allocate('status', 'uint32x4', 1).view
        }
      })
    );
    return {
      outputs: {
        sites,
        outline: builder.upload('outline', 'float32x4', outline),
        outlineCount: outline.length / 4
      }
    };
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.outline, count: outputs.outlineCount},
    {layer: 'dots', positions: outputs.sites, radius: 3.5, color: [1, 0.85, 0.3, 1]}
  ]
});
