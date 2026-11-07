// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// makePointCloudLocalPositions on a small survey patch at large projected coordinates. Orange dots
// are what a plain float32 cast of the absolute coordinates gives (it snaps to the float32 grid,
// about half a meter at 4.5 Mm); blue dots are the local-origin float32 positions, which keep the
// patch shape. This is a CPU helper: its output is uploaded and drawn.

import {
  getPointCloudLocalOrigin,
  makePointCloudLocalPositions
} from '@luma-contributors/analysis/pointcloud-analysis';
import {defineOperationRun} from '../../define-run';
import {drawNormal} from './_lib/helpers';
import {parameters} from './pointcloud-local-positions.parameters';

const POINT_COUNT = 1500;

export default defineOperationRun({
  parameters,
  caption: 'Plain float32 cast (orange) against local-origin positions (blue).',
  createInputs: ({random}) => ({
    uniform: Float32Array.from({length: POINT_COUNT * 2}, () => random()),
    normals: Float32Array.from({length: POINT_COUNT}, () => drawNormal(random))
  }),
  build(builder, inputs, values) {
    const extent = values.extent;
    const absolute = new Float64Array(POINT_COUNT * 3);
    for (let index = 0; index < POINT_COUNT; index++) {
      // A gently curved ridge so quantized positions visibly distort it.
      const x = inputs.uniform[index * 2] * extent * 1.78;
      const y = inputs.uniform[index * 2 + 1] * extent;
      absolute.set(
        [
          values.originDistance * 1e6 + x,
          values.originDistance * 1e6 * 0.9 + y,
          100 + inputs.normals[index] * 0.05
        ],
        index * 3
      );
    }
    const origin = getPointCloudLocalOrigin(absolute);
    const local = makePointCloudLocalPositions(absolute, origin);
    const scale = 1 / extent;
    const minimum = [0, 1].map(component =>
      Math.min(...Array.from({length: POINT_COUNT}, (_, index) => local[index * 3 + component]))
    );
    const naive = new Float32Array(POINT_COUNT * 2);
    const aligned = new Float32Array(POINT_COUNT * 2);
    for (let index = 0; index < POINT_COUNT; index++) {
      for (const component of [0, 1]) {
        // Naive: round the absolute value to float32, then subtract the origin in double precision.
        naive[index * 2 + component] =
          (Math.fround(absolute[index * 3 + component]) - origin[component] - minimum[component]) *
          scale;
        aligned[index * 2 + component] =
          (local[index * 3 + component] - minimum[component]) * scale;
      }
    }
    return {
      outputs: {
        naive: builder.upload('naive', 'float32x2', naive),
        aligned: builder.upload('local', 'float32x2', aligned)
      }
    };
  },
  draw: ({outputs}) => [
    {layer: 'dots', positions: outputs.naive, color: [1, 0.6, 0.2, 0.9], radius: 4.5},
    {layer: 'dots', positions: outputs.aligned, color: [0.3, 0.75, 1, 0.9], radius: 2}
  ]
});
