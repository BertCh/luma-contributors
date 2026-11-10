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
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {bindAxes} from '../../interaction';
import {createTerrainCloud} from './_lib/helpers';
import {parameters} from './pointcloud-local-positions.parameters';

const POINT_COUNT = 1500;
const LATTICE_CAPACITY = 256;
const MINIMUM_SPACING = 0.012;

function createInputs({
  world,
  random
}: {
  world: {width: number; height: number};
  random: () => number;
}) {
  return {
    cloud: createTerrainCloud(random, world.width, POINT_COUNT),
    width: world.width,
    result: {latticeCount: 0}
  };
}
type Inputs = ReturnType<typeof createInputs>;
type Outputs = Record<'naive' | 'aligned' | 'lattice', RunBuffer>;

/** The spacing of float32 values near `magnitude`, in the units of `magnitude`. */
function getFloatStep(magnitude: number): number {
  return 2 ** (Math.floor(Math.log2(magnitude)) - 23);
}

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Plain float32 cast (orange) against local-origin positions (blue); the faint lattice is the float32 grid the plain cast snaps to.',
  createInputs,
  build(builder, inputs) {
    const naive = builder.allocate('naive', 'float32x2', POINT_COUNT);
    const aligned = builder.allocate('local', 'float32x2', POINT_COUNT);
    const lattice = builder.allocate('lattice', 'float32x4', LATTICE_CAPACITY);
    return {
      outputs: {naive, aligned, lattice},
      update: current => {
        const cloud = inputs.cloud;
        const extent = current.extent;
        const absolute = new Float64Array(POINT_COUNT * 3);
        for (let index = 0; index < POINT_COUNT; index++) {
          // Move and scale the survey to compare float32 precision at different coordinate magnitudes.
          absolute.set(
            [
              current.originDistance * 1e6 + cloud.positions[index * 3] * extent,
              current.originDistance * 1e6 * 0.9 + cloud.positions[index * 3 + 1] * extent,
              cloud.positions[index * 3 + 2] * extent
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
        const naiveValues = new Float32Array(POINT_COUNT * 2);
        const alignedValues = new Float32Array(POINT_COUNT * 2);
        for (let index = 0; index < POINT_COUNT; index++) {
          for (const component of [0, 1]) {
            // Naive: round the absolute value to float32, then subtract the origin in double precision.
            naiveValues[index * 2 + component] =
              (Math.fround(absolute[index * 3 + component]) -
                origin[component] -
                minimum[component]) *
              scale;
            alignedValues[index * 2 + component] =
              (local[index * 3 + component] - minimum[component]) * scale;
          }
        }
        naive.write(naiveValues);
        aligned.write(alignedValues);
        // The float32 grid of the absolute x and y: where the plain cast can place a point.
        const segments = new Float32Array(LATTICE_CAPACITY * 4);
        let count = 0;
        for (const component of [0, 1]) {
          const step = getFloatStep(current.originDistance * 1e6 * (component === 0 ? 1 : 0.9));
          const spacing = step * scale;
          const length = component === 0 ? inputs.width : 1;
          if (spacing < MINIMUM_SPACING) continue;
          const first = Math.ceil((origin[component] + minimum[component]) / step) * step;
          for (let line = first; count < LATTICE_CAPACITY; line += step) {
            const position = (line - origin[component] - minimum[component]) * scale;
            if (position > length) break;
            segments.set(
              component === 0 ? [position, 0, position, 1] : [0, position, inputs.width, position],
              count * 4
            );
            count++;
          }
        }
        lattice.write(segments);
        inputs.result = {latticeCount: count};
      }
    };
  },
  // The pointer is a pad on the two things that set the float32 error: x is the coordinate
  // magnitude (logarithmic), y the extent of the surveyed patch.
  interaction: bindAxes<Inputs, Outputs>(parameters, {
    kind: 'pad',
    hint: 'magnitude across, extent up',
    x: {parameter: 'originDistance', scale: 'logarithmic'},
    y: 'extent',
    decorate: values => {
      const step = getFloatStep(values.originDistance * 1e6);
      return {
        readout: `${values.originDistance.toFixed(1)} Mm · float32 step ${step.toPrecision(2)} m · plain cast off by up to ${(step / 2).toPrecision(2)} m`
      };
    }
  }),
  draw: ({inputs, outputs}) => [
    {
      layer: 'lines',
      context: true,
      segments: outputs.lattice,
      count: Math.max(1, inputs.result.latticeCount),
      color: [1, 0.6, 0.2, 0.3]
    },
    {layer: 'dots', positions: outputs.naive, color: [1, 0.6, 0.2, 0.9], radius: 4.5},
    {layer: 'dots', positions: outputs.aligned, color: [0.3, 0.75, 1, 0.9], radius: 2}
  ]
});
