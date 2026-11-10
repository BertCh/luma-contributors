// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointCloudICP registering a rotated and shifted copy of a terrain cloud onto the original.
// Gray dots are the target, orange dots are the source after the transform the GPU solved; the
// curve below is the RMSE per iteration from the history output.

import {GPUPointCloudICP} from '@luma-contributors/analysis/pointcloud-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {bindAxes} from '../../interaction';
import {addFloatCopy} from './_lib/convert';
import {createTerrainCloud} from './_lib/helpers';
import {addTransformedFootprint} from './_lib/transform-points';
import {parameters} from './pointcloud-icp.parameters';

const TERRAIN_COUNT = 9000;
const POINT_COUNT = 1500;
// The misalignment's shift is along this unit direction; the `shift` parameter is its length.
const SHIFT_DIRECTION = [0.832, -0.555] as const;

function createInputs({
  random,
  world
}: {
  random: () => number;
  world: {width: number; height: number};
}) {
  const cloud = createTerrainCloud(random, world.width, TERRAIN_COUNT);
  const target = new Float32Array(POINT_COUNT * 3);
  for (let index = 0; index < POINT_COUNT; index++) {
    target.set(cloud.positions.subarray(index * 6 * 3, index * 6 * 3 + 3), index * 3);
  }
  // The first RMSE of the history is at most this: the root mean square offset of the source.
  return {width: world.width, target, rmseTop: 0.05};
}
type Inputs = ReturnType<typeof createInputs>;
type Outputs = Record<'target' | 'initial' | 'aligned' | 'rmse', RunBuffer>;

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'A misaligned copy of the King County scan (faint orange) after ICP (orange) over the original (gray); the curve is the RMSE per iteration.',
  createInputs,
  build(builder, inputs, values) {
    const target = builder.upload('target', 'float32x3', inputs.target);
    // The source is the target turned about the cloud center and shifted: ICP has to undo it.
    const source = builder.upload('source', 'float32x3', new Float32Array(POINT_COUNT * 3));
    const transform = builder.allocate('transform', 'float32', 16);
    const history = builder.allocate('history', 'float32x3', values.iterations);
    builder.graph.add(
      new GPUPointCloudICP({
        source: source.view,
        target: target.view,
        maximumDistance: values.maximumDistance,
        iterations: values.iterations,
        transform: transform.view,
        history: history.view,
        inlierMask: builder.allocate('inlier-mask', 'uint32', POINT_COUNT).view,
        status: builder.allocate('status', 'uint32', 1).view
      })
    );
    const identity = identityTransform(builder);
    return {
      outputs: {
        target: addTransformedFootprint(builder, 'target-footprint', target, identity),
        initial: addTransformedFootprint(builder, 'initial-footprint', source, identity),
        aligned: addTransformedFootprint(builder, 'aligned-footprint', source, transform),
        rmse: addFloatCopy(builder, 'rmse', history, values.iterations, {componentStart: 1})
      },
      update: current => {
        const angle = (current.rotation * Math.PI) / 180;
        const centerX = inputs.width / 2;
        let sumOfSquares = 0;
        const moved = Float32Array.from({length: POINT_COUNT * 3}, (_, index) => {
          const row = Math.floor(index / 3);
          const x = inputs.target[row * 3] - centerX;
          const y = inputs.target[row * 3 + 1] - 0.5;
          if (index % 3 === 0) {
            return (
              centerX +
              x * Math.cos(angle) -
              y * Math.sin(angle) +
              current.shift * SHIFT_DIRECTION[0]
            );
          }
          if (index % 3 === 1) {
            return (
              0.5 + x * Math.sin(angle) + y * Math.cos(angle) + current.shift * SHIFT_DIRECTION[1]
            );
          }
          return inputs.target[index];
        });
        for (let row = 0; row < POINT_COUNT; row++) {
          sumOfSquares +=
            (moved[row * 3] - inputs.target[row * 3]) ** 2 +
            (moved[row * 3 + 1] - inputs.target[row * 3 + 1]) ** 2;
        }
        inputs.rmseTop = Math.max(Math.sqrt(sumOfSquares / POINT_COUNT), 0.01);
        source.write(moved);
      }
    };
  },
  rebuildOn: ['iterations', 'maximumDistance'],
  // The pointer sets the misalignment ICP has to undo: x is the rotation of the source about the
  // cloud center, y its shift. An arc at the center marks the rotation.
  interaction: bindAxes<Inputs, Outputs>(parameters, {
    kind: 'pad',
    hint: 'rotate across, shift up',
    x: 'rotation',
    y: 'shift',
    decorate: (values, _pointer, {inputs, outputs, peek}) => {
      const centerX = inputs.width / 2;
      const arc: number[] = [];
      const sweep = (values.rotation * Math.PI) / 180;
      for (let step = 0; step <= 12; step++) {
        const angle = (step / 12) * sweep;
        arc.push(centerX + Math.cos(angle) * 0.12, 0.5 + Math.sin(angle) * 0.12);
      }
      const history = peek(outputs.rmse);
      return {
        glyphs: [
          {shape: 'crosshair', at: [centerX, 0.5], size: 6},
          {shape: 'polyline', points: arc}
        ],
        readout: `rotation ${values.rotation}° · shift ${values.shift.toFixed(3)}${
          history
            ? ` · RMSE ${history[0].toFixed(3)} -> ${history[history.length - 1].toFixed(3)}`
            : ''
        }`
      };
    }
  }),
  draw: ({inputs, outputs}) => [
    {
      layer: 'dots',
      context: true,
      positions: outputs.target,
      color: [0.6, 0.65, 0.75, 0.7],
      radius: 3.4
    },
    {
      layer: 'dots',
      context: true,
      positions: outputs.initial,
      color: [1, 0.6, 0.2, 0.7],
      radius: 2
    },
    {layer: 'dots', positions: outputs.aligned, color: [1, 0.6, 0.2, 0.9], radius: 2},
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.rmse,
      range: [0, inputs.rmseTop],
      rectangle: [0.62, 0.06, 0.97, 0.3],
      radius: 2.4,
      color: [0.3, 0.75, 1, 1]
    }
  ]
});

function identityTransform(builder: Parameters<typeof addTransformedFootprint>[0]) {
  return builder.upload(
    'identity',
    'float32',
    Float32Array.from({length: 16}, (_, index) => (index % 5 === 0 ? 1 : 0))
  );
}

// Bound to GPUPointCloudICP's `iterations` and `maximumDistance` (compile-time), and the rotation
// and shift that misalign the source cloud.
