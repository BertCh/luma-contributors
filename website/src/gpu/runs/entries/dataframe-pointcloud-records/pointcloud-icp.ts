// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointCloudICP registering a rotated and shifted copy of a terrain cloud onto the original.
// Gray dots are the target, orange dots are the source after the transform the GPU solved; the
// curve below is the RMSE per iteration from the history output.

import {GPUPointCloudICP} from '@luma-contributors/analysis/pointcloud-analysis';
import {defineOperationRun} from '../../define-run';
import {addFloatCopy} from './_lib/convert';
import {createTerrainCloud} from './_lib/helpers';
import {addTransformedFootprint} from './_lib/transform-points';
import {parameters} from './pointcloud-icp.parameters';

const TERRAIN_COUNT = 9000;
const POINT_COUNT = 1500;

export default defineOperationRun({
  parameters,
  caption: 'Source cloud after ICP (orange) over the target (gray); RMSE per iteration below.',
  createInputs: ({random, world}) => {
    const cloud = createTerrainCloud(random, world.width, TERRAIN_COUNT);
    const target = new Float32Array(POINT_COUNT * 3);
    for (let index = 0; index < POINT_COUNT; index++) {
      target.set(cloud.positions.subarray(index * 6 * 3, index * 6 * 3 + 3), index * 3);
    }
    return {width: world.width, target};
  },
  build(builder, inputs, values) {
    const target = builder.upload('target', 'float32x3', inputs.target);
    // The source is the target turned about the cloud center and shifted: ICP has to undo it.
    const angle = (values.rotation * Math.PI) / 180;
    const centerX = inputs.width / 2;
    const source = builder.upload(
      'source',
      'float32x3',
      Float32Array.from({length: POINT_COUNT * 3}, (_, index) => {
        const row = Math.floor(index / 3);
        const x = inputs.target[row * 3] - centerX;
        const y = inputs.target[row * 3 + 1] - 0.5;
        if (index % 3 === 0) return centerX + x * Math.cos(angle) - y * Math.sin(angle) + 0.03;
        if (index % 3 === 1) return 0.5 + x * Math.sin(angle) + y * Math.cos(angle) - 0.02;
        return inputs.target[index];
      })
    );
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
    return {
      outputs: {
        target: addTransformedFootprint(
          builder,
          'target-footprint',
          target,
          identityTransform(builder)
        ),
        aligned: addTransformedFootprint(builder, 'aligned-footprint', source, transform),
        rmse: addFloatCopy(builder, 'rmse', history, values.iterations, {componentStart: 1})
      }
    };
  },
  rebuildOn: ['iterations', 'maximumDistance', 'rotation'],
  draw: ({outputs}) => [
    {
      layer: 'dots',
      context: true,
      positions: outputs.target,
      color: [0.6, 0.65, 0.75, 0.7],
      radius: 3.4
    },
    {layer: 'dots', positions: outputs.aligned, color: [1, 0.6, 0.2, 0.9], radius: 2},
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.rmse,
      range: 'auto',
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
