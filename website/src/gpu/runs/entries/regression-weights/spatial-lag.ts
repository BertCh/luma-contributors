// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialLag over rook-contiguous tracts. The noisy tract values (dots) are replaced by the weighted sum of each tract's neighbors; with `normalize` that is the
// neighbor average, which smooths the map.

import {GPUSpatialLag} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts, uploadSpatialWeights} from '../../inputs';
import {parameters} from './spatial-lag.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Tract values replaced by the average of their neighbors.',
  createInputs: ({random, world}) => createTracts(random, world),
  build(builder, tracts, values) {
    const weights = uploadSpatialWeights(builder, tracts.contiguity);
    const noisy = builder.allocate('values', 'float32', tracts.count);
    const lag = builder.allocate('lag', 'float32', tracts.count);
    builder.graph.add(
      new GPUSpatialLag({
        values: noisy.view,
        weights,
        normalize: values.normalize === 1,
        output: lag.view
      })
    );
    return {
      outputs: {
        noisy,
        lag,
        vertices: builder.upload('vertices', 'float32x2', tracts.vertices),
        indices: builder.upload('indices', 'uint32', tracts.triangleIndices),
        owners: builder.upload('owners', 'uint32', tracts.triangleOwners),
        centers: builder.upload('centers', 'float32x2', tracts.centers),
        outline: builder.upload('outline', 'float32x4', tracts.outline)
      },
      update: current => {
        const scaled = Float32Array.from(tracts.values, (value, index) => {
          const jitter = Math.sin(index * 12.9898) * 43758.5453;
          return value + (jitter - Math.floor(jitter) - 0.5) * current.noise * 0.6;
        });
        noisy.write(scaled);
      }
    };
  },
  rebuildOn: ['normalize'],
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.lag,
      triangleCount: inputs.triangleCount,
      range: [0, values.normalize ? 1.2 : 6]
    },
    {
      layer: 'lines',
      context: true,
      segments: outputs.outline,
      count: inputs.outlineCount
    },
    // The input values at the tract centers, on the same scale as the lag behind them.
    {
      layer: 'dots',
      positions: outputs.centers,
      values: outputs.noisy,
      range: [0, values.normalize ? 1.2 : 6],
      radius: 4
    }
  ]
});
