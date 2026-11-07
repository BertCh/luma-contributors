// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMapColoring assigns every tract a color so that no two touching tracts share one, using as few
// colors as it can find. The fill is the color index the operation wrote; the seed reshuffles the
// choice when the priority is 'random'.

import {
  getGPUMapColoringParameterValues,
  GPUMapColoring
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts, uploadSpatialWeights} from '../../inputs';
import {parameters} from './map-coloring.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Tracts colored so that no two neighbors share a color.',
  createInputs: ({random, world}) => createTracts(random, world, 9),
  build(builder, tracts, values) {
    const weights = uploadSpatialWeights(builder, tracts.contiguity);
    const parameterBuffer = builder.allocate('seed', 'uint32', 1);
    const colors = builder.allocate('colors', 'uint32', tracts.count);
    const colorCount = builder.allocate('color-count', 'uint32', 1);
    builder.graph.add(
      new GPUMapColoring({
        weights,
        parameters: parameterBuffer.view,
        priority: values.priority === 0 ? 'largest-degree-first' : 'random',
        output: {colors: colors.view, colorCount: colorCount.view}
      })
    );
    return {
      outputs: {
        colors,
        vertices: builder.upload('vertices', 'float32x2', tracts.vertices),
        indices: builder.upload('indices', 'uint32', tracts.triangleIndices),
        owners: builder.upload('owners', 'uint32', tracts.triangleOwners),
        outline: builder.upload('outline', 'float32x4', tracts.outline)
      },
      update: current =>
        parameterBuffer.write(getGPUMapColoringParameterValues({seed: current.seed}))
    };
  },
  rebuildOn: ['priority'],
  draw: ({inputs, outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.colors,
      triangleCount: inputs.triangleCount,
      alpha: 0.9
    },
    {
      layer: 'lines',
      segments: outputs.outline,
      color: [0.05, 0.07, 0.1, 0.9],
      radius: 1.4,
      count: inputs.outlineCount
    }
  ]
});
