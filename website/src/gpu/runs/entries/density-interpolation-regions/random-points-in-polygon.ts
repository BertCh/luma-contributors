// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURandomPointsInPolygon draws `counts[i]` uniform points inside polygon i; the points are
// colored by the polygon that produced them.

import {
  getGPUDotDensityParameterValues,
  GPURandomPointsInPolygon
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts} from '../../inputs';
import {parameters} from './random-points-in-polygon.parameters';
import {getPackedPolygons, uploadPackedPolygons, uploadTractMesh} from './support/geometry';

const CAPACITY = 18000;

export default defineOperationRun({
  parameters,
  caption: 'The same number of random points inside every polygon; color is the source polygon.',
  createInputs: ({random, world}) => createTracts(random, world, 6, 0.2),
  build(builder, tracts, values) {
    const counts = builder.upload(
      'counts',
      'uint32',
      new Uint32Array(tracts.count).fill(values.pointsPerPolygon)
    );
    const parameterBuffer = builder.allocate('parameters', 'uint32', 8);
    parameterBuffer.write(getGPUDotDensityParameterValues({seed: values.seed}));
    const samples = builder.allocate('samples', 'float32x2', CAPACITY);
    const ids = builder.allocate('ids', 'uint32', CAPACITY);
    const count = builder.allocate('count', 'uint32', 1);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPURandomPointsInPolygon({
        ...uploadPackedPolygons(builder, getPackedPolygons(tracts)),
        counts: counts.view,
        parameters: parameterBuffer.view,
        output: {
          positions: samples.view,
          dots: {ids: ids.view, status: {count: count.view, overflow: overflow.view}}
        }
      })
    );
    return {outputs: {samples, ids, ...uploadTractMesh(builder, tracts)}};
  },
  draw: ({inputs, outputs}) => [
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.outlineCount},
    {layer: 'dots', positions: outputs.samples, values: outputs.ids, radius: 2.2}
  ]
});
