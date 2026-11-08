// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUDotDensity places dots inside each tract in proportion to two population categories; the
// dots (colored by category) are the compacted `positions` output over the tract outlines.

import {
  getGPUDotDensityParameterValues,
  GPUDotDensity
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts} from '../../inputs';
import {parameters} from './dot-density.parameters';
import {getPackedPolygons, uploadPackedPolygons, uploadTractMesh} from './support/geometry';

const CAPACITY = 18000;

export default defineOperationRun({
  parameters,
  caption: 'One dot per population unit, colored by category, placed inside each tract.',
  createInputs: ({random, world}) => createTracts(random, world, 8, 0.3),
  build(builder, tracts, values) {
    const categoryValues = new Float32Array(tracts.count * 2);
    for (let tract = 0; tract < tracts.count; tract++) {
      categoryValues[tract * 2] = tracts.values[tract] * 700;
      categoryValues[tract * 2 + 1] = (1 - tracts.values[tract]) * 400;
    }
    const parameterBuffer = builder.allocate('parameters', 'uint32', 8);
    parameterBuffer.write(
      getGPUDotDensityParameterValues({seed: values.seed, dotsPerUnit: values.dotsPerUnit})
    );
    const dotPositions = builder.allocate('dot-positions', 'float32x2', CAPACITY);
    const categories = builder.allocate('categories', 'uint32', CAPACITY);
    const ids = builder.allocate('ids', 'uint32', CAPACITY);
    const count = builder.allocate('count', 'uint32', 1);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPUDotDensity({
        ...uploadPackedPolygons(builder, getPackedPolygons(tracts)),
        values: builder.upload('values', 'float32', categoryValues).view,
        categoryCount: 2,
        parameters: parameterBuffer.view,
        output: {
          positions: dotPositions.view,
          categories: categories.view,
          dots: {ids: ids.view, status: {count: count.view, overflow: overflow.view}}
        }
      })
    );
    return {outputs: {dotPositions, categories, ...uploadTractMesh(builder, tracts)}};
  },
  draw: ({inputs, outputs, theme}) => [
    {
      layer: 'lines',
      context: true,
      segments: outputs.outline,
      count: inputs.outlineCount,
      radius: 1.6,
      color: [...theme.muted, 0.9]
    },
    {
      layer: 'dots',
      positions: outputs.dotPositions,
      values: outputs.categories,
      radius: 3.2
    }
  ]
});
