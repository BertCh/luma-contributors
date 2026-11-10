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
import {
  createTractGlyph,
  findMiddleTract,
  findTractAt,
  getPackedPolygons,
  uploadPackedPolygons,
  uploadTractMesh
} from './support/geometry';

const CAPACITY = 18000;

export default defineOperationRun({
  parameters,
  caption: 'Chicago residents aged 65+ and under 65, sampled inside each Census tract.',
  createInputs: ({random, world}) => {
    const tracts = createTracts(random, world, 8, 0.3);
    return {tracts, world, focus: {row: findMiddleTract(tracts, world)}};
  },
  build(builder, {tracts}, values) {
    // One input unit represents ten residents. At the default density, one dot represents 100 residents.
    const categoryValues = new Float32Array(tracts.count * 2);
    for (let tract = 0; tract < tracts.count; tract++) {
      categoryValues[tract * 2] = tracts.age65[tract] / 10;
      categoryValues[tract * 2 + 1] = (tracts.population[tract] - tracts.age65[tract]) / 10;
    }
    const parameterBuffer = builder.allocate('parameters', 'uint32', 8);
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
    return {
      outputs: {dotPositions, categories, ...uploadTractMesh(builder, tracts)},
      // dotsPerUnit and the seed are per-frame parameters.
      update: current =>
        parameterBuffer.write(
          getGPUDotDensityParameterValues({seed: current.seed, dotsPerUnit: current.dotsPerUnit})
        )
    };
  },
  // The hovered tract is outlined and its two categories are read out.
  interaction: {
    kind: 'probe',
    hint: 'hover a tract',
    rest: ({inputs, toUnit}) =>
      toUnit([
        inputs.tracts.centers[inputs.focus.row * 2],
        inputs.tracts.centers[inputs.focus.row * 2 + 1]
      ]),
    resolve(pointer, {inputs, values}) {
      const row = findTractAt(inputs.tracts, pointer.x, pointer.y);
      inputs.focus.row = row < 0 ? inputs.focus.row : row;
      const {tracts} = inputs;
      const residentsPerDot = Math.round(10 / values.dotsPerUnit);
      return {
        focus: inputs.focus.row,
        glyphs: [createTractGlyph(tracts, inputs.focus.row)],
        readout: `65+ ${Math.round(tracts.age65[inputs.focus.row]).toLocaleString('en-US')} · under 65 ${Math.round(tracts.population[inputs.focus.row] - tracts.age65[inputs.focus.row]).toLocaleString('en-US')} · 1 dot = ${residentsPerDot}`
      };
    }
  },
  draw: ({inputs, outputs, theme}) => [
    {
      layer: 'lines',
      context: true,
      segments: outputs.outline,
      count: inputs.tracts.outlineCount,
      radius: 1.6,
      color: [...theme.muted, 0.9]
    },
    {
      layer: 'dots',
      positions: outputs.dotPositions,
      values: outputs.categories,
      radius: 2.6
    }
  ]
});
