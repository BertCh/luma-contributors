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
import {
  createTractGlyph,
  findMiddleTract,
  findTractAt,
  getPackedPolygons,
  uploadPackedPolygons,
  uploadTractMesh
} from './support/geometry';

const CAPACITY = 18000;
const BOOST = 3;

export default defineOperationRun({
  parameters,
  caption:
    'The same number of random points inside every polygon (the hovered one asks for three times as many); color is the source polygon.',
  createInputs: ({random, world}) => {
    const tracts = createTracts(random, world, 6, 0.2);
    return {tracts, world, focus: {row: findMiddleTract(tracts, world)}};
  },
  build(builder, {tracts, focus}, values) {
    const counts = builder.allocate('counts', 'uint32', tracts.count);
    const parameterBuffer = builder.allocate('parameters', 'uint32', 8);
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
    return {
      outputs: {samples, ids, ...uploadTractMesh(builder, tracts)},
      // counts and the seed are per-frame inputs: the hovered polygon asks for more points.
      update: current => {
        counts.write(
          new Uint32Array(tracts.count).map((_, row) =>
            row === focus.row ? current.pointsPerPolygon * BOOST : current.pointsPerPolygon
          )
        );
        parameterBuffer.write(getGPUDotDensityParameterValues({seed: current.seed}));
      }
    };
  },
  // The hovered tract draws BOOST times as many points; the others keep `counts`.
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
      const changed = row !== inputs.focus.row;
      inputs.focus.row = row;
      if (row < 0)
        return {rerun: changed, readout: `counts ${values.pointsPerPolygon} in every tract`};
      return {
        rerun: changed,
        glyphs: [createTractGlyph(inputs.tracts, row)],
        readout: `tract ${row} · counts ${values.pointsPerPolygon * BOOST} · others ${values.pointsPerPolygon}`
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
    {layer: 'dots', positions: outputs.samples, values: outputs.ids, radius: 4}
  ]
});
