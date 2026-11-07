// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMinimumBounds: for each polygon, the minimum rotated rectangle, the bounding circle and the
// longest line, all drawn from the output rows the operation wrote.

import {GPUMinimumBounds} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createPolygonSet, uploadPolygonSet, writePolygonSet} from './shared/polygon-set';
import {addCircleSegments, addQuadSegments} from './shared/shape-segments';
import {parameters} from './minimum-bounds.parameters';

const OPTIONS = {irregularity: 0.3, vertexCount: 10, fill: 0.62};

export default defineOperationRun({
  parameters,
  caption: 'Rotated rectangle, bounding circle and longest line of each polygon.',
  createInputs: ({seed, world}) => ({
    seed,
    world,
    set: createPolygonSet(makeRandom(seed), world, {...OPTIONS, stretch: 0.7})
  }),
  build(builder, inputs, values) {
    const polygons = uploadPolygonSet(builder, inputs.set);
    const count = inputs.set.count;
    const rectangleCorners = builder.allocate('rectangle-corners', 'float32x2', count * 4);
    const circles = builder.allocate('circles', 'float32x3', count);
    const longestLines = builder.allocate('longest-lines', 'float32x4', count);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPUMinimumBounds({
        positions: polygons.positions.view,
        featureOffsets: polygons.featureOffsets.view,
        polygonOffsets: polygons.polygonOffsets.view,
        ringOffsets: polygons.ringOffsets.view,
        maximumHullVertices: values.maximumHullVertices,
        output: {
          rectangleCorners: rectangleCorners.view,
          circles: circles.view,
          longestLines: longestLines.view,
          status: {overflow: overflow.view}
        }
      })
    );
    return {
      outputs: {
        ...polygons,
        longestLines,
        rectangleSegments: addQuadSegments(builder, 'rectangle', rectangleCorners),
        circleSegments: addCircleSegments(builder, 'circle', circles)
      },
      update: current =>
        writePolygonSet(
          polygons,
          createPolygonSet(makeRandom(inputs.seed), inputs.world, {
            ...OPTIONS,
            stretch: current.stretch
          })
        )
    };
  },
  rebuildOn: ['maximumHullVertices'],
  draw: ({inputs, outputs, theme}) => [
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.set.outlineCount},
    {layer: 'lines', segments: outputs.rectangleSegments, color: [1, 0.6, 0.2, 1]},
    {layer: 'lines', segments: outputs.circleSegments, color: [0.4, 0.75, 1, 1]},
    {layer: 'lines', segments: outputs.longestLines, color: [...theme.ink, 1], radius: 1.8}
  ]
});
