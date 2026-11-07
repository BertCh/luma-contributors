// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMaximumInscribedCircle: the largest circle inside each polygon (shells with holes). The circle
// outlines are generated from the circles the operation wrote; dots are the touching witnesses.

import {GPUMaximumInscribedCircle} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createPolygonSet, uploadPolygonSet, writePolygonSet} from './shared/polygon-set';
import {addCircleSegments} from './shared/shape-segments';
import {parameters} from './maximum-inscribed-circle.parameters';

export default defineOperationRun({
  parameters,
  caption: 'The biggest circle that still fits in each polygon, and the boundary point it touches.',
  createInputs: ({seed, world}) => ({
    seed,
    world,
    set: createPolygonSet(makeRandom(seed), world, {irregularity: 0.2, bend: 0.6})
  }),
  build(builder, inputs, values) {
    const polygons = uploadPolygonSet(builder, inputs.set);
    const circles = builder.allocate('circles', 'float32x4', inputs.set.count);
    const witnesses = builder.allocate('witnesses', 'float32x2', inputs.set.count);
    const status = builder.allocate('status', 'uint32', inputs.set.count);
    builder.graph.add(
      new GPUMaximumInscribedCircle({
        positions: polygons.positions.view,
        featureOffsets: polygons.featureOffsets.view,
        polygonOffsets: polygons.polygonOffsets.view,
        ringOffsets: polygons.ringOffsets.view,
        tolerance: values.tolerance,
        output: {circles: circles.view, witnesses: witnesses.view, status: status.view}
      })
    );
    const circleSegments = addCircleSegments(builder, 'circle', circles);
    return {
      outputs: {...polygons, witnesses, circleSegments},
      update: current =>
        writePolygonSet(
          polygons,
          createPolygonSet(makeRandom(inputs.seed), inputs.world, {
            irregularity: 0.2,
            bend: current.bend
          })
        )
    };
  },
  rebuildOn: ['tolerance'],
  draw: ({inputs, outputs}) => [
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.set.outlineCount},
    {layer: 'lines', segments: outputs.circleSegments, radius: 1.8},
    {layer: 'dots', positions: outputs.witnesses, radius: 4, color: [1, 0.6, 0.2, 1]}
  ]
});
