// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMaximumInscribedCircle: the largest circle inside each polygon (shells with holes). The circle
// outlines are generated from the circles the operation wrote; dots are the touching witnesses.

import {GPUMaximumInscribedCircle} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {
  applyVertexDrag,
  createPolygonSet,
  uploadPolygonSet,
  writePolygonSet
} from './shared/polygon-set';
import {carryNearestVertex, createDragFields, getHeroRest} from './shared/pointer-support';
import {addCircleSegments} from './shared/shape-segments';
import {parameters} from './maximum-inscribed-circle.parameters';

const OPTIONS = {irregularity: 0.2, rows: 2, columns: 4};

export default defineOperationRun({
  parameters,
  caption: 'The biggest circle that still fits in each polygon, and the boundary point it touches.',
  createInputs: ({seed, world}) => {
    const set = createPolygonSet(makeRandom(seed), world, {...OPTIONS, bend: 0.6});
    return {seed, world, set, ...createDragFields(set, 2)};
  },
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
      outputs: {...polygons, witnesses, circles, circleSegments},
      update: current => {
        const next = createPolygonSet(makeRandom(inputs.seed), inputs.world, {
          ...OPTIONS,
          bend: current.bend
        });
        applyVertexDrag(next, inputs.drag);
        inputs.set = next;
        writePolygonSet(polygons, next);
      }
    };
  },
  // The pointer pushes the nearest boundary vertex; the circle shrinks or jumps to another lobe.
  interaction: {
    kind: 'drag',
    hint: 'push the boundary',
    rest: ({inputs, toUnit}) => toUnit(getHeroRest(inputs)),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const carried = carryNearestVertex(inputs, pointer, 1, 0.3);
      const circles = peek(outputs.circles);
      const feature = inputs.drag?.feature ?? 0;
      const radius = circles?.[feature * 4 + 2];
      return {
        rerun: true,
        glyphs: carried.glyphs,
        readout: `radius ${radius === undefined ? '...' : radius.toFixed(3)} · tolerance ${values.tolerance.toFixed(3)}`
      };
    }
  },
  rebuildOn: ['tolerance'],
  draw: ({inputs, outputs}) => [
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.set.outlineCount},
    {layer: 'lines', segments: outputs.circleSegments, radius: 2.8},
    {layer: 'dots', positions: outputs.witnesses, radius: 6, color: [1, 0.6, 0.2, 1]}
  ]
});
