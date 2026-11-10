// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUStudyWindowPreparation prepares one polygon as an analysis window and writes its summary
// (bounds, area, perimeter, edge count). The orange box is the summary bounds, and the bars are the
// summary area (x10) and perimeter, plotted straight from the output buffer.

import {GPUStudyWindowPreparation} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {
  applyVertexDrag,
  createPolygonSet,
  uploadPolygonSet,
  writePolygonSet
} from './shared/polygon-set';
import {carryNearestVertex, createDragFields, getHeroRest} from './shared/pointer-support';
import {addBoxSegmentsOfRows, addKernel} from './shared/shape-segments';
import {parameters} from './study-window-preparation.parameters';

const WINDOW_OPTIONS = {rows: 1, columns: 1, vertexCount: 16, hasHoles: true};

export default defineOperationRun({
  parameters,
  caption: 'The window polygon, its summary bounds, and its area and perimeter.',
  createInputs: ({seed, world}) => {
    const panel = {width: world.width * 0.55, height: 1};
    const set = createPolygonSet(makeRandom(seed), panel, {...WINDOW_OPTIONS, irregularity: 0.4});
    return {seed, panel, set, ...createDragFields(set, 0)};
  },
  build(builder, inputs) {
    const polygons = uploadPolygonSet(builder, inputs.set);
    const summary = builder.allocate('summary', 'float32', 8);
    builder.graph.add(
      new GPUStudyWindowPreparation({
        positions: polygons.positions.view,
        featureOffsets: polygons.featureOffsets.view,
        polygonOffsets: polygons.polygonOffsets.view,
        ringOffsets: polygons.ringOffsets.view,
        samplingGrid: [64, 64],
        // The window is rebuilt only on request; its polygon changes on every frame here.
        rebuildWhen: () => true,
        output: {summary: summary.view}
      })
    );
    // Bars for area (scaled by ten, to share an axis with the perimeter), perimeter and, as a reference,
    // the area of the summary bounds box (also scaled by ten): the gap to the first bar is the empty space.
    const measures = builder.allocate('measures', 'float32', 3);
    addKernel(builder, {
      id: 'summary-measures',
      invocationCount: 1,
      bindings: [
        {name: 'summary', buffer: summary, access: 'read'},
        {name: 'measures', buffer: measures, access: 'read_write'}
      ],
      body: `measures[measuresOffset] = summary[summaryOffset + 4u] * 10.0;
  measures[measuresOffset + 1u] = summary[summaryOffset + 5u];
  measures[measuresOffset + 2u] = (summary[summaryOffset + 2u] - summary[summaryOffset]) * (summary[summaryOffset + 3u] - summary[summaryOffset + 1u]) * 10.0;`
    });
    return {
      outputs: {
        ...polygons,
        measures,
        summary,
        boxSegments: addBoxSegmentsOfRows(builder, 'summary-bounds', summary, 1)
      },
      update: values => {
        // `fill` sizes the window around its cell center, so the radius slider scales the polygon.
        const next = createPolygonSet(makeRandom(inputs.seed), inputs.panel, {
          ...WINDOW_OPTIONS,
          irregularity: values.irregularity,
          fill: 0.78 * values.radius
        });
        applyVertexDrag(next, inputs.drag);
        inputs.set = next;
        writePolygonSet(polygons, next);
      }
    };
  },
  // The pointer carries a window vertex with its neighbors; the summary box and bars follow.
  interaction: {
    kind: 'drag',
    hint: 'drag the window outline',
    rest: ({inputs, toUnit}) => toUnit(getHeroRest(inputs)),
    resolve(pointer, {inputs, outputs, peek}) {
      const carried = carryNearestVertex(inputs, pointer, 1, 0.3);
      const summary = peek(outputs.summary);
      return {
        rerun: true,
        glyphs: carried.glyphs,
        readout: summary
          ? `area ${summary[4].toFixed(3)} · perimeter ${summary[5].toFixed(2)}`
          : 'area ... · perimeter ...'
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.set.outlineCount},
    {layer: 'lines', segments: outputs.boxSegments, color: [1, 0.6, 0.2, 1], radius: 2},
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.measures,
      rectangle: [0.62, 0.15, 0.95, 0.85],
      range: [0, 6]
    }
  ]
});
