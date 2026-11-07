// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUStudyWindowPreparation prepares one polygon as an analysis window and writes its summary
// (bounds, area, perimeter, edge count). The orange box is the summary bounds, and the bars are the
// summary area (x10) and perimeter, plotted straight from the output buffer.

import {GPUStudyWindowPreparation} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createPolygonSet, uploadPolygonSet} from './shared/polygon-set';
import {addBoxSegmentsOfRows, addKernel} from './shared/shape-segments';
import {parameters} from './study-window-preparation.parameters';

const WINDOW_OPTIONS = {rows: 1, columns: 1, vertexCount: 16, hasHoles: true};

export default defineOperationRun({
  parameters,
  caption: 'The window polygon, its summary bounds, and its area and perimeter.',
  createInputs: ({seed, world}) => {
    const panel = {width: world.width * 0.55, height: 1};
    return {
      seed,
      panel,
      set: createPolygonSet(makeRandom(seed), panel, {...WINDOW_OPTIONS, irregularity: 0.4})
    };
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
        output: {summary: summary.view}
      })
    );
    // Bars for area (scaled by ten, to share an axis with the perimeter) and perimeter.
    const measures = builder.allocate('measures', 'float32', 2);
    addKernel(builder, {
      id: 'summary-measures',
      invocationCount: 1,
      bindings: [
        {name: 'summary', buffer: summary, access: 'read'},
        {name: 'measures', buffer: measures, access: 'read_write'}
      ],
      body: `measures[measuresOffset] = summary[summaryOffset + 4u] * 10.0;
  measures[measuresOffset + 1u] = summary[summaryOffset + 5u];`
    });
    return {
      outputs: {
        ...polygons,
        measures,
        boxSegments: addBoxSegmentsOfRows(builder, 'summary-bounds', summary, 1)
      },
      update: values => {
        const set = createPolygonSet(makeRandom(inputs.seed), inputs.panel, {
          ...WINDOW_OPTIONS,
          irregularity: values.irregularity
        });
        // Scale the polygon about its center to change the window's size.
        const center = [inputs.panel.width / 2, 0.5];
        for (let index = 0; index < set.positions.length; index += 2) {
          set.positions[index] = center[0] + (set.positions[index] - center[0]) * values.radius;
          set.positions[index + 1] =
            center[1] + (set.positions[index + 1] - center[1]) * values.radius;
        }
        polygons.positions.write(set.positions);
        polygons.outline.write(
          Float32Array.from({length: set.outline.length}, (_, component) => {
            const axis = component % 2;
            return center[axis] + (set.outline[component] - center[axis]) * values.radius;
          })
        );
      }
    };
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
