// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMinimumClearance: how far each polygon's vertices can move before it becomes invalid. The
// white segment is the witness line the operation wrote; polygons are filled by their clearance.

import {GPUMinimumClearance} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createPolygonSet, uploadPolygonSet, writePolygonSet} from './shared/polygon-set';
import {parameters} from './minimum-clearance.parameters';

export default defineOperationRun({
  parameters,
  caption: 'The tightest spot of each polygon, joined by its witness line.',
  createInputs: ({seed, world}) => ({
    seed,
    world,
    set: createPolygonSet(makeRandom(seed), world, {irregularity: 0.5})
  }),
  build(builder, inputs) {
    const polygons = uploadPolygonSet(builder, inputs.set);
    const clearances = builder.allocate('clearances', 'float32', inputs.set.count);
    const lines = builder.allocate('lines', 'float32x4', inputs.set.count);
    builder.graph.add(
      new GPUMinimumClearance({
        geometry: {
          kind: 'polygons',
          positions: polygons.positions.view,
          featureOffsets: polygons.featureOffsets.view,
          polygonOffsets: polygons.polygonOffsets.view,
          ringOffsets: polygons.ringOffsets.view
        },
        output: {clearances: clearances.view, lines: lines.view}
      })
    );
    return {
      outputs: {...polygons, clearances, lines},
      update: values =>
        writePolygonSet(
          polygons,
          createPolygonSet(makeRandom(inputs.seed), inputs.world, {
            irregularity: values.irregularity
          })
        )
    };
  },
  draw: ({inputs, outputs, theme}) => [
    {
      layer: 'polygons',
      positions: outputs.fillVertices,
      indices: outputs.fillIndices,
      owners: outputs.fillOwners,
      values: outputs.clearances,
      triangleCount: inputs.set.fillTriangleCount
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.set.outlineCount},
    {layer: 'lines', segments: outputs.lines, color: [...theme.ink, 1], radius: 2.4}
  ]
});
