// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMinimumClearance: how far each polygon's vertices can move before it becomes invalid. The
// white segment is the witness line the operation wrote; polygons are filled by their clearance.

import {GPUMinimumClearance} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {
  applyVertexDrag,
  createPolygonSet,
  uploadPolygonSet,
  writePolygonSet
} from './shared/polygon-set';
import {carryNearestVertex, createDragFields, getHeroRest} from './shared/pointer-support';
import {parameters} from './minimum-clearance.parameters';

const OPTIONS = {rows: 2, columns: 3};

export default defineOperationRun({
  parameters,
  caption: 'The tightest spot of each polygon, joined by its witness line.',
  createInputs: ({seed, world}) => {
    const set = createPolygonSet(makeRandom(seed), world, {...OPTIONS, irregularity: 0.5});
    return {seed, world, set, ...createDragFields(set, 1)};
  },
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
      update: values => {
        const next = createPolygonSet(makeRandom(inputs.seed), inputs.world, {
          ...OPTIONS,
          irregularity: values.irregularity
        });
        applyVertexDrag(next, inputs.drag);
        inputs.set = next;
        writePolygonSet(polygons, next);
      }
    };
  },
  // The pointer carries a vertex toward a neighboring edge; the clearance and its witness follow.
  interaction: {
    kind: 'drag',
    hint: 'squeeze a vertex',
    rest: ({inputs, toUnit}) => toUnit(getHeroRest(inputs)),
    resolve(pointer, {inputs, outputs, peek}) {
      const carried = carryNearestVertex(inputs, pointer, 0, 0.3);
      const feature = inputs.drag?.feature ?? 0;
      const clearances = peek(outputs.clearances);
      const witnesses = peek(outputs.lines);
      return {
        rerun: true,
        glyphs: [
          ...carried.glyphs,
          ...(witnesses
            ? [
                {
                  shape: 'line' as const,
                  from: [witnesses[feature * 4], witnesses[feature * 4 + 1]] as const,
                  to: [witnesses[feature * 4 + 2], witnesses[feature * 4 + 3]] as const
                }
              ]
            : [])
        ],
        readout: `clearance ${clearances ? clearances[feature].toFixed(3) : '...'}`
      };
    }
  },
  draw: ({inputs, outputs, theme}) => [
    {
      layer: 'polygons',
      positions: outputs.fillVertices,
      indices: outputs.fillIndices,
      owners: outputs.fillOwners,
      values: outputs.clearances,
      triangleCount: inputs.set.fillTriangleCount,
      range: [0, 0.08]
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.set.outlineCount},
    {layer: 'lines', segments: outputs.lines, color: [...theme.ink, 1], radius: 3.2}
  ]
});
