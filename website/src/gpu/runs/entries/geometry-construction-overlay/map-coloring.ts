// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMapColoring assigns every tract a color so that no two touching tracts share one, using as few
// colors as it can find. The fill is the color index the operation wrote; the seed reshuffles the
// choice when the priority is 'random'.

import {
  getGPUMapColoringParameterValues,
  GPUMapColoring
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts, uploadSpatialWeights} from '../../inputs';
import type {RunGlyph} from '../../interaction';
import {findPolygonAt} from '../../pick';
import {parameters} from './map-coloring.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Tracts colored so that no two neighbors share a color.',
  createInputs: ({random, world}) => createTracts(random, world, 9),
  build(builder, tracts, values) {
    const weights = uploadSpatialWeights(builder, tracts.contiguity);
    const parameterBuffer = builder.allocate('seed', 'uint32', 1);
    const colors = builder.allocate('colors', 'uint32', tracts.count);
    const colorCount = builder.allocate('color-count', 'uint32', 1);
    builder.graph.add(
      new GPUMapColoring({
        weights,
        parameters: parameterBuffer.view,
        priority: values.priority === 0 ? 'largest-degree-first' : 'random',
        output: {colors: colors.view, colorCount: colorCount.view}
      })
    );
    return {
      outputs: {
        colors,
        vertices: builder.upload('vertices', 'float32x2', tracts.vertices),
        indices: builder.upload('indices', 'uint32', tracts.triangleIndices),
        owners: builder.upload('owners', 'uint32', tracts.triangleOwners),
        outline: builder.upload('outline', 'float32x4', tracts.outline)
      },
      update: current =>
        parameterBuffer.write(getGPUMapColoringParameterValues({seed: current.seed}))
    };
  },
  rebuildOn: ['priority'],
  // The pointer picks a tract; its rook neighbors are outlined, and none of them carries its color.
  interaction: {
    kind: 'probe',
    hint: 'hover a tract',
    resolve(pointer, {inputs, outputs, peek}) {
      const row = findPolygonAt(
        inputs.vertices,
        inputs.triangleIndices,
        inputs.triangleOwners,
        pointer.x,
        pointer.y,
        inputs.triangleCount
      );
      if (row < 0) return {};
      const getRing = (tract: number) =>
        Array.from(
          inputs.vertices.subarray(inputs.ringOffsets[tract] * 2, inputs.ringOffsets[tract + 1] * 2)
        );
      const glyphs: RunGlyph[] = [{shape: 'polyline', points: getRing(row), closed: true}];
      const colors = peek(outputs.colors);
      let neighborCount = 0;
      let sharedCount = 0;
      for (
        let slot = inputs.contiguity.offsets[row];
        slot < inputs.contiguity.offsets[row + 1];
        slot++
      ) {
        const neighbor = inputs.contiguity.neighbors[slot];
        neighborCount++;
        if (colors && colors[neighbor] === colors[row]) sharedCount++;
        glyphs.push({shape: 'polyline', points: getRing(neighbor), closed: true, dashed: true});
      }
      return {
        focus: row,
        glyphs,
        readout: `tract ${row}${colors ? ` · color ${colors[row]}` : ''} · ${neighborCount} neighbors, ${sharedCount} share it`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.colors,
      triangleCount: inputs.triangleCount,
      alpha: 0.9
    },
    {
      layer: 'lines',
      segments: outputs.outline,
      color: [0.05, 0.07, 0.1, 0.9],
      radius: 1.4,
      count: inputs.outlineCount
    }
  ]
});
