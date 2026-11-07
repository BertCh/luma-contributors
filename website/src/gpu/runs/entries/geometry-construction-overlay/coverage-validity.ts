// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCoverageValidity flags ring edges that break a polygon coverage: overlaps, crossings,
// mismatched shared boundaries and gaps narrower than `gapWidth`. Every ring edge is muted; the
// flagged edges are drawn thick in orange. The cells are shrunk by "Data: gap" (negative values
// overlap), so raising `gapWidth` above the real gap flags the strips between cells.

import {
  getGPUCoverageValidityParameterValues,
  GPUCoverageValidity
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addPackedSegments, HIGHLIGHT_COLOR} from './shared/lines';
import {getPolygonFields, PolygonBuilder, uploadPolygonSet} from './shared/polygons';
import {parameters} from './coverage-validity.parameters';

const COLUMNS = 4;
const ROWS = 3;

export default defineOperationRun({
  parameters,
  caption: 'Coverage edges flagged as invalid (orange): overlaps and gaps narrower than gapWidth.',
  createInputs: ({world}) => {
    const getSet = (gap: number) => {
      const builder = new PolygonBuilder();
      const margin = 0.06;
      const cellWidth = (world.width - 2 * margin) / COLUMNS;
      const cellHeight = (1 - 2 * margin) / ROWS;
      for (let row = 0; row < ROWS; row++) {
        for (let column = 0; column < COLUMNS; column++) {
          const left = margin + column * cellWidth + gap / 2;
          const right = margin + (column + 1) * cellWidth - gap / 2;
          const bottom = margin + row * cellHeight + gap / 2;
          const top = margin + (row + 1) * cellHeight - gap / 2;
          const middleX = (left + right) / 2;
          const middleY = (bottom + top) / 2;
          builder.addFeature([
            [
              [left, bottom],
              [middleX, bottom],
              [right, bottom],
              [right, middleY],
              [right, top],
              [middleX, top],
              [left, top],
              [left, middleY]
            ]
          ]);
        }
      }
      return builder.build();
    };
    return {getSet};
  },
  build(builder, inputs, values) {
    const initial = inputs.getSet(values.gap);
    const polygons = uploadPolygonSet(builder, 'coverage', initial);
    const vertexCount = initial.positions.length / 2;
    const segmentFlags = builder.allocate('segment-flags', 'uint32', vertexCount);
    const invalidCount = builder.allocate('invalid-count', 'uint32', 1);
    const isValid = builder.allocate('is-valid', 'uint32', 1);
    const parameterBuffer = builder.allocate('gap-width', 'float32', 4);
    builder.graph.add(
      new GPUCoverageValidity({
        ...getPolygonFields(polygons),
        parameters: parameterBuffer.view,
        output: {
          segmentFlags: segmentFlags.view,
          status: {invalidCount: invalidCount.view},
          isValid: isValid.view
        }
      })
    );
    const drawing = addPackedSegments(
      builder,
      'coverage',
      polygons.positions,
      polygons.ringOffsets,
      {
        closed: true,
        flags: segmentFlags
      }
    );
    return {
      outputs: {...drawing, flagged: drawing.segments},
      update: current => {
        polygons.positions.write(inputs.getSet(current.gap).positions);
        parameterBuffer.write(getGPUCoverageValidityParameterValues({gapWidth: current.gapWidth}));
      }
    };
  },
  draw: ({outputs}) => [
    {layer: 'segments', context: true, nodes: outputs.points, edges: outputs.edges, radius: 1.2},
    {layer: 'lines', segments: outputs.flagged, color: HIGHLIGHT_COLOR, radius: 3.5}
  ]
});
