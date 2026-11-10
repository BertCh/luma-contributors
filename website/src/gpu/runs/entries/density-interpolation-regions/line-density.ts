// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULineDensity accumulates segment length per grid cell (divided by cell area); the white lines
// are the input polylines, the raster is the `densities` output.

import {
  getGPULineDensityParameterValues,
  GPULineDensity
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findNearestSegment, getRasterCell} from '../../pick';
import {parameters} from './line-density.parameters';
import {createSpreadPolylines, flipVertical} from './support/geometry';

export default defineOperationRun({
  parameters,
  caption: 'Length of lines per cell area; the lines are the input.',
  createInputs: ({random, world}) => {
    const lines = createSpreadPolylines(random, world, 40, 14);
    // Fixed color range: eight times the mean density of the default lines, so that neither the
    // grid resolution nor the line count rescales the colors.
    const defaultSegments = lines.offsets[18] - 18;
    let length = 0;
    for (let segment = 0; segment < defaultSegments; segment++) {
      length += Math.hypot(
        lines.segments[segment * 4 + 2] - lines.segments[segment * 4],
        lines.segments[segment * 4 + 3] - lines.segments[segment * 4 + 1]
      );
    }
    return {width: world.width, lines, maximumDensity: (8 * length) / world.width};
  },
  build(builder, inputs, values) {
    const rows = values.rows;
    const columns = Math.round(inputs.width * rows);
    const {lines} = inputs;
    const lineCount = values.lineCount;
    const vertexEnd = lines.offsets[lineCount];
    const positions = builder.upload(
      'positions',
      'float32x2',
      flipVertical(lines.vertices.slice(0, vertexEnd * 2))
    );
    const lineOffsets = builder.upload(
      'line-offsets',
      'uint32',
      lines.offsets.slice(0, lineCount + 1)
    );
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    parameterBuffer.write(
      getGPULineDensityParameterValues({
        minX: 0,
        minY: 0,
        cellWidth: inputs.width / columns,
        cellHeight: 1 / rows
      })
    );
    const lengths = builder.allocate('lengths', 'float32', columns * rows);
    const densities = builder.allocate('densities', 'float32', columns * rows);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPULineDensity({
        positions: positions.view,
        lineOffsets: lineOffsets.view,
        columns,
        rows,
        parameters: parameterBuffer.view,
        output: {
          lengths: lengths.view,
          densities: densities.view,
          status: {overflow: overflow.view}
        }
      })
    );
    const segments = builder.upload('segments', 'float32x4', lines.segments);
    return {
      outputs: {
        densities,
        columns,
        rows,
        segments,
        segmentCount: vertexEnd - lineCount,
        // Pieces of the lines that cross the hovered cell; unused rows stay NaN.
        crossing: builder.upload(
          'crossing',
          'float32x4',
          new Float32Array((vertexEnd - lineCount) * 4).fill(NaN)
        )
      }
    };
  },
  // The hovered cell is outlined and the pieces of the lines inside it are lit: their total
  // length divided by the cell area is the density the operation wrote.
  interaction: {
    kind: 'probe',
    hint: 'hover a cell',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs}) {
      const bounds = [0, 0, inputs.width, 1] as const;
      // Snap to the cell of the nearest line, so the probe always lands on something measured.
      const hit = findNearestSegment(
        inputs.lines.segments,
        pointer.x,
        pointer.y,
        outputs.segmentCount
      );
      const cell = getRasterCell(
        outputs.columns,
        outputs.rows,
        bounds,
        hit?.point[0] ?? pointer.x,
        hit?.point[1] ?? pointer.y
      );
      const cellWidth = inputs.width / outputs.columns;
      const cellHeight = 1 / outputs.rows;
      const minimum: [number, number] = [cell.column * cellWidth, 1 - (cell.row + 1) * cellHeight];
      const maximum: [number, number] = [minimum[0] + cellWidth, minimum[1] + cellHeight];
      const pieces = new Float32Array(outputs.segmentCount * 4).fill(NaN);
      let count = 0;
      let length = 0;
      for (let segment = 0; segment < outputs.segmentCount; segment++) {
        const clipped = clipSegment(
          inputs.lines.segments.subarray(segment * 4, segment * 4 + 4),
          minimum,
          maximum
        );
        if (!clipped) continue;
        pieces.set(clipped, count * 4);
        length += Math.hypot(clipped[2] - clipped[0], clipped[3] - clipped[1]);
        count++;
      }
      outputs.crossing.write(pieces);
      return {
        focus: cell.index,
        glyphs: [{shape: 'box', minimum, maximum}],
        readout: `cell ${cell.column}, ${cell.row} · length ${length.toFixed(3)} / area ${(cellWidth * cellHeight).toFixed(4)} = ${(length / (cellWidth * cellHeight)).toFixed(1)}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      values: outputs.densities,
      columns: outputs.columns,
      rows: outputs.rows,
      range: [0, inputs.maximumDensity]
    },
    {layer: 'lines', context: true, segments: outputs.segments, count: outputs.segmentCount},
    {
      layer: 'lines',
      segments: outputs.crossing,
      count: outputs.segmentCount,
      radius: 2.4,
      color: [1, 0.9, 0.35, 1]
    }
  ]
});

/** Liang-Barsky clip of one segment (x0, y0, x1, y1) to a rectangle; null when it misses. */
function clipSegment(
  segment: Float32Array,
  minimum: readonly [number, number],
  maximum: readonly [number, number]
): [number, number, number, number] | null {
  const [x0, y0, x1, y1] = segment;
  let start = 0;
  let end = 1;
  const checks: [number, number][] = [
    [-(x1 - x0), x0 - minimum[0]],
    [x1 - x0, maximum[0] - x0],
    [-(y1 - y0), y0 - minimum[1]],
    [y1 - y0, maximum[1] - y0]
  ];
  for (const [direction, distance] of checks) {
    if (direction === 0) {
      if (distance < 0) return null;
      continue;
    }
    const ratio = distance / direction;
    if (direction < 0) start = Math.max(start, ratio);
    else end = Math.min(end, ratio);
    if (start > end) return null;
  }
  return [
    x0 + (x1 - x0) * start,
    y0 + (y1 - y0) * start,
    x0 + (x1 - x0) * end,
    y0 + (y1 - y0) * end
  ];
}
