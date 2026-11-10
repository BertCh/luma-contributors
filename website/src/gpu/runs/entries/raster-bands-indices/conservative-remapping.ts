// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterConservativeRemapping averages a fine raster onto a coarser, offset grid by exact
// overlap area: every destination cell takes the area-weighted mean of the source cells it covers
// (or their sum, for extensive quantities). Cells that hang over the source edge are only
// partly covered, which the normalization handles differently. The destination grid is a
// construction prop, so the run rebuilds when it changes. The destination is drawn in its true
// geometry (overhang included) with its grid lines; the pointer probes a destination cell and the
// source cells it overlaps.

import {GPURasterConservativeRemapping} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createElevationRaster} from '../../inputs';
import {formatNumber} from '../../interaction';
import {parameters} from './conservative-remapping.parameters';
import {drawRaster} from './support/bands';
import {getCellAt} from './support/gestures';

const NORMALIZATIONS = ['destination-area', 'covered-area', 'extensive'] as const;

/** Destination grid in source-cell units: cells of `factor` source cells, starting `shift` before. */
function getDestination(
  inputs: {columns: number; rows: number},
  values: Readonly<Record<string, number>>
) {
  const spacing = values.factor;
  const origin = -spacing * values.shift;
  return {
    spacing,
    origin,
    width: Math.ceil((inputs.columns - origin) / spacing),
    height: Math.ceil((inputs.rows - origin) / spacing)
  };
}

export default defineOperationRun({
  parameters,
  rebuildOn: ['factor', 'normalization', 'shift'],
  caption: 'A fine raster in; the area-weighted values of a coarse grid out.',
  createInputs: ({world}) => createElevationRaster(world, 60),
  build(builder, inputs, values) {
    const cellSize = inputs.cellSize;
    const destination = getDestination(inputs, values);
    const destinationSpacing = cellSize * destination.spacing;
    const origin = destination.origin * cellSize;
    const destinationCount = destination.width * destination.height;
    const source = builder.upload('source', 'float32', inputs.values);
    const result = builder.allocate('values', 'float32', destinationCount);
    const coveredArea = builder.allocate('covered-area', 'float32', destinationCount);
    const status = builder.allocate('status', 'uint32', destinationCount);
    builder.graph.add(
      new GPURasterConservativeRemapping({
        source: {
          width: inputs.columns,
          height: inputs.rows,
          origin: [0, 0],
          spacing: [cellSize, cellSize]
        },
        destination: {
          width: destination.width,
          height: destination.height,
          origin: [origin, origin],
          spacing: [destinationSpacing, destinationSpacing]
        },
        values: source.view,
        normalization: NORMALIZATIONS[Math.round(values.normalization)],
        output: {values: result.view, coveredArea: coveredArea.view, status: status.view}
      })
    );
    const lines: number[] = [];
    const cellWidth = inputs.cellSize;
    for (let column = 0; column <= destination.width; column++) {
      const x = (destination.origin + column * destination.spacing) * cellWidth;
      if (x >= 0 && x <= inputs.columns * cellWidth) lines.push(x, 0, x, inputs.rows * cellWidth);
    }
    for (let row = 0; row <= destination.height; row++) {
      const y = (destination.origin + row * destination.spacing) * cellWidth;
      if (y >= 0 && y <= inputs.rows * cellWidth) {
        lines.push(
          0,
          inputs.rows * cellWidth - y,
          inputs.columns * cellWidth,
          inputs.rows * cellWidth - y
        );
      }
    }
    return {
      outputs: {
        source,
        result,
        coveredArea,
        overlap: builder.allocate('overlap', 'float32', inputs.values.length),
        grid: builder.upload('destination-grid', 'float32x4', Float32Array.from(lines)),
        destination
      }
    };
  },
  // The destination values are spread back onto the source grid in the destination's true
  // geometry, so the coarse cells sit where the operation put them, overhang included.
  async afterRun({inputs, outputs, upload}) {
    const bytes = await outputs.result.buffer.readAsync();
    const result = new Float32Array(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
    );
    const {origin, spacing, width, height} = outputs.destination;
    const display = new Float32Array(inputs.values.length);
    for (let row = 0; row < inputs.rows; row++) {
      for (let column = 0; column < inputs.columns; column++) {
        const destinationColumn = Math.min(
          width - 1,
          Math.floor((column + 0.5 - origin) / spacing)
        );
        const destinationRow = Math.min(height - 1, Math.floor((row + 0.5 - origin) / spacing));
        display[row * inputs.columns + column] = result[destinationRow * width + destinationColumn];
      }
    }
    return {display: upload('display', 'float32', display)};
  },
  interaction: {
    kind: 'probe',
    hint: 'probe a coarse cell',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, outputs, world, peek}) => {
      const {origin, spacing, width, height} = outputs.destination;
      const cell = getCellAt(inputs, world, pointer);
      const destinationColumn = Math.min(
        width - 1,
        Math.floor((cell.column + 0.5 - origin) / spacing)
      );
      const destinationRow = Math.min(height - 1, Math.floor((cell.row + 0.5 - origin) / spacing));
      const left = origin + destinationColumn * spacing;
      const top = origin + destinationRow * spacing;
      const overlap = new Float32Array(inputs.values.length).fill(Number.NaN);
      let cells = 0;
      let covered = 0;
      for (
        let row = Math.max(0, Math.floor(top));
        row < Math.min(inputs.rows, Math.ceil(top + spacing));
        row++
      ) {
        for (
          let column = Math.max(0, Math.floor(left));
          column < Math.min(inputs.columns, Math.ceil(left + spacing));
          column++
        ) {
          const share =
            Math.max(0, Math.min(column + 1, left + spacing) - Math.max(column, left)) *
            Math.max(0, Math.min(row + 1, top + spacing) - Math.max(row, top));
          if (share <= 0) continue;
          overlap[row * inputs.columns + column] = share;
          covered += share;
          cells++;
        }
      }
      outputs.overlap.write(overlap);
      const result = peek(outputs.result);
      const value = result ? result[destinationRow * width + destinationColumn] : Number.NaN;
      const cellWidth = world.width / inputs.columns;
      const cellHeight = world.height / inputs.rows;
      return {
        glyphs: [
          {
            shape: 'box',
            minimum: [
              Math.max(0, left) * cellWidth,
              world.height - Math.min(inputs.rows, top + spacing) * cellHeight
            ],
            maximum: [
              Math.min(inputs.columns, left + spacing) * cellWidth,
              world.height - Math.max(0, top) * cellHeight
            ]
          }
        ],
        readout: `dest [${destinationColumn}, ${destinationRow}] | ${cells} cells | covered ${Math.round((100 * covered) / (spacing * spacing))}% | ${formatNumber(value)}`
      };
    }
  },
  draw: ({inputs, outputs, extra, values, theme}) => [
    drawRaster(inputs, outputs.source, {context: true, range: [0, 1]}),
    ...(extra
      ? [
          drawRaster(inputs, extra.display, {
            ramp: 'sequential',
            range: values.normalization === 2 ? [0, values.factor * values.factor] : [0, 1],
            alpha: 0.85
          })
        ]
      : []),
    {layer: 'lines', context: true, segments: outputs.grid, color: [...theme.ink, 0.45]},
    drawRaster(inputs, outputs.overlap, {ramp: 'sequential', range: [0, 1], fade: true})
  ]
});
