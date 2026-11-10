// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCellAggregation bins points into a sorted cell table (count, sum, minimum and maximum per
// cell). The table's cell keys are decoded with GPUCellGeometry only to draw them.

import {GPUCellAggregation} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {getHonshuEvents, HONSHU_WINDOW} from '../../../data/point-observations';
import {
  addCellFill,
  addPointerCell,
  addPointKeys,
  allocateCellTable,
  drawPointerCell,
  getCellFamily,
  getCellResolution,
  getCellTableViews
} from './support/cells';
import {findKeyRows, readPointerCell, getWindowCenter} from './support/pointer';
import {parameters} from './cell-aggregation.parameters';

export default defineOperationRun({
  parameters,
  geographicWindow: HONSHU_WINDOW,
  caption: 'Points binned into cells: one table row per occupied cell.',
  createInputs: ({random}) => ({
    ...getHonshuEvents(random, 900),
    pointer: [...getWindowCenter(HONSHU_WINDOW)] as [number, number]
  }),
  build(builder, points, values) {
    const family = getCellFamily(values.family);
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const pointValues = builder.upload('values', 'float32', points.values);
    const table = allocateCellTable(builder, 'table', 1024);
    const resolution = getCellResolution(family, values.resolution);
    const keys = addPointKeys(builder, family, resolution, positions);
    builder.graph.add(
      new GPUCellAggregation({
        family,
        resolution,
        cells: keys.view,
        values: pointValues.view,
        sumScale: 65536,
        output: getCellTableViews(table)
      })
    );
    const pointer = builder.allocate('pointer', 'float32x2', 1);
    return {
      outputs: {
        table,
        positions,
        keys,
        pointerCell: addPointerCell(builder, family, resolution, pointer),
        fill: addCellFill(builder, family, table.cells)
      },
      update: () => pointer.write(Float32Array.from(points.pointer))
    };
  },
  rebuildOn: ['family', 'resolution'],
  // The pointer is a probe: the table row of the cell under the cursor, read back from the GPU.
  interaction: {
    kind: 'probe',
    hint: 'hover a cell',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, peek}) {
      inputs.pointer = [pointer.x, pointer.y];
      const cell = readPointerCell(peek, outputs.pointerCell);
      const tableKeys = peek(outputs.table.cells);
      const tableCount = peek(outputs.table.count);
      const counts = peek(outputs.table.counts);
      const sums = peek(outputs.table.sumValues);
      const maximums = peek(outputs.table.maximums);
      const pointKeys = peek(outputs.keys);
      const glyphs: RunGlyph[] = cell.outline ? [cell.outline] : [];
      let readout = `cell ${cell.key} · empty`;
      if (cell.keys && tableKeys && tableCount && counts && sums && maximums) {
        const [row] = findKeyRows(tableKeys, tableCount[0], cell.keys);
        if (row !== undefined) {
          readout = `count ${counts[row]} · sum ${sums[row].toFixed(1)} · max ${maximums[row].toFixed(1)}`;
          if (pointKeys) {
            for (const member of findKeyRows(pointKeys, inputs.count, cell.keys).slice(0, 30)) {
              glyphs.push({
                shape: 'dot',
                at: [inputs.positions[member * 2], inputs.positions[member * 2 + 1]]
              });
            }
          }
        }
      }
      return {rerun: true, glyphs, readout};
    }
  },
  draw: ({outputs, values}) => [
    {layer: 'dots', context: true, positions: outputs.positions, radius: 1.4},
    {
      layer: 'polygons',
      positions: outputs.fill.boundaries,
      indices: outputs.fill.indices,
      owners: outputs.fill.owners,
      values: [outputs.table.sumValues, outputs.table.minimums, outputs.table.maximums][
        values.measure
      ],
      triangleCount: outputs.fill.triangleCount,
      ramp: 'sequential',
      range: values.measure === 0 ? [0, 80] : [4, 7]
    },
    drawPointerCell(outputs.pointerCell, [1, 0.82, 0.3, 0.3])
  ]
});
