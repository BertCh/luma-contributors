// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCellPyramid aggregates points once at the finest resolution and rolls the result up through
// every coarser level. All four level tables exist after one run; the slider picks the one drawn.

import {GPUCellPyramid} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {getHonshuEvents, HONSHU_WINDOW} from '../../../data/point-observations';
import {
  addCellFill,
  addPointerCell,
  addPointKeys,
  allocateCellTable,
  getCellFamily,
  getCellTableViews
} from './support/cells';
import {findKeyRows, getWindowCenter, readPointerCell} from './support/pointer';
import {parameters} from './cell-pyramid.parameters';

const LEVEL_COUNT = 4;
const LEVEL_CAPACITY = 2048;

export default defineOperationRun({
  parameters,
  geographicWindow: HONSHU_WINDOW,
  caption: 'One aggregation pass, four zoom levels: finest first.',
  createInputs: ({random}) => ({
    ...getHonshuEvents(random, 900),
    pointer: [...getWindowCenter(HONSHU_WINDOW)] as [number, number]
  }),
  build(builder, points, values) {
    const family = getCellFamily(values['family']);
    const resolutions = family === 'h3' ? [6, 5, 4, 3] : [11, 10, 9, 8];
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const pointValues = builder.upload('values', 'float32', points.values);
    const keys = addPointKeys(builder, family, resolutions[0], positions);
    const levels = resolutions.map((_, level) =>
      allocateCellTable(builder, `level-${level}`, LEVEL_CAPACITY)
    );
    const levelCounts = builder.allocate('level-counts', 'uint32', LEVEL_COUNT);
    builder.graph.add(
      new GPUCellPyramid({
        family,
        cells: keys.view,
        values: pointValues.view,
        sumScale: 65536,
        resolutions,
        output: {levels: levels.map(getCellTableViews), levelCounts: levelCounts.view}
      })
    );
    const pointer = builder.allocate('pointer', 'float32x2', 1);
    return {
      outputs: {
        levels,
        // The pointer keyed at every level of the pyramid by the same GPUPointToCell.
        pointerCells: resolutions.map((resolution, level) =>
          addPointerCell(builder, family, resolution, pointer, `pointer-${level}`)
        ),
        levelFills: levels.map((level, index) =>
          addCellFill(builder, family, level.cells, `fill-${index}`)
        )
      },
      update: () => pointer.write(Float32Array.from(points.pointer))
    };
  },
  rebuildOn: ['family'],
  // The cell under the pointer at all four levels, nested, with each level's sum from its table.
  interaction: {
    kind: 'probe',
    hint: 'hover a cell',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, peek}) {
      inputs.pointer = [pointer.x, pointer.y];
      const glyphs: RunGlyph[] = [];
      const sums: string[] = [];
      outputs.pointerCells.forEach((pointerCell, level) => {
        const cell = readPointerCell(peek, pointerCell);
        if (cell.outline) glyphs.push(cell.outline);
        const table = outputs.levels[level];
        const keys = peek(table.cells);
        const count = peek(table.count);
        const tableSums = peek(table.sumValues);
        const overflow = peek(table.overflow);
        const candidateOverflow = peek(table.candidateOverflow);
        let sum = '…';
        if (overflow?.[0] || candidateOverflow?.[0]) sum = 'incomplete';
        else if (cell.keys && keys && count && tableSums && overflow && candidateOverflow) {
          const [row] = findKeyRows(keys, count[0], cell.keys);
          sum = row === undefined ? '0' : tableSums[row].toFixed(0);
        }
        sums.push(`L${level} ${sum}`);
      });
      return {rerun: true, glyphs, readout: `sum ${sums.join(' · ')}`};
    }
  },
  draw: ({outputs, values}) => {
    const fill = outputs.levelFills[values['level']];
    return [
      {
        layer: 'polygons',
        positions: fill.boundaries,
        indices: fill.indices,
        owners: fill.owners,
        values: outputs.levels[values['level']].sumValues,
        triangleCount: fill.triangleCount,
        range: [0, 8 * Math.pow(3, values['level'])]
      }
    ];
  }
});
