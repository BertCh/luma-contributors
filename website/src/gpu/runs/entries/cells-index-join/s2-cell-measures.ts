// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUS2CellMeasures writes the spherical area (steradians) and perimeter (radians) of S2 cells.
// Honshu earthquake epicenters are keyed at one S2 level; the cells are filled by the measure the operation
// wrote (the colors stretch over the small variation across the window).

import {GPUS2CellMeasures} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {getHonshuEvents, HONSHU_WINDOW} from '../../../data/point-observations';
import {addCellFill, addPointerCell, addPointKeys, drawPointerCell} from './support/cells';
import {getWindowCenter, readPointerCell} from './support/pointer';
import {addKernel} from './support/kernels';
import {parameters} from './s2-cell-measures.parameters';

const EARTH_RADIUS = 6371.0088;

function getMeasureRange(measure: number, level: number): [number, number] {
  // The six faces share the sphere: an average cell of a level covers 4π / (6 · 4^level) steradians.
  // Cells in this window are 0.7 to 0.8 of that area (S2 cells shrink toward the face edges) and
  // their perimeters 0.9 to 0.93 of a square of the average area.
  const area = (4 * Math.PI) / (6 * 4 ** level);
  return measure === 0
    ? [area * 0.6, area * 0.9]
    : [4 * Math.sqrt(area) * 0.85, 4 * Math.sqrt(area) * 1.0];
}

export default defineOperationRun({
  parameters,
  geographicWindow: HONSHU_WINDOW,
  caption:
    'Spherical area or perimeter of S2 cells containing Honshu epicenters, against the average cell of the level.',
  createInputs: ({random}) => ({
    ...getHonshuEvents(random, 700),
    pointer: [...getWindowCenter(HONSHU_WINDOW)] as [number, number]
  }),
  build(builder, grid, values) {
    const positions = builder.upload('positions', 'float32x2', grid.positions);
    const cells = addPointKeys(builder, 's2', values.level, positions);
    const measures = builder.allocate('measures', 'float32x2', grid.count);
    const status = builder.allocate('measure-status', 'uint32', grid.count);
    builder.graph.add(
      new GPUS2CellMeasures({
        cells: cells.view,
        output: {measures: measures.view, status: status.view}
      })
    );
    // Extract the selected component so auto-ranging only sees that measure's units.
    const selected = builder.allocate('selected-measure', 'float32', grid.count);
    addKernel(builder, {
      id: 'select-cell-measure',
      invocationCount: grid.count,
      bindings: [
        {name: 'measures', buffer: measures, access: 'read'},
        {name: 'selected', buffer: selected, access: 'read_write'}
      ],
      body: `selected[selectedOffset + index] = measures[measuresOffset + 2u * index + ${values.measure}u];`
    });
    // The same operation on the one cell under the pointer.
    const pointer = builder.allocate('pointer', 'float32x2', 1);
    const pointerCell = addPointerCell(builder, 's2', values.level, pointer);
    const pointerMeasures = builder.allocate('pointer-measures', 'float32x2', 1);
    builder.graph.add(
      new GPUS2CellMeasures({
        id: 'pointer-s2-measures',
        cells: pointerCell.cells.view,
        output: {
          measures: pointerMeasures.view,
          status: builder.allocate('pointer-measure-status', 'uint32', 1).view
        }
      })
    );
    return {
      outputs: {
        selected,
        pointerCell,
        pointerMeasures,
        fill: addCellFill(builder, 's2', cells, 'fill')
      },
      update: () => pointer.write(Float32Array.from(grid.pointer))
    };
  },
  rebuildOn: ['level', 'measure'],
  interaction: {
    kind: 'probe',
    hint: 'hover a cell',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, peek}) {
      inputs.pointer = [pointer.x, pointer.y];
      const cell = readPointerCell(peek, outputs.pointerCell);
      const measures = peek(outputs.pointerMeasures);
      return {
        rerun: true,
        glyphs: cell.outline ? [cell.outline] : [],
        readout: measures
          ? `area ${(measures[0] * EARTH_RADIUS ** 2).toFixed(0)} km² · perimeter ${measures[1].toFixed(4)} rad`
          : 'area …'
      };
    }
  },
  draw: ({outputs, values}) => [
    {
      layer: 'polygons',
      positions: outputs.fill.boundaries,
      indices: outputs.fill.indices,
      owners: outputs.fill.owners,
      values: outputs.selected,
      triangleCount: outputs.fill.triangleCount,
      // A fixed range around the average cell of the level: the small variation across the window
      // is real but must not fill the whole ramp.
      range: getMeasureRange(values.measure, values.level)
    },
    drawPointerCell(outputs.pointerCell, [1, 1, 1, 0.25])
  ]
});
