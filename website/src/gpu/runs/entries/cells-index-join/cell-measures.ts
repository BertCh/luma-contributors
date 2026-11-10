// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCellMeasures computes the geodesic area and perimeter of H3 cells. Honshu epicenters
// are keyed with GPUPointToCell; the cells are filled by the measure the operation wrote.

import {GPUCellMeasures, GPUPointToCell} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {getHonshuEvents, HONSHU_WINDOW} from '../../../data/point-observations';
import {addCellFill, addPointerCell, drawPointerCell} from './support/cells';
import {getWindowCenter, readPointerCell} from './support/pointer';
import {parameters} from './cell-measures.parameters';

const NOMINAL_AREAS = [12393, 1770, 252.9, 36.1];

function getMeasureRange(measure: number, resolution: number): [number, number] {
  const area = NOMINAL_AREAS[resolution - 3];
  // A regular hexagon of the nominal area has this perimeter.
  const perimeter = 6 * Math.sqrt(area / 2.598);
  if (measure === 0) return [area * 0.85, area * 1.15];
  if (measure === 1) return [perimeter * 0.92, perimeter * 1.08];
  return [0, 8];
}

export default defineOperationRun({
  parameters,
  geographicWindow: HONSHU_WINDOW,
  caption: 'Geodesic measures of occupied H3 cells at earthquake epicenters east of Honshu.',
  createInputs: ({random}) => ({
    ...getHonshuEvents(random, 700),
    pointer: [...getWindowCenter(HONSHU_WINDOW)] as [number, number]
  }),
  build(builder, points, values) {
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const cells = builder.allocate('cells', 'uint32x2', points.count);
    builder.graph.add(
      new GPUPointToCell({
        family: 'h3',
        resolution: values.resolution,
        positions: positions.view,
        output: {cells: cells.view}
      })
    );
    const areas = builder.allocate('areas', 'float32', points.count);
    const perimeters = builder.allocate('perimeters', 'float32', points.count);
    const edgeCounts = builder.allocate('edge-counts', 'uint32', points.count);
    builder.graph.add(
      new GPUCellMeasures({
        family: 'h3',
        cells: cells.view,
        units: 'kilometers',
        output: {areas: areas.view, perimeters: perimeters.view, edgeCounts: edgeCounts.view}
      })
    );
    const pointer = builder.allocate('pointer', 'float32x2', 1);
    const pointerCell = addPointerCell(builder, 'h3', values.resolution, pointer);
    const pointerMeasures = {
      areas: builder.allocate('pointer-areas', 'float32', 1),
      perimeters: builder.allocate('pointer-perimeters', 'float32', 1),
      edgeCounts: builder.allocate('pointer-edge-counts', 'uint32', 1)
    };
    builder.graph.add(
      new GPUCellMeasures({
        id: 'pointer-measures',
        family: 'h3',
        cells: pointerCell.cells.view,
        units: 'kilometers',
        output: {
          areas: pointerMeasures.areas.view,
          perimeters: pointerMeasures.perimeters.view,
          edgeCounts: pointerMeasures.edgeCounts.view
        }
      })
    );
    return {
      outputs: {
        areas,
        perimeters,
        edgeCounts,
        pointerCell,
        pointerMeasures,
        fill: addCellFill(builder, 'h3', cells)
      },
      update: () => pointer.write(Float32Array.from(points.pointer))
    };
  },
  rebuildOn: ['resolution'],
  interaction: {
    kind: 'probe',
    hint: 'hover a cell',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, peek}) {
      inputs.pointer = [pointer.x, pointer.y];
      const cell = readPointerCell(peek, outputs.pointerCell);
      const area = peek(outputs.pointerMeasures.areas);
      const perimeter = peek(outputs.pointerMeasures.perimeters);
      return {
        rerun: true,
        glyphs: cell.outline ? [cell.outline] : [],
        readout:
          area && perimeter
            ? `area ${area[0].toFixed(1)} km² · perimeter ${perimeter[0].toFixed(1)} km · ${cell.vertexCount} edges`
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
      values: [outputs.areas, outputs.perimeters, outputs.edgeCounts][values.measure],
      triangleCount: outputs.fill.triangleCount,
      ramp: 'sequential',
      // Fixed ranges around the nominal H3 cell: the small latitude trend must not fill the ramp.
      range: getMeasureRange(values.measure, values.resolution)
    },
    drawPointerCell(outputs.pointerCell, [1, 1, 1, 0.25])
  ]
});
