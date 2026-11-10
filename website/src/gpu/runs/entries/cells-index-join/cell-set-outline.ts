// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCellSetOutline emits only the boundary edges of a set of cells: edges whose neighbor is not in
// the set. The set is a GPUCellAggregation table (sorted, distinct keys); the bright lines are the
// operation's output, drawn exactly `status.count` times.

import {
  GPUCellAggregation,
  GPUCellSetOutline
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {getHonshuEvents, HONSHU_WINDOW} from '../../../data/point-observations';
import {
  addCellFill,
  addPointKeys,
  allocateCellTable,
  getCellFamily,
  getCellResolution,
  getCountTableViews
} from './support/cells';
import {getWindowCenter} from './support/pointer';
import {parameters} from './cell-set-outline.parameters';

const TABLE_CAPACITY = 1024;
const EDGES_PER_CELL = {h3: 10, quadbin: 4};
const BRUSH_POINTS = 36;
/** Brush radius in degrees of latitude; longitude is stretched by the cosine of the latitude. */
const BRUSH_RADIUS = 0.3;
const LONGITUDE_SCALE = Math.cos((((HONSHU_WINDOW[1] + HONSHU_WINDOW[3]) / 2) * Math.PI) / 180);

export default defineOperationRun({
  parameters,
  geographicWindow: HONSHU_WINDOW,
  caption: 'Only the edges on the border of the cell set survive.',
  createInputs: ({random}) => ({
    ...getHonshuEvents(random, 700),
    pointer: [...getWindowCenter(HONSHU_WINDOW)] as [number, number]
  }),
  build(builder, points, values) {
    const family = getCellFamily(values.family);
    const resolution = getCellResolution(family, values.resolution);
    // The epicenters, then a brush of extra points at the pointer; all of them are keyed.
    const positions = builder.allocate('positions', 'float32x2', points.count + BRUSH_POINTS);
    const keys = addPointKeys(builder, family, resolution, positions);
    const table = allocateCellTable(builder, 'table', TABLE_CAPACITY);
    builder.graph.add(
      new GPUCellAggregation({
        family,
        resolution,
        cells: keys.view,
        output: getCountTableViews(table)
      })
    );
    const capacity = TABLE_CAPACITY * EDGES_PER_CELL[family];
    const outline = {
      rows: builder.allocate('outline-rows', 'uint32', capacity),
      cells: builder.allocate('outline-cells', 'uint32x2', capacity),
      edgeIndices: builder.allocate('outline-edges', 'uint32', capacity),
      endpoints: builder.allocate('outline-endpoints', 'float32x4', capacity),
      count: builder.allocate('outline-count', 'uint32', 1),
      overflow: builder.allocate('outline-overflow', 'uint32', 1)
    };
    builder.graph.add(
      new GPUCellSetOutline({
        family,
        cells: table.cells.view,
        count: table.count.view,
        output: {
          rows: outline.rows.view,
          cells: outline.cells.view,
          edgeIndices: outline.edgeIndices.view,
          endpoints: outline.endpoints.view,
          status: {count: outline.count.view, overflow: outline.overflow.view}
        }
      })
    );
    return {
      outputs: {outline, table, fill: addCellFill(builder, family, table.cells)},
      update: current => {
        const rows = new Float32Array((points.count + BRUSH_POINTS) * 2);
        rows.set(
          Float32Array.from(points.positions, (value, index) => {
            const axis = index % 2;
            const anchor = (HONSHU_WINDOW[axis] + HONSHU_WINDOW[axis + 2]) / 2;
            return anchor + (value - anchor) * current.spread;
          })
        );
        for (let brush = 0; brush < BRUSH_POINTS; brush++) {
          const reach = Math.sqrt((brush + 0.5) / BRUSH_POINTS) * BRUSH_RADIUS;
          const angle = brush * 2.399963;
          rows[(points.count + brush) * 2] =
            points.pointer[0] + (Math.cos(angle) * reach) / LONGITUDE_SCALE;
          rows[(points.count + brush) * 2 + 1] = points.pointer[1] + Math.sin(angle) * reach;
        }
        positions.write(rows);
      }
    };
  },
  rebuildOn: ['family', 'resolution'],
  // The pointer is a brush: its points join the set and the outline re-forms around the new cells.
  interaction: {
    kind: 'drag',
    hint: 'paint cells',
    rest: [0.62, 0.62],
    resolve(pointer, {inputs, outputs, peek}) {
      inputs.pointer = [pointer.x, pointer.y];
      const cellCount = peek(outputs.table.count);
      const edgeCount = peek(outputs.outline.count);
      return {
        rerun: true,
        glyphs: [{shape: 'ring', at: inputs.pointer, radius: BRUSH_RADIUS, dashed: true}],
        readout: `set ${cellCount ? cellCount[0] : '…'} cells · ${edgeCount ? edgeCount[0] : '…'} boundary edges`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'polygons',
      context: true,
      positions: outputs.fill.boundaries,
      indices: outputs.fill.indices,
      owners: outputs.fill.owners,
      values: outputs.fill.owners,
      triangleCount: outputs.fill.triangleCount,
      color: [0.4, 0.5, 0.7, 0.35]
    },
    {
      layer: 'lines',
      segments: outputs.outline.endpoints,
      count: outputs.outline.count,
      color: [1, 0.75, 0.2, 1],
      radius: 2
    }
  ]
});
