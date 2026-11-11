// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainViewshed on a Grand Canyon DEM: for every cell, whether a person of observerHeight standing
// at the observer can see it (visible, hidden behind terrain, or out of range). The observer
// position, height and range are per-frame values, so dragging them never recompiles.

import {
  getGPUTerrainViewshedParameterValues,
  GPUTerrainViewshed
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {createLegend} from '../../text';
import {clamp} from '../../interaction';
import {
  createTerrainInput,
  drawElevationContext,
  drawGrid,
  getCellOfPercentage,
  getCellWorldPosition,
  uploadTerrain,
  VISIBILITY_COLORS
} from './support/terrain';
import {parameters} from './viewshed.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Elevation and an observer in; what the observer can see, hide from or not reach, out.',
  createInputs: ({world}) => createTerrainInput(world),
  build(builder, terrain) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const parameterBuffer = builder.allocate('parameters', 'float32', 10);
    const visibility = builder.allocate('visibility', 'uint32', cellCount);
    const observerDot = builder.allocate('observer', 'float32x2', 1);
    builder.graph.add(
      new GPUTerrainViewshed({
        grid,
        elevation: band,
        parameters: parameterBuffer.view,
        output: {visibility: visibility.view}
      })
    );
    return {
      outputs: {elevation, visibility, observerDot},
      update: current => {
        const [column, row] = getCellOfPercentage(
          terrain,
          current.observerColumn,
          current.observerRow
        );
        parameterBuffer.write(
          getGPUTerrainViewshedParameterValues(
            {
              observer: [column, row],
              observerHeight: current.observerHeight,
              maximumDistance: current.maximumDistance
            },
            grid
          )
        );
        observerDot.write(
          Float32Array.from(getCellWorldPosition(terrain, terrain.world, column, row))
        );
      }
    };
  },
  // The pointer is the observer: the viewshed is recomputed for the cell under it.
  interaction: {
    kind: 'place',
    hint: 'move the observer',
    rest: [0.3, 0.55],
    resolve(pointer, {inputs, values, world}) {
      const observerColumn = clamp(Math.round((pointer.x / world.width) * 100), 0, 100);
      const observerRow = clamp(Math.round((1 - pointer.y / world.height) * 100), 0, 100);
      const [column, row] = getCellOfPercentage(inputs, observerColumn, observerRow);
      const observer = getCellWorldPosition(inputs, world, column, row);
      // maximumDistance is in meters; one world unit of height spans rows * cellSize meters.
      const rangeRadius = (values.maximumDistance / (inputs.rows * inputs.cellSize)) * world.height;
      return {
        values: {observerColumn, observerRow},
        glyphs: [
          {shape: 'ring', at: observer, radius: rangeRadius, dashed: true},
          {shape: 'crosshair', at: observer}
        ],
        readout: `observer [${column}, ${row}] · ${values.observerHeight} m up · sees ${values.maximumDistance} m`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.visibility, {
      ramp: 'categorical',
      categories: VISIBILITY_COLORS,
      alpha: 0.7
    }),
    {layer: 'dots', positions: outputs.observerDot, radius: 7, color: [1, 1, 1, 1]},
    {
      layer: 'text',
      ...createLegend({
        corner: 'bottom-left',
        entries: [
          {label: 'visible', swatch: [0.2, 0.85, 0.4]},
          {label: 'hidden by terrain', swatch: [0.27, 0.1, 0.4]}
        ]
      })
    }
  ]
});
