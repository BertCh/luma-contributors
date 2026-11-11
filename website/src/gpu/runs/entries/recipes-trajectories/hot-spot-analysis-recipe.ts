// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Getis-Ord Gi* on counts of observed 2023 earthquakes in a regional grid.

import {
  addHotSpotAnalysisRecipe,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {earthquakeEvents, getEventDots, getEventPosition} from './lib/earthquakes';
import {defineOperationRun} from '../../define-run';
import {createLegend} from '../../text';
import {getRasterCell, getRasterCellCenter} from '../../pick';
import {parameters} from './hot-spot-analysis-recipe.parameters';

const COLUMNS = 32;
const ROWS = 18;

export default defineOperationRun({
  parameters,
  caption: 'Gi* hot spots of gridded 2023 earthquake counts in the Türkiye–Syria region.',
  createInputs: ({world}) => ({
    world,
    dots: getEventDots(world.width, world.height),
    events: earthquakeEvents.map(event => {
      const [x, y] = getEventPosition(event, COLUMNS, ROWS);
      return {
        cell:
          (ROWS - 1 - Math.min(ROWS - 1, Math.floor(y))) * COLUMNS +
          Math.min(COLUMNS - 1, Math.floor(x)),
        magnitude: event[3]
      };
    })
  }),
  build(builder, {events, dots}, values) {
    const cellValues = builder.allocate('values', 'float32', COLUMNS * ROWS);
    const zScores = builder.allocate('z-scores', 'float32', COLUMNS * ROWS);
    const parameterBuffer = new GPUParameterBuffer(builder.device, {
      id: 'autocorrelation-parameters',
      format: 'float32',
      length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
    });
    const neighborCount = (2 * values.radius + 1) ** 2;
    addHotSpotAnalysisRecipe(builder.graph, {
      source: {
        kind: 'lattice',
        values: cellValues.view,
        width: COLUMNS,
        height: ROWS,
        radius: values.radius,
        neighborCapacity: COLUMNS * ROWS * neighborCount
      },
      parameters: parameterBuffer,
      minimumNeighbors: values.minimumNeighbors,
      outputs: {zScores: zScores.view}
    });
    return {
      outputs: {zScores, dots: builder.upload('events', 'float32x2', dots)},
      update: current => {
        const counts = new Float32Array(COLUMNS * ROWS);
        for (const event of events)
          if (event.magnitude >= current.minimumMagnitude) counts[event.cell]++;
        cellValues.write(counts);
      }
    };
  },
  rebuildOn: ['radius', 'minimumNeighbors'],
  // The pointer picks a cell: the box is the neighborhood Gi* summed for it, with its z-score.
  interaction: {
    kind: 'window',
    hint: 'hover a cell',
    rest: ({toUnit, world}) =>
      toUnit(getRasterCellCenter(COLUMNS, ROWS, [0, 0, world.width, world.height], 9, 11)),
    resolve(pointer, {outputs, values, world, peek}) {
      const bounds = [0, 0, world.width, world.height] as const;
      const cell = getRasterCell(COLUMNS, ROWS, bounds, pointer.x, pointer.y);
      const radius = values.radius;
      const [centerX, centerY] = getRasterCellCenter(COLUMNS, ROWS, bounds, cell.column, cell.row);
      const halfWidth = ((radius + 0.5) * world.width) / COLUMNS;
      const halfHeight = (radius + 0.5) / ROWS;
      const zScores = peek(outputs.zScores);
      const neighborCount = (2 * radius + 1) ** 2;
      return {
        glyphs: [
          {
            shape: 'box',
            minimum: [centerX - halfWidth, centerY - halfHeight],
            maximum: [centerX + halfWidth, centerY + halfHeight]
          }
        ],
        readout: `cell z ${zScores ? zScores[cell.index].toFixed(1) : '...'} · radius ${radius} · ${neighborCount} cells`
      };
    }
  },
  draw: ({outputs, theme}) => [
    {
      layer: 'raster',
      values: outputs.zScores,
      columns: COLUMNS,
      rows: ROWS,
      ramp: 'diverging',
      range: [-4, 4],
      alpha: 0.85
    },
    {layer: 'dots', context: true, positions: outputs.dots, radius: 3, color: [...theme.ink, 0.75]},
    {
      layer: 'text',
      ...createLegend({
        corner: 'top-right',
        entries: [
          {label: 'cell hot by Gi* (ramp tops out at +4)', swatch: [...theme.diverging[5]]},
          {label: 'earthquake epicentre', swatch: [...theme.ink]}
        ]
      })
    }
  ]
});
