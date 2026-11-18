// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSinkhornDivergence between two mass fields on a 32 x 18 grid. The source is the count of
// 2020-2024 earthquakes off Honshu per cell; the target is the same field, moved by the pointer.
// With no shift the divergence is zero (the debiasing); it grows as the mass has to travel farther.
// Bars: OT(source, target), OT(source, source), OT(target, target) and the divergence S.

import {
  getGPUSinkhornDivergenceParameterValues,
  GPUSinkhornDivergence
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {getHonshuEvents, HONSHU_WINDOW} from '../../../data/point-observations';
import {drawPanel, uploadPanel} from './_lib/grid-panel';
import {readFloats} from './_lib/readback';
import {parameters} from './sinkhorn-divergence.parameters';

const COLUMNS = 32;
const ROWS = 18;
const MAXIMUM_SHIFT = [12, 7] as const;
const DEFAULT_SHIFT = [4, 2] as const;
const CHART = [0.64, 0.08, 0.97, 0.38] as const;
const COMPARE = [0.64, 0.46, 0.97, 0.72] as const;
const PANEL = [0.64, 0.08, 0.97, 0.72] as const;

/** Event counts per grid cell, row 0 at the top. */
function countEvents(positions: Float32Array): Float32Array {
  const [west, south, east, north] = HONSHU_WINDOW;
  const counts = new Float32Array(COLUMNS * ROWS);
  for (let event = 0; event < positions.length / 2; event++) {
    const column = Math.min(
      COLUMNS - 1,
      Math.floor(((positions[event * 2] - west) / (east - west)) * COLUMNS)
    );
    const row = Math.min(
      ROWS - 1,
      Math.floor(((north - positions[event * 2 + 1]) / (north - south)) * ROWS)
    );
    counts[row * COLUMNS + column]++;
  }
  return counts;
}

/** The field moved by whole cells; mass pushed off the grid is dropped. */
function shiftField(field: Float32Array, columns: number, rows: number): Float32Array {
  const shifted = new Float32Array(field.length);
  for (let row = 0; row < ROWS; row++) {
    for (let column = 0; column < COLUMNS; column++) {
      const targetColumn = column + columns;
      const targetRow = row + rows;
      if (targetColumn < 0 || targetColumn >= COLUMNS || targetRow < 0 || targetRow >= ROWS)
        continue;
      shifted[targetRow * COLUMNS + targetColumn] = field[row * COLUMNS + column];
    }
  }
  return shifted;
}

export default defineOperationRun({
  parameters,
  caption:
    'Honshu earthquake counts per cell and the same field moved by the pointer. Upper bars: plain difference (total variation) and sqrt(S) in cells (of 16); lower bars: OT(a,b), OT(a,a), OT(b,b), S. Mass moved off the grid is dropped and the target renormalized, and total mass is ignored.',
  createInputs: ({random, world}) => ({
    world,
    source: countEvents(getHonshuEvents(random, 4000).positions),
    shift: [...DEFAULT_SHIFT] as [number, number],
    distance: 0
  }),
  build(builder, inputs, values) {
    const source = builder.upload('source', 'float32', inputs.source);
    const target = builder.allocate('target', 'float32', COLUMNS * ROWS);
    const parameterBuffer = builder.allocate('parameters', 'float32', 6);
    const divergence = builder.allocate('divergence', 'float32', 1);
    const components = builder.allocate('components', 'float32', 3);
    const iterationLimitReached = builder.allocate('iterationLimitReached', 'uint32', 1);
    builder.graph.add(
      new GPUSinkhornDivergence({
        grid: makeGPURasterMetadata({width: COLUMNS, height: ROWS}),
        source: source.view,
        target: target.view,
        parameters: parameterBuffer.view,
        maximumIterations: values['maximumIterations'],
        output: {
          divergence: divergence.view,
          components: components.view,
          status: {iterationLimitReached: iterationLimitReached.view}
        }
      })
    );
    return {
      outputs: {
        source,
        target,
        divergence,
        components,
        iterationLimitReached,
        panel: uploadPanel(builder, PANEL, inputs.world, 0.015)
      },
      update: current => {
        target.write(shiftField(inputs.source, inputs.shift[0], inputs.shift[1]));
        parameterBuffer.write(
          getGPUSinkhornDivergenceParameterValues({
            cellWidth: 1,
            cellHeight: 1,
            epsilon: current['epsilon']
          })
        );
      }
    };
  },
  rebuildOn: ['maximumIterations'],
  buildCache: 3,
  async afterRun({inputs, outputs, upload}) {
    const components = await readFloats(outputs.components);
    const divergence = await readFloats(outputs.divergence);
    const bars = Float32Array.of(components[0], components[1], components[2], divergence[0]).map(
      value => (Number.isFinite(value) ? Math.max(value, 0) : 0)
    );
    // The plain cell-by-cell difference (total variation of the normalized fields) saturates at 1 once
    // the fields no longer overlap; sqrt(S), the typical distance moved in cells, keeps growing.
    const target = shiftField(inputs.source, inputs.shift[0], inputs.shift[1]);
    const sourceTotal = inputs.source.reduce((sum, value) => sum + value, 0);
    const targetTotal = target.reduce((sum, value) => sum + value, 0) || 1;
    let variation = 0;
    for (let cell = 0; cell < target.length; cell++) {
      variation += Math.abs(inputs.source[cell] / sourceTotal - target[cell] / targetTotal) / 2;
    }
    const distance = Number.isFinite(divergence[0]) ? Math.sqrt(Math.max(divergence[0], 0)) : 0;
    inputs.distance = distance;
    return {
      bars: upload('bars', 'float32', bars),
      compare: upload('compare', 'float32', Float32Array.of(variation, distance / 16))
    };
  },
  interaction: {
    kind: 'drag',
    hint: 'drag the target',
    rest: [
      0.5 + DEFAULT_SHIFT[0] / (2 * MAXIMUM_SHIFT[0]),
      0.5 + DEFAULT_SHIFT[1] / (2 * MAXIMUM_SHIFT[1])
    ],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const shift: [number, number] = [
        Math.round((pointer.unitX - 0.5) * 2 * MAXIMUM_SHIFT[0]),
        Math.round((pointer.unitY - 0.5) * 2 * MAXIMUM_SHIFT[1])
      ];
      const changed = shift[0] !== inputs.shift[0] || shift[1] !== inputs.shift[1];
      inputs.shift = shift;
      const divergence = peek(outputs.divergence)?.[0];
      const iterationLimitReached = peek(outputs.iterationLimitReached)?.[0];
      const cellWidth = world.width / COLUMNS;
      const cellHeight = world.height / ROWS;
      return {
        rerun: changed,
        glyphs: [
          {
            shape: 'arrow',
            from: [world.width / 2, world.height / 2],
            to: [world.width / 2 + shift[0] * cellWidth, world.height / 2 - shift[1] * cellHeight]
          }
        ],
        readout: `shift ${shift[0]},${shift[1]} cells · epsilon ${values['epsilon']} · typical distance ${inputs.distance.toFixed(1)} cells · S ${
          divergence === undefined ? '…' : Number.isFinite(divergence) ? divergence.toFixed(2) : '–'
        }${iterationLimitReached === undefined ? ' · status pending' : iterationLimitReached ? ' · iteration limit reached (estimate)' : ''}`
      };
    }
  },
  draw: ({outputs, extra, theme}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.source,
      columns: COLUMNS,
      rows: ROWS,
      ramp: 'grey',
      range: [0, 10]
    },
    {
      layer: 'raster',
      values: outputs.target,
      columns: COLUMNS,
      rows: ROWS,
      ramp: 'sequential',
      range: [0, 10],
      alpha: 0.6
    },
    drawPanel(outputs.panel, theme.background),
    {layer: 'chart', kind: 'bars', values: extra.bars, ramp: 'sequential', rectangle: CHART},
    {
      layer: 'chart',
      kind: 'bars',
      values: extra.compare,
      ramp: 'diverging',
      range: [0, 1],
      rectangle: COMPARE
    }
  ]
});
