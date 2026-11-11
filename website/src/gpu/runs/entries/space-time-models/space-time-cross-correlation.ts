// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpaceTimeCrossCorrelation of two real variables per lattice cell and period from the GOES
// lightning extract: flash count (first) against mean flash area (second), over lags of -k to k
// periods of 2 hours. The map shows each cell's best lag (the lag of largest |r|; negative means
// the second leads) where its adjusted p-value passes `significanceLevel`; other cells stay on the
// muted total flash counts. The pointer probes a cell: its correlation at every lag as bars in an
// inset, the best lag marked.

import {GPUSpaceTimeCrossCorrelation} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {type GPUMultipleTestingMethod} from '@luma-contributors/analysis/analysis-core';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import type {Point, RunGlyph} from '../../interaction';
import {
  createFlashCube,
  getCellAt,
  getCellBox,
  readRunBuffer,
  type FlashCube
} from './_lib/flash-cube';
import {
  createInsetBacking,
  getInsetBackingLayer,
  getRectangleCorners,
  type InsetBacking,
  type InsetRectangle
} from './_lib/inset';
import {parameters} from './space-time-cross-correlation.parameters';

const ROWS = 6;
const PERIOD_COUNT = 9;
const METHODS: GPUMultipleTestingMethod[] = [
  'none',
  'bonferroni',
  'holm',
  'benjamini-hochberg',
  'benjamini-yekutieli'
];
/** Inset rectangle as fractions of the view (S1), and its mirror for a probe in the bottom left. */
const INSET = [0.04, 0.06, 0.46, 0.34] as const;
const INSET_MIRRORED = [0.54, 0.06, 0.96, 0.34] as const;

type Inputs = {cube: FlashCube; totals: Float32Array; focus: number};
type Outputs = {
  totals: RunBuffer;
  correlations: RunBuffer;
  adjustedPValues: RunBuffer;
  pairCounts: RunBuffer;
  backing: InsetBacking;
};

function getLags(maximumLag: number): number[] {
  return Array.from({length: 2 * maximumLag + 1}, (_, index) => index - maximumLag);
}

/** Lag index of the largest |r| among the finite correlations of a cell, or -1. */
function findBestLag(correlations: ArrayLike<number>, cell: number, lagCount: number): number {
  let best = -1;
  let strength = -1;
  for (let index = 0; index < lagCount; index++) {
    const correlation = correlations[cell * lagCount + index];
    if (Number.isFinite(correlation) && Math.abs(correlation) > strength) {
      strength = Math.abs(correlation);
      best = index;
    }
  }
  return best;
}

function getInsetRectangle(
  world: {width: number; height: number},
  mirrored: boolean
): InsetRectangle {
  const fractions = mirrored ? INSET_MIRRORED : INSET;
  return {
    x0: fractions[0] * world.width,
    y0: fractions[1] * world.height,
    x1: fractions[2] * world.width,
    y1: fractions[3] * world.height
  };
}

export default defineOperationRun<Inputs, Outputs, {mapped: RunBuffer}>({
  parameters,
  caption:
    'GOES lightning, May 21–22 2024, in coarse lattice cells and 9 periods: flash count against mean flash area (empty periods take the cell mean); best lag where significant.',
  createInputs: ({world}) => {
    const cube = createFlashCube(world, ROWS, PERIOD_COUNT);
    const totals = new Float32Array(cube.cellCount);
    for (let period = 0; period < cube.periodCount; period++) {
      for (let cell = 0; cell < cube.cellCount; cell++) {
        totals[cell] += cube.counts[period * cube.cellCount + cell];
      }
    }
    return {cube, totals, focus: -1};
  },
  build(builder, {cube, totals}, values) {
    const lags = getLags(values.lags);
    const rowCount = cube.cellCount * lags.length;
    const outputs = {
      totals: builder.upload('totals', 'float32', totals),
      correlations: builder.allocate('correlations', 'float32', rowCount),
      adjustedPValues: builder.allocate('adjusted-p-values', 'float32', rowCount),
      pairCounts: builder.allocate('pair-counts', 'uint32', rowCount),
      backing: createInsetBacking(
        builder,
        'inset',
        getInsetRectangle({width: cube.width, height: 1}, false)
      )
    };
    builder.graph.add(
      new GPUSpaceTimeCrossCorrelation({
        first: builder.upload('first', 'float32', cube.counts).view,
        second: builder.upload('second', 'float32', cube.meanArea).view,
        layout: 'period-major',
        cellCount: cube.cellCount,
        periodCount: cube.periodCount,
        lags,
        multipleTesting: METHODS[values.multipleTesting],
        output: {
          correlations: outputs.correlations.view,
          pValues: builder.allocate('p-values', 'float32', rowCount).view,
          adjustedPValues: outputs.adjustedPValues.view,
          pairCounts: outputs.pairCounts.view
        }
      })
    );
    return {outputs};
  },
  rebuildOn: ['lags', 'multipleTesting'],
  async afterRun({inputs, outputs, values, upload}) {
    const {cube} = inputs;
    const lags = getLags(values.lags);
    const correlations = (await readRunBuffer(outputs.correlations)) as Float32Array;
    const adjusted = (await readRunBuffer(outputs.adjustedPValues)) as Float32Array;
    const mapped = new Float32Array(cube.cellCount);
    for (let cell = 0; cell < cube.cellCount; cell++) {
      const best = findBestLag(correlations, cell, lags.length);
      mapped[cell] =
        best >= 0 && adjusted[cell * lags.length + best] < values.significanceLevel
          ? lags[best]
          : Number.NaN;
    }
    return {mapped: upload('mapped', 'float32', mapped)};
  },
  interaction: {
    kind: 'probe',
    hint: 'hover a cell',
    rest: [0.41, 0.33],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const {cube} = inputs;
      const lags = getLags(values.lags);
      const cell = getCellAt(cube, pointer.x, pointer.y).index;
      inputs.focus = cell;
      const mirrored = pointer.unitX < 0.4 && pointer.unitY > 0.6;
      const panel = getInsetRectangle(world, mirrored);
      outputs.backing.positions.write(getRectangleCorners(panel));
      const glyphs: RunGlyph[] = [
        {shape: 'box', ...getCellBox(cube, cell)},
        {shape: 'box', minimum: [panel.x0, panel.y0], maximum: [panel.x1, panel.y1]}
      ];
      const padding = 0.025;
      const panelWidth = panel.x1 - panel.x0 - 2 * padding;
      const panelHeight = panel.y1 - panel.y0 - 2 * padding;
      const slotWidth = panelWidth / lags.length;
      const getBarX = (index: number) => panel.x0 + padding + (index + 0.5) * slotWidth;
      const baseline = panel.y0 + padding + panelHeight / 2;
      const toPoint = (index: number, correlation: number): Point => [
        getBarX(index),
        baseline + (correlation * panelHeight) / 2
      ];
      glyphs.push({
        shape: 'line',
        from: [panel.x0 + padding, baseline],
        to: [panel.x1 - padding, baseline],
        dashed: true
      });
      const correlations = peek(outputs.correlations);
      const adjusted = peek(outputs.adjustedPValues);
      const pairs = peek(outputs.pairCounts);
      if (!correlations || !adjusted || !pairs) return {glyphs, spotlight: null, readout: '…'};
      const zeroLag = lags.indexOf(0);
      glyphs.push({
        shape: 'line',
        from: toPoint(zeroLag, -1),
        to: toPoint(zeroLag, -0.82),
        dashed: true
      });
      for (let index = 0; index < lags.length; index++) {
        const correlation = correlations[cell * lags.length + index];
        if (Number.isFinite(correlation)) {
          glyphs.push({shape: 'line', from: toPoint(index, 0), to: toPoint(index, correlation)});
        }
      }
      const best = findBestLag(correlations, cell, lags.length);
      if (best < 0) return {glyphs, spotlight: null, readout: 'untestable · flat series'};
      const row = cell * lags.length + best;
      glyphs.push({shape: 'ring', at: toPoint(best, correlations[row]), radius: 0.018});
      const lag = lags[best];
      return {
        glyphs,
        spotlight: null,
        readout: `best lag ${lag > 0 ? '+' : ''}${lag} · r ${correlations[row].toFixed(2)} · adj p ${adjusted[row].toFixed(2)}`
      };
    }
  },
  draw: ({inputs, outputs, extra, theme}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.totals,
      columns: inputs.cube.columns,
      rows: inputs.cube.rows,
      ramp: 'grey',
      range: [0, 12]
    },
    ...(extra
      ? [
          {
            layer: 'raster' as const,
            values: extra.mapped,
            columns: inputs.cube.columns,
            rows: inputs.cube.rows,
            ramp: 'diverging' as const,
            range: [-5, 5] as const
          }
        ]
      : []),
    getInsetBackingLayer(outputs.backing, theme)
  ]
});
