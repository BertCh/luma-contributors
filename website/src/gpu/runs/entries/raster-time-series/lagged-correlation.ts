// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterLaggedCorrelation correlates every pixel's series with one reference index at a set of
// slice lags, on the pairs where both are valid: a teleconnection map. At a positive lag the index
// leads the pixel. The reference here is the mean series of the three westernmost map columns; the
// seasonal peak arrives later toward the east (a synthetic green-up wave on measured Dixie NDVI
// levels), so each pixel correlates best at its own lag. The map shows the correlation at the lag
// selected by `lag` (the operation computes all seven). Above, the reference and the pixel that
// correlates best with it at the chosen lag (the arrow); below, that pixel's correlation at every lag.
// Pixels with fewer than `minimumPairCount` pairs stay blank.

import {
  getGPURasterLaggedCorrelationParameterValues,
  GPURasterLaggedCorrelation
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {formatNumber} from '../../interaction';
import {
  allocateDots,
  getAccent,
  getChartRectangles,
  getDotsLayer,
  getFrameGlyph,
  getHorizontalRule,
  getInk,
  getMapLayer,
  getVerticalRule,
  writeDots
} from './_lib/charts';
import {getRasterCellCenter} from '../../pick';
import {formatCell, getCellOutline, getProbeCell} from './_lib/probe';
import {createBlankMap, readFloat32} from './_lib/readback';
import {createSeasonalStack} from './_lib/stacks';
import {parameters} from './lagged-correlation.parameters';

const SLICE_COUNT = 64;
const LAGS = [-4, -2, 0, 2, 4, 6, 8] as const;
const REST = [0.55, 0.5] as const;
const [SERIES_FRAME, LAG_FRAME] = getChartRectangles(2);
const SERIES_RANGE = [-0.2, 1] as const;
const CORRELATION_RANGE = [-1, 1] as const;
const SLICE_RANGE = [0, SLICE_COUNT - 1] as const;
const LAG_RANGE = [LAGS[0] - 1, LAGS[LAGS.length - 1] + 1] as const;

/** The reference index: the series of the pixel the pointer chose. */
function getReference(stack: {observed: Float32Array; referenceCell: number}): Float32Array {
  return stack.observed.slice(
    stack.referenceCell * SLICE_COUNT,
    (stack.referenceCell + 1) * SLICE_COUNT
  );
}

export default defineOperationRun({
  parameters,
  caption:
    'Synthetic green-up wave moving east on measured Dixie NDVI levels; reference is the picked pixel. Regular slots.',
  createInputs: ({aspect, random}) => {
    const stack = createSeasonalStack(aspect, random, {
      sliceCount: SLICE_COUNT,
      yearCount: 4,
      jitter: 0,
      phaseDrift: 0.35
    });
    const observed = stack.observe(0.05, 0.1);
    const column = Math.floor(REST[0] * stack.grid.columns);
    const row = Math.floor(REST[1] * stack.grid.rows);
    return {...stack, observed, referenceCell: row * stack.grid.columns + column};
  },
  build(builder, stack) {
    const {grid, sliceCount} = stack;
    const series = builder.upload('series', 'float32', stack.observed);
    const reference = builder.upload('reference', 'float32', getReference(stack));
    const parameterBuffer = builder.allocate('parameters', 'float32', 1);
    const output = {
      correlations: builder.allocate('correlations', 'float32', grid.cellCount * LAGS.length),
      status: builder.allocate('status', 'uint32', grid.cellCount * LAGS.length),
      pairCount: builder.allocate('pair-count', 'uint32', grid.cellCount * LAGS.length)
    };
    builder.graph.add(
      new GPURasterLaggedCorrelation({
        cellCount: grid.cellCount,
        sliceCount,
        values: series.view,
        referenceIndex: reference.view,
        parameters: parameterBuffer.view,
        lags: LAGS,
        output: {
          correlations: output.correlations.view,
          status: output.status.view,
          pairCount: output.pairCount.view
        }
      })
    );
    return {
      outputs: {
        ...output,
        blank: createBlankMap(builder, 'blank', grid.cellCount),
        level: builder.upload('level', 'float32', stack.level),
        cellSeries: allocateDots(builder, 'cell-series', sliceCount),
        referenceSeries: allocateDots(builder, 'reference-series', sliceCount),
        lagDots: allocateDots(builder, 'lag-dots', LAGS.length)
      },
      update: current => {
        stack.observed = stack.observe(0.05, current.gapFraction);
        series.write(stack.observed);
        reference.write(getReference(stack));
        parameterBuffer.write(
          getGPURasterLaggedCorrelationParameterValues({minimumPairCount: current.minimumPairCount})
        );
      }
    };
  },
  async afterRun({outputs, inputs, values, upload}) {
    const correlations = await readFloat32(outputs.correlations);
    const lagIndex = Math.max(0, LAGS.indexOf(values.lag as (typeof LAGS)[number]));
    const map = Float32Array.from({length: inputs.grid.cellCount}, (_, cell) =>
      inputs.level[cell] > 0 ? correlations[cell * LAGS.length + lagIndex] : Number.NaN
    );
    return {map: upload('correlation-map', 'float32', map)};
  },
  interaction: {
    kind: 'place',
    hint: 'pick the reference pixel',
    rest: REST,
    resolve: (pointer, {inputs: stack, outputs, values, world, peek}) => {
      const {grid} = stack;
      const cell = getProbeCell(grid, world, pointer);
      stack.referenceCell = cell.index;
      const correlations = peek(outputs.correlations);
      const lagIndex = Math.max(0, LAGS.indexOf(values.lag as (typeof LAGS)[number]));
      // The partner is the pixel that correlates best with the reference at the chosen lag.
      let partner = -1;
      let best = -2;
      if (correlations) {
        for (let index = 0; index < grid.cellCount; index++) {
          const value = correlations[index * LAGS.length + lagIndex];
          if (index !== cell.index && Number.isFinite(value) && value > best) {
            best = value;
            partner = index;
          }
        }
      }
      const slices = Array.from({length: SLICE_COUNT}, (_, slice) => slice);
      writeDots(outputs.referenceSeries, slices, getReference(stack));
      writeDots(
        outputs.cellSeries,
        slices,
        partner >= 0
          ? stack.observed.subarray(partner * SLICE_COUNT, (partner + 1) * SLICE_COUNT)
          : []
      );
      const row =
        correlations && partner >= 0
          ? Array.from(correlations.subarray(partner * LAGS.length, (partner + 1) * LAGS.length))
          : [];
      writeDots(outputs.lagDots, LAGS, row);
      const [referenceX, referenceY] = getRasterCellCenter(
        grid.columns,
        grid.rows,
        [0, 0, world.width, world.height],
        cell.column,
        cell.row
      );
      const partnerColumn = partner % grid.columns;
      const partnerRow = Math.floor(partner / grid.columns);
      const [partnerX, partnerY] =
        partner >= 0
          ? getRasterCellCenter(
              grid.columns,
              grid.rows,
              [0, 0, world.width, world.height],
              partnerColumn,
              partnerRow
            )
          : [referenceX, referenceY];
      return {
        rerun: true,
        spotlight: null,
        glyphs: [
          getCellOutline(grid, world, cell),
          getFrameGlyph(world, SERIES_FRAME),
          getFrameGlyph(world, LAG_FRAME),
          getHorizontalRule(world, LAG_FRAME, CORRELATION_RANGE, 0),
          getVerticalRule(world, LAG_FRAME, LAG_RANGE, values.lag, true),
          ...(partner >= 0
            ? [
                {
                  shape: 'arrow' as const,
                  from: [referenceX, referenceY] as [number, number],
                  to: [partnerX, partnerY] as [number, number]
                }
              ]
            : [])
        ],
        readout:
          partner >= 0
            ? `lag ${values.lag} · best r ${formatNumber(best)} · ${partnerColumn - cell.column} columns east · ${formatCell(grid, cell)}`
            : 'reading the correlations'
      };
    }
  },
  draw: ({inputs: stack, outputs, extra, theme}) => [
    getMapLayer(outputs.level, stack.grid, stack.grid.mapLeft, {
      context: true,
      ramp: 'sequential',
      range: [0, 1]
    }),
    getMapLayer(extra?.map ?? outputs.blank, stack.grid, stack.grid.mapLeft, {
      ramp: 'diverging',
      range: [-1, 1]
    }),
    getDotsLayer(
      outputs.referenceSeries,
      SERIES_FRAME,
      SERIES_RANGE,
      SLICE_RANGE,
      getAccent(theme, 1),
      3.5
    ),
    getDotsLayer(outputs.cellSeries, SERIES_FRAME, SERIES_RANGE, SLICE_RANGE, getInk(theme), 4),
    getDotsLayer(outputs.lagDots, LAG_FRAME, CORRELATION_RANGE, LAG_RANGE, getAccent(theme, 0), 6)
  ]
});
