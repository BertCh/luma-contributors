// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpaceTimeLocalMoran of GOES lightning flash counts over the Midwest, binned into a lattice of
// cells and 9 equal periods of the 18-hour window. The map is the cluster map of one `period`:
// each cell is colored by its Moran quadrant where its pseudo p-value (conditional permutation,
// uncorrected, level 0.05) is significant, and left on the flash counts of that period where
// not. `lagged` relates a cell now to its neighbors `lag` periods earlier; `differential` is the
// local Moran of the change over `lag` periods. The pointer is the probe: the cell, and its queen
// neighbors one `lag` earlier, with the cell's statistic and pseudo p-value.

import {
  getGPUPermutationParameterValues,
  GPUSpaceTimeLocalMoran,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  type GPUSpaceTimeLocalMoranVariant
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import type {RunGlyph} from '../../interaction';
import {
  createFlashCube,
  getCellAt,
  getCellBox,
  getCellCenter,
  getCountRadius,
  readRunBuffer,
  sliceFlashPeriod,
  type FlashCube
} from './_lib/flash-cube';
import {createInsetBacking, getInsetBackingLayer} from './_lib/inset';
import {parameters} from './space-time-local-moran.parameters';

const ROWS = 12;
const PERIOD_COUNT = 9;
const MAXIMUM_PERMUTATIONS = 499;
const VARIANTS: GPUSpaceTimeLocalMoranVariant[] = ['lagged', 'differential'];
const CLASS_NAMES = ['Not significant', 'High-High', 'Low-High', 'Low-Low', 'High-Low'];
/** Timeline strip (S3) as fractions of the view. */
const STRIP = [0.06, 0.04, 0.94, 0.16] as const;
const NOT_SIGNIFICANT = 0xffffffff;

type Inputs = {cube: FlashCube; focus: number; periodTotals: Float32Array};
type Outputs = {
  counts: RunBuffer;
  observed: RunBuffer;
  pseudoPValues: RunBuffer;
  adjustedPValues: RunBuffer;
  significant: RunBuffer;
  quadrants: RunBuffer;
  totals: RunBuffer;
  backing: ReturnType<typeof createInsetBacking>;
};

/** Moran quadrant 1 to 4; the differential variant takes it from the signs of I and the change. */
function getClass(
  cube: FlashCube,
  values: Readonly<Record<string, number>>,
  cell: number,
  observed: number,
  lagged: number
): number {
  if (values.variant === 0) return lagged;
  // The change is centred on its mean over valid cells before its sign is read, as GeoDa does.
  const change = getChange(cube, values, cell) - getMeanChange(cube, values);
  return observed >= 0 ? (change >= 0 ? 1 : 3) : change >= 0 ? 4 : 2;
}

function getChange(
  cube: FlashCube,
  values: Readonly<Record<string, number>>,
  cell: number
): number {
  return (
    cube.counts[values.period * cube.cellCount + cell] -
    cube.counts[(values.period - values.lag) * cube.cellCount + cell]
  );
}

const meanChangeCache = new WeakMap<FlashCube, Map<string, number>>();

/** Mean of the change over the cells where both periods are finite (cached per period and lag). */
function getMeanChange(cube: FlashCube, values: Readonly<Record<string, number>>): number {
  const key = `${values.period}|${values.lag}`;
  let byKey = meanChangeCache.get(cube);
  if (!byKey) {
    byKey = new Map();
    meanChangeCache.set(cube, byKey);
  }
  const cached = byKey.get(key);
  if (cached !== undefined) return cached;
  let total = 0;
  let count = 0;
  for (let cell = 0; cell < cube.cellCount; cell++) {
    const change = getChange(cube, values, cell);
    if (Number.isFinite(change)) {
      total += change;
      count++;
    }
  }
  const mean = count > 0 ? total / count : 0;
  byKey.set(key, mean);
  return mean;
}

export default defineOperationRun<Inputs, Outputs, {clusters: RunBuffer}>({
  parameters,
  caption:
    'GOES lightning flashes, May 21–22 2024, counted into lattice cells and 9 periods: local Moran cluster map of one period; High-High is a storm cell that persists.',
  createInputs: ({world}) => {
    const cube = createFlashCube(world, ROWS, PERIOD_COUNT);
    const periodTotals = Float32Array.from({length: cube.periodCount}, (_, period) =>
      sliceFlashPeriod(cube, period).reduce((sum, count) => sum + count, 0)
    );
    return {cube, focus: -1, periodTotals};
  },
  build(builder, {cube, periodTotals}, values) {
    const {cellCount} = cube;
    const cubeBuffer = builder.upload('cube', 'float32', cube.counts);
    const parameterBuffer = builder.allocate(
      'permutation',
      'uint32',
      GPU_PERMUTATION_PARAMETER_LENGTH
    );
    const outputs = {
      counts: builder.upload('counts', 'float32', sliceFlashPeriod(cube, values.period)),
      observed: builder.allocate('observed', 'float32', cellCount),
      pseudoPValues: builder.allocate('pseudo-p-values', 'float32', cellCount),
      adjustedPValues: builder.allocate('adjusted-p-values', 'float32', cellCount),
      significant: builder.allocate('significant', 'uint32', cellCount),
      quadrants: builder.allocate('quadrants', 'uint32', cellCount),
      totals: builder.upload('period-totals', 'float32', periodTotals),
      backing: createInsetBacking(builder, 'strip', {
        x0: STRIP[0] * cube.width,
        y0: STRIP[1],
        x1: STRIP[2] * cube.width,
        y1: STRIP[3]
      })
    };
    const variant = VARIANTS[values.variant];
    builder.graph.add(
      new GPUSpaceTimeLocalMoran({
        weights: uploadSpatialWeights(builder, cube.weights),
        cube: cubeBuffer.view,
        layout: 'period-major',
        periodCount: cube.periodCount,
        period: values.period,
        lag: values.lag,
        variant,
        parameters: parameterBuffer.view,
        maximumPermutations: MAXIMUM_PERMUTATIONS,
        maximumNeighbors: 8,
        islands: 'keep',
        multipleTesting: 'none',
        output: {
          exceedances: builder.allocate('exceedances', 'uint32', cellCount).view,
          pseudoPValues: outputs.pseudoPValues.view,
          observed: outputs.observed.view,
          significant: outputs.significant.view,
          adjustedPValues: outputs.adjustedPValues.view,
          islandCount: builder.allocate('island-count', 'uint32', 1).view,
          quadrants: variant === 'lagged' ? outputs.quadrants.view : undefined,
          status: {overflow: builder.allocate('overflow', 'uint32', 1).view}
        }
      })
    );
    return {
      outputs,
      update: current =>
        parameterBuffer.write(
          getGPUPermutationParameterValues(
            {seed: 7, permutationCount: current.permutationCount},
            {maximumPermutations: MAXIMUM_PERMUTATIONS}
          )
        )
    };
  },
  rebuildOn: ['period', 'variant', 'lag'],
  async afterRun({inputs, outputs, values, upload}) {
    const {cube} = inputs;
    const significant = (await readRunBuffer(outputs.significant)) as Uint32Array;
    const observed = (await readRunBuffer(outputs.observed)) as Float32Array;
    const quadrants = (await readRunBuffer(outputs.quadrants)) as Uint32Array;
    const clusters = new Uint32Array(cube.cellCount);
    for (let cell = 0; cell < cube.cellCount; cell++) {
      const quadrant = getClass(cube, values, cell, observed[cell], quadrants[cell]);
      clusters[cell] = significant[cell] === 1 && quadrant > 0 ? quadrant : NOT_SIGNIFICANT;
    }
    return {clusters: upload('clusters', 'uint32', clusters)};
  },
  interaction: {
    kind: 'probe',
    hint: 'hover a cell',
    rest: [0.37, 0.45],
    resolve(pointer, {inputs, outputs, values, peek, pixelSize, world}) {
      const {cube} = inputs;
      const hit = getCellAt(cube, pointer.x, pointer.y);
      const cell = hit.index;
      inputs.focus = cell;
      const earlier = (values.period - values.lag) * cube.cellCount;
      const glyphs: RunGlyph[] = [];
      const focusBox = getCellBox(cube, cell);
      glyphs.push({shape: 'box', ...focusBox});
      const slotEnd = cube.weights.offsets[cell + 1];
      for (let slot = cube.weights.offsets[cell]; slot < slotEnd; slot++) {
        const neighbor = cube.weights.neighbors[slot];
        glyphs.push({shape: 'box', ...getCellBox(cube, neighbor), dashed: true});
        glyphs.push({
          shape: 'ring',
          at: getCellCenter(cube, neighbor),
          radius: Math.max(pixelSize * 2, getCountRadius(cube, cube.counts[earlier + neighbor]))
        });
      }
      const observed = peek(outputs.observed)?.[cell];
      const pseudoP = peek(outputs.pseudoPValues)?.[cell];
      const significant = peek(outputs.significant)?.[cell];
      const quadrants = peek(outputs.quadrants);
      // Timeline strip marks: the analysed period solid, the lagged period dashed.
      const stripLeft = STRIP[0] * world.width;
      const slotWidth = ((STRIP[2] - STRIP[0]) * world.width) / cube.periodCount;
      const getSlotX = (period: number) => stripLeft + (period + 0.5) * slotWidth;
      glyphs.push(
        {
          shape: 'line',
          from: [getSlotX(values.period), STRIP[1]],
          to: [getSlotX(values.period), STRIP[3]]
        },
        {
          shape: 'line',
          from: [getSlotX(values.period - values.lag), STRIP[1]],
          to: [getSlotX(values.period - values.lag), STRIP[3]],
          dashed: true
        }
      );
      if (observed === undefined || pseudoP === undefined || !quadrants) {
        return {glyphs, spotlight: null, readout: '…'};
      }
      const quadrant = getClass(cube, values, cell, observed, quadrants[cell]);
      const label = significant === 1 ? CLASS_NAMES[quadrant] : CLASS_NAMES[0];
      return {
        glyphs,
        spotlight: null,
        readout: `${label} · I ${observed.toFixed(1)} · p ${pseudoP.toFixed(3)}`
      };
    }
  },
  draw: ({inputs, outputs, extra, theme}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.counts,
      columns: inputs.cube.columns,
      rows: inputs.cube.rows,
      ramp: 'grey',
      range: [0, 4]
    },
    ...(extra
      ? [
          {
            layer: 'raster' as const,
            values: extra.clusters,
            columns: inputs.cube.columns,
            rows: inputs.cube.rows,
            // High-High amber (the signal color), Low-Low blue, Low-High teal, High-Low red.
            categories: [
              theme.muted,
              theme.categories[1],
              theme.categories[3],
              theme.categories[0],
              theme.categories[2]
            ]
          }
        ]
      : []),
    getInsetBackingLayer(outputs.backing, theme),
    {
      layer: 'chart',
      kind: 'bars',
      context: true,
      values: outputs.totals,
      range: [0, Math.max(...inputs.periodTotals)],
      rectangle: [STRIP[0] + 0.005, STRIP[1] + 0.01, STRIP[2] - 0.005, STRIP[3] - 0.01]
    }
  ]
});
