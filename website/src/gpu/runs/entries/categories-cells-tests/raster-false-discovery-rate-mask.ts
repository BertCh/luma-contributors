// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterFalseDiscoveryRateMask on real p-values: GPUPettittTest runs on the Dixie fire NDVI stack
// (16 Sentinel-2 scenes) in the same graph and its per-cell p-values are the raster that gets
// corrected. Swipe: left of the curtain is every cell with raw p <= alpha, right of it only the cells
// that survive the correction. The bars are the histogram of the raw p-values (a spike at 0 is
// signal); the dashed line is the critical p-value.

import {
  getGPURasterFalseDiscoveryRateMaskParameterValues,
  GPUPettittTest,
  GPURasterFalseDiscoveryRateMask,
  GPU_RASTER_FALSE_DISCOVERY_RATE_MASK_PARAMETER_LENGTH,
  type GPURasterFalseDiscoveryRateMethod
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {MAP_SHARE} from './_lib/tract-map';
import {
  CUBE_CELL_COUNT,
  CUBE_ROWS,
  createDixieCube,
  DISPLAY_COLUMNS,
  padToDisplay
} from './_lib/dixie-cube';
import {readFloats, readUints} from './_lib/readback';
import {parameters} from './raster-false-discovery-rate-mask.parameters';

const METHODS: GPURasterFalseDiscoveryRateMethod[] = [
  'benjamini-hochberg',
  'benjamini-yekutieli',
  'holm',
  'bonferroni'
];
const SLICE_COUNT = 16;
const BIN_COUNT = 20;
const CHART = [0.66, 0.14, 0.97, 0.86] as const;
const DEFAULT_CURTAIN = 0.5;

export default defineOperationRun({
  parameters,
  caption:
    'p-values of a Pettitt test per cell on Dixie fire NDVI (16 real scenes); left raw, right corrected.',
  createInputs: ({world}) => ({
    cube: createDixieCube(SLICE_COUNT),
    curtain: world.width * MAP_SHARE * DEFAULT_CURTAIN,
    counts: {raw: 0, corrected: 0}
  }),
  build(builder, inputs, values) {
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_RASTER_FALSE_DISCOVERY_RATE_MASK_PARAMETER_LENGTH
    );
    const pValues = builder.allocate('p-values', 'float32', CUBE_CELL_COUNT);
    const significant = builder.allocate('significant', 'uint32', CUBE_CELL_COUNT);
    const significantCount = builder.allocate('significant-count', 'uint32', 1);
    const criticalPValue = builder.allocate('critical-p-value', 'float32', 1);
    const histogram = builder.allocate('histogram', 'uint32', BIN_COUNT);
    builder.graph.add(
      new GPUPettittTest({
        values: builder.upload('cube', 'float32', inputs.cube).view,
        layout: 'cell-major',
        cellCount: CUBE_CELL_COUNT,
        sliceCount: SLICE_COUNT,
        output: {
          statistic: builder.allocate('statistic', 'uint32', CUBE_CELL_COUNT).view,
          changeSlice: builder.allocate('change-slice', 'uint32', CUBE_CELL_COUNT).view,
          pValues: pValues.view
        }
      })
    );
    builder.graph.add(
      new GPURasterFalseDiscoveryRateMask({
        cellCount: CUBE_CELL_COUNT,
        pValues: pValues.view,
        method: METHODS[values.method],
        parameters: parameterBuffer.view,
        output: {
          significant: significant.view,
          significantCount: significantCount.view,
          criticalPValue: criticalPValue.view,
          histogram: histogram.view
        }
      })
    );
    const firstScene = Float32Array.from(
      {length: CUBE_CELL_COUNT},
      (_, cell) => inputs.cube[cell * SLICE_COUNT]
    );
    return {
      outputs: {
        pValues,
        significant,
        significantCount,
        criticalPValue,
        histogram,
        context: builder.upload('context', 'float32', padToDisplay(firstScene))
      },
      update: current =>
        parameterBuffer.write(
          getGPURasterFalseDiscoveryRateMaskParameterValues({alpha: current.alpha})
        )
    };
  },
  rebuildOn: ['method'],
  buildCache: 4,
  async afterRun({inputs, outputs, values, upload}) {
    const mask = await readUints(outputs.significant);
    const p = await readFloats(outputs.pValues);
    const bins = await readUints(outputs.histogram);
    const raw = Float32Array.from(p, value => (value <= values.alpha ? 1 : 0));
    inputs.counts = {
      raw: raw.reduce((sum, value) => sum + value, 0),
      corrected: mask.reduce((sum, value) => sum + value, 0)
    };
    const tallest = Math.max(...bins, 1);
    return {
      raw: upload('raw', 'float32', padToDisplay(raw)),
      mask: upload('mask', 'float32', padToDisplay(Float32Array.from(mask))),
      bars: upload(
        'bars',
        'float32',
        Float32Array.from(bins, count => count / tallest)
      )
    };
  },
  interaction: {
    kind: 'swipe',
    hint: 'swipe raw | corrected',
    rest: [MAP_SHARE * DEFAULT_CURTAIN, 0.5],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const curtain = Math.min(pointer.x, world.width * MAP_SHARE);
      const critical = peek(outputs.criticalPValue)?.[0];
      const x =
        (CHART[0] +
          (Number.isFinite(critical) ? (critical as number) : 0) * (CHART[2] - CHART[0])) *
        world.width;
      return {
        curtain,
        glyphs: [
          {shape: 'curtain', x: curtain},
          // With nothing significant the critical p-value is NaN: draw no rule.
          ...(Number.isFinite(critical)
            ? [
                {
                  shape: 'line' as const,
                  from: [x, CHART[1] * world.height] as [number, number],
                  to: [x, CHART[3] * world.height] as [number, number],
                  dashed: true
                }
              ]
            : [])
        ],
        spotlight: null,
        readout: `alpha ${values.alpha.toFixed(2)} · raw ${inputs.counts.raw} | corrected ${inputs.counts.corrected} · critical p ${
          critical !== undefined && Number.isFinite(critical) ? critical.toFixed(4) : '–'
        }`
      };
    }
  },
  draw: ({outputs, extra, interaction, world}) => {
    const curtain = interaction.curtain ?? world.width * MAP_SHARE * DEFAULT_CURTAIN;
    return [
      {
        layer: 'raster',
        context: true,
        values: outputs.context,
        columns: DISPLAY_COLUMNS,
        rows: CUBE_ROWS,
        ramp: 'grey',
        range: [0, 1]
      },
      {
        layer: 'raster',
        values: extra.raw,
        columns: DISPLAY_COLUMNS,
        rows: CUBE_ROWS,
        ramp: 'sequential',
        range: [0, 1],
        fade: true,
        clip: {side: 'left', x: curtain}
      },
      {
        layer: 'raster',
        values: extra.mask,
        columns: DISPLAY_COLUMNS,
        rows: CUBE_ROWS,
        ramp: 'sequential',
        range: [0, 1],
        fade: true,
        clip: {side: 'right', x: curtain}
      },
      {
        layer: 'chart',
        kind: 'bars',
        values: extra.bars,
        ramp: 'sequential',
        range: [0, 1],
        rectangle: CHART
      }
    ];
  }
});
