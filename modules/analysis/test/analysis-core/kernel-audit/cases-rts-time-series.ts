// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUChangeDetection,
  GPURasterClimateAnomaly,
  GPURasterClimateRunAccumulator,
  GPURasterHANTS,
  GPURasterRollingMean,
  GPURasterSeasonalBreak,
  GPURasterStandardizedIndex,
  GPURasterStandardizedIndexCalibration,
  GPURasterStreamingAccumulator,
  GPURasterTemporal,
  GPURasterTemporalResample,
  GPURasterWhittakerSmoother,
  type GPURasterIrregularDateAxis
} from '@luma-contributors/analysis/raster-analysis';
import {MAXIMUM_STANDARDIZED_INDEX_CHUNK_SLICE_COUNT} from '../../../src/raster-analysis/raster-time-series/gpu-raster-standardized-index';
import type {GPUTemporalContext} from '../../../src/analysis-core/contracts/index';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

const CONTEXT: GPUTemporalContext = {
  unit: 'nanosecond',
  epoch: 'unix',
  encoding: 'int64-words',
  axis: {kind: 'irregular'}
};

const CELLS = 4096;

function makeAxis({view}: KernelAuditContext, slices: number): GPURasterIrregularDateAxis {
  return {acquisitionTimes: view('uint32x2', slices), temporalContext: CONTEXT};
}

function whittaker(
  context: KernelAuditContext,
  order: number,
  passes: number,
  reweightFunction: 'bisquare' | 'upper-envelope',
  withAxis: boolean,
  sliceCount = 256
): void {
  const {graph, view} = context;
  graph.add(
    context.construct(GPURasterWhittakerSmoother, {
      cellCount: CELLS,
      sliceCount,
      differenceOrder: order,
      reweightPassCount: passes,
      reweightFunction,
      values: view('float32', CELLS * sliceCount),
      mask: view('uint32', CELLS * sliceCount),
      qualityWeights: view('float32', CELLS * sliceCount),
      parameters: view('float32', 2),
      dateAxis: withAxis ? makeAxis(context, sliceCount) : undefined,
      output: {smoothed: view('float32', CELLS * sliceCount), status: view('uint32', CELLS)}
    })
  );
}

function hants(
  context: KernelAuditContext,
  slices: number,
  harmonicCount: number,
  passes = 16
): void {
  const {graph, view} = context;
  graph.add(
    context.construct(GPURasterHANTS, {
      cellCount: CELLS,
      sliceCount: slices,
      harmonicCount,
      rejectionPassCount: passes,
      rejectionSide: 'both',
      values: view('float32', CELLS * slices),
      mask: view('uint32', CELLS * slices),
      parameters: view('float32', 3),
      dateAxis: makeAxis(context, slices),
      output: {
        smoothed: view('float32', CELLS * slices),
        coefficients: view('float32', CELLS * (2 * harmonicCount + 1)),
        status: view('uint32', CELLS),
        diagnostics: view('uint32', CELLS * 2)
      }
    })
  );
}

function streamingAccumulator(
  context: KernelAuditContext,
  chunkSliceCount: number,
  slotCount: number
): void {
  const {graph, view} = context;
  const entries = CELLS * slotCount;
  graph.add(
    context.construct(GPURasterStreamingAccumulator, {
      cellCount: CELLS,
      chunkSliceCount,
      slotCount,
      values: view('float32', CELLS * chunkSliceCount),
      mask: view('uint32', CELLS * chunkSliceCount),
      slotIndices: view('uint32', chunkSliceCount),
      parameters: view('float32', 5),
      state: {
        counts: view('uint32', entries),
        sums: view('uint32', entries * 3),
        moments: view('uint32', entries * 3),
        extrema: view('float32', entries * 2)
      }
    })
  );
}

function calibration(
  context: KernelAuditContext,
  distribution: 'gamma' | 'fisk',
  zeroAtom: boolean,
  chunkSliceCount = 256
): void {
  const {graph, view} = context;
  const slotCount = 12;
  const entries = CELLS * slotCount;
  graph.add(
    context.construct(GPURasterStandardizedIndexCalibration, {
      cellCount: CELLS,
      chunkSliceCount,
      slotCount,
      distribution,
      zeroAtom,
      values: view('float32', CELLS * chunkSliceCount),
      slotIndices: view('uint32', chunkSliceCount),
      parameters: view('float32', 7),
      state: {counts: view('uint32', entries * 3), moments: view('uint32', entries * 9)},
      fit: view('float32', entries * 3)
    })
  );
}

function standardizedIndex(
  context: KernelAuditContext,
  distribution: 'gamma' | 'fisk',
  zeroAtom: boolean,
  chunkSliceCount = 256,
  cellCount = CELLS
): void {
  const {graph, view} = context;
  const slotCount = 12;
  graph.add(
    context.construct(GPURasterStandardizedIndex, {
      cellCount,
      chunkSliceCount,
      slotCount,
      distribution,
      zeroAtom,
      values: view('float32', cellCount * chunkSliceCount),
      slotIndices: view('uint32', chunkSliceCount),
      fit: view('float32', cellCount * slotCount * 3),
      parameters: view('float32', 2),
      output: {
        index: view('float32', cellCount * chunkSliceCount),
        status: view('uint32', cellCount * chunkSliceCount)
      }
    })
  );
}

function runAccumulator(
  context: KernelAuditContext,
  condition: 'above' | 'at-or-below',
  chunkSliceCount = 256
): void {
  const {graph, view} = context;
  graph.add(
    context.construct(GPURasterClimateRunAccumulator, {
      cellCount: CELLS,
      chunkSliceCount,
      condition,
      minimumSpellLength: 3,
      values: view('float32', CELLS * chunkSliceCount),
      mask: view('uint32', CELLS * chunkSliceCount),
      parameters: view('float32', 4),
      state: {runs: view('uint32', CELLS * 6), degreeDays: view('uint32', CELLS * 3)}
    })
  );
}

/** RTS-02 and RTS-06 contributors and the TC-07 irregular-date variants of the raster peers at their documented limits. */
export const RTS_TIME_SERIES_AUDIT_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPURasterStreamingAccumulator',
    name: 'tc07-maximum-chunk-all-state',
    build: context => streamingAccumulator(context, 256, 366)
  },
  {
    exportName: 'GPURasterStreamingAccumulator',
    name: 'tc07-chunk-above-maximum',
    build: context => streamingAccumulator(context, 257, 12)
  },
  {
    exportName: 'GPURasterRollingMean',
    name: 'rts06-minimum-window',
    build: ({graph, view, construct}) =>
      graph.add(
        construct(GPURasterRollingMean, {
          cellCount: 2,
          chunkSliceCount: 2,
          windowLength: 1,
          values: view('float32', 4),
          parameters: view('float32', 5),
          ring: view('float32', 2),
          seenCount: view('uint32', 2),
          output: view('float32', 4)
        })
      )
  },
  {
    exportName: 'GPURasterRollingMean',
    name: 'rts06-maximum-window',
    build: ({graph, view, construct}) =>
      graph.add(
        construct(GPURasterRollingMean, {
          cellCount: CELLS,
          chunkSliceCount: 256,
          windowLength: 128,
          values: view('float32', CELLS * 256),
          parameters: view('float32', 5),
          ring: view('float32', CELLS * 128),
          seenCount: view('uint32', CELLS),
          output: view('float32', CELLS * 256)
        })
      )
  },
  {
    exportName: 'GPURasterRollingMean',
    name: 'rts06-window-above-maximum',
    build: ({graph, view, construct}) =>
      graph.add(
        construct(GPURasterRollingMean, {
          cellCount: CELLS,
          chunkSliceCount: 256,
          windowLength: 129,
          values: view('float32', CELLS * 256),
          parameters: view('float32', 5),
          ring: view('float32', CELLS * 129),
          seenCount: view('uint32', CELLS),
          output: view('float32', CELLS * 256)
        })
      )
  },
  {
    exportName: 'GPURasterWhittakerSmoother',
    name: 'rts02-maximum-slices-order3-irregular',
    build: context => whittaker(context, 3, 8, 'bisquare', true)
  },
  {
    exportName: 'GPURasterWhittakerSmoother',
    name: 'rts02-regular-upper-envelope',
    build: context => whittaker(context, 2, 3, 'upper-envelope', false)
  },
  {
    exportName: 'GPURasterWhittakerSmoother',
    name: 'rts02-slices-above-maximum',
    build: context => whittaker(context, 2, 3, 'bisquare', false, 257)
  },
  {
    exportName: 'GPURasterHANTS',
    name: 'rts02-maximum-memory-two-harmonics',
    build: context => hants(context, 256, 2)
  },
  {
    exportName: 'GPURasterHANTS',
    name: 'rts02-memory-above-maximum',
    build: context => hants(context, 256, 3)
  },
  {
    exportName: 'GPURasterTemporalResample',
    name: 'rts02-maximum-grid',
    build: context => {
      const {graph, view} = context;
      graph.add(
        context.construct(GPURasterTemporalResample, {
          cellCount: CELLS,
          sliceCount: 256,
          gridCount: 4096,
          values: view('float32', CELLS * 256),
          parameters: view('float32', 3),
          dateAxis: makeAxis(context, 256),
          output: {values: view('float32', CELLS * 4096), status: view('uint32', CELLS * 4096)}
        })
      );
    }
  },
  {
    exportName: 'GPURasterTemporalResample',
    name: 'rts02-grid-above-maximum',
    build: context => {
      const {graph, view} = context;
      graph.add(
        context.construct(GPURasterTemporalResample, {
          cellCount: CELLS,
          sliceCount: 256,
          gridCount: 4097,
          values: view('float32', CELLS * 256),
          parameters: view('float32', 3),
          dateAxis: makeAxis(context, 256),
          output: {values: view('float32', CELLS * 4097), status: view('uint32', CELLS * 4097)}
        })
      );
    }
  },
  {
    exportName: 'GPURasterStandardizedIndexCalibration',
    name: 'rts06-gamma-zero-inflated',
    build: context => calibration(context, 'gamma', true)
  },
  {
    exportName: 'GPURasterStandardizedIndexCalibration',
    name: 'rts06-fisk',
    build: context => calibration(context, 'fisk', false)
  },
  {
    exportName: 'GPURasterStandardizedIndexCalibration',
    name: 'rts06-calibration-chunk-above-maximum',
    build: context => calibration(context, 'gamma', true, 257)
  },
  {
    exportName: 'GPURasterStandardizedIndex',
    name: 'rts06-gamma-zero-inflated-index',
    build: context => standardizedIndex(context, 'gamma', true)
  },
  {
    exportName: 'GPURasterStandardizedIndex',
    name: 'rts06-fisk-index',
    build: context => standardizedIndex(context, 'fisk', false)
  },
  {
    exportName: 'GPURasterStandardizedIndex',
    name: 'rts06-index-chunk-above-maximum',
    build: context =>
      standardizedIndex(context, 'gamma', true, MAXIMUM_STANDARDIZED_INDEX_CHUNK_SLICE_COUNT + 1, 1)
  },
  {
    exportName: 'GPURasterStandardizedIndex',
    name: 'rts06-index-chunk-at-maximum',
    build: context =>
      standardizedIndex(context, 'gamma', true, MAXIMUM_STANDARDIZED_INDEX_CHUNK_SLICE_COUNT, 1)
  },
  {
    exportName: 'GPURasterStandardizedIndex',
    name: 'rts06-index-chunk-above-former-maximum',
    build: context => standardizedIndex(context, 'fisk', false, 257)
  },
  {
    exportName: 'GPURasterClimateRunAccumulator',
    name: 'rts06-above-runs-degree-days',
    build: context => runAccumulator(context, 'above')
  },
  {
    exportName: 'GPURasterClimateRunAccumulator',
    name: 'rts06-at-or-below',
    build: context => runAccumulator(context, 'at-or-below')
  },
  {
    exportName: 'GPURasterClimateRunAccumulator',
    name: 'rts06-run-chunk-above-maximum',
    build: context => runAccumulator(context, 'above', 257)
  },
  {
    exportName: 'GPURasterClimateAnomaly',
    name: 'rts06-climatology-and-standardized-anomaly',
    build: ({graph, view}) => {
      const entries = CELLS * 12;
      graph.add(
        new GPURasterClimateAnomaly({
          cellCount: CELLS,
          slotCount: 12,
          chunkSliceCount: 256,
          state: {
            counts: view('uint32', entries),
            sums: view('uint32', entries * 3),
            moments: view('uint32', entries * 3)
          },
          accumulatorParameters: view('float32', 5),
          values: view('float32', CELLS * 256),
          slotIndices: view('uint32', 256),
          standardized: true,
          output: {
            climatologyMean: view('float32', entries),
            standardDeviation: view('float32', entries),
            anomaly: view('float32', CELLS * 256)
          }
        })
      );
    }
  },
  {
    exportName: 'GPURasterTemporal',
    name: 'tc07-savitzky-golay-irregular-dates',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPURasterTemporal({
          cellCount: CELLS,
          sliceCount: 256,
          values: view('float32', CELLS * 256),
          operation: 'savitzky-golay',
          windowLength: 15,
          polynomialOrder: 3,
          dateAxis: makeAxis(context, 256),
          output: {
            values: view('float32', CELLS * 256),
            counts: view('uint32', CELLS * 256),
            chosenSlice: view('uint32', CELLS * 256),
            status: view('uint32', CELLS * 256)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUChangeDetection',
    name: 'tc07-sen-slope-irregular-dates',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUChangeDetection({
          cellCount: CELLS,
          sliceCount: 64,
          slices: view('float32', CELLS * 64),
          parameters: view('float32', 5),
          dateAxis: makeAxis(context, 64),
          output: {senSlope: view('float32', CELLS)}
        })
      );
    }
  },
  {
    exportName: 'GPURasterSeasonalBreak',
    name: 'tc07-harmonic-irregular-dates-default',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPURasterSeasonalBreak({
          cellCount: CELLS,
          sliceCount: 256,
          period: 1,
          minimumSegmentLength: 12,
          dateAxis: makeAxis(context, 256),
          values: view('float32', CELLS * 256),
          mask: view('uint32', CELLS * 256),
          output: {
            coefficients: view('float32', CELLS * 5),
            components: view('float32', CELLS * 256 * 3),
            breakIndex: view('uint32', CELLS),
            improvement: view('float32', CELLS),
            status: view('uint32', CELLS)
          }
        })
      );
    }
  },
  {
    exportName: 'GPURasterSeasonalBreak',
    name: 'tc07-harmonic-irregular-dates-extended-interpolate',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPURasterSeasonalBreak({
          cellCount: CELLS,
          sliceCount: 128,
          period: 1,
          minimumSegmentLength: 6,
          missingPolicy: 'interpolate',
          monitoring: {historyLength: 64, threshold: 3, minimumConsecutive: 2},
          dateAxis: makeAxis(context, 128),
          values: view('float32', CELLS * 128),
          output: {
            coefficients: view('float32', CELLS * 5),
            components: view('float32', CELLS * 128 * 3),
            breakIndex: view('uint32', CELLS),
            improvement: view('float32', CELLS),
            status: view('uint32', CELLS),
            diagnostics: view('float32', CELLS * 6)
          }
        })
      );
    }
  }
];
