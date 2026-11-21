// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandNodeProducer, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPURasterLaggedCorrelation} from '../../../src/raster-analysis/raster-time-series/gpu-raster-lagged-correlation';
import {GPURasterMedoidComposite} from '../../../src/raster-analysis/raster-time-series/gpu-raster-medoid-composite';
import {GPURasterPhenologyMetrics} from '../../../src/raster-analysis/raster-time-series/gpu-raster-phenology-metrics';
import {GPURasterRollingMean} from '../../../src/raster-analysis/raster-time-series/gpu-raster-rolling-mean';
import {GPURasterStandardizedIndex} from '../../../src/raster-analysis/raster-time-series/gpu-raster-standardized-index';
import {
  GPURasterStandardizedIndexCalibration,
  getGPURasterStandardizedIndexCalibrationParameterValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-standardized-index-calibration';
import {
  GPURasterStreamingAccumulator,
  getGPURasterStreamingAccumulatorParameterValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-streaming-accumulator';
import {GPURasterTemporalResample} from '../../../src/raster-analysis/raster-time-series/gpu-raster-temporal-resample';
import {GPURasterWhittakerSmoother} from '../../../src/raster-analysis/raster-time-series/gpu-raster-whittaker-smoother';
import {GPURasterTemporal} from '../../../src/raster-analysis/remote-sensing/gpu-raster-temporal';
import {GPURasterSeasonalBreak} from '../../../src/raster-analysis/remote-sensing/gpu-raster-seasonal-break';
import {GPUChangeDetection} from '../../../src/raster-analysis/change-detection/gpu-change-detection';
import type {RasterTimeSeriesView} from '../../../src/raster-analysis/raster-time-series/raster-time-series-chunks-b';
import type {GPURasterIrregularDateAxis} from '../../../src/raster-analysis/raster-time-series/raster-irregular-date-axis';

export type ChunkCaseHarness = {
  cellCount: number;
  field<Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    entriesPerCell: number,
    value?: (cell: number, entry: number) => number
  ): RasterTimeSeriesView<Format>;
  shared(values: ArrayLike<number>): GraphDataView<'float32'>;
  dates(): GPURasterIrregularDateAxis;
};
export type ChunkCase = {
  contributor: GPUCommandNodeProducer & {estimatedCommandNodeCount: number};
  expected: Record<string, number[]>;
};
export const RASTER_TIME_SERIES_CHUNK_CASE_NAMES = [
  'lagged',
  'medoid',
  'phenology',
  'rolling',
  'rolling-short',
  'rolling-short-mask',
  'standardized',
  'calibration',
  'streaming',
  'resample',
  'whittaker',
  'temporal-mean',
  'temporal-median',
  'temporal-sg',
  'seasonal',
  'seasonal-extended',
  'change'
] as const;
export type RasterTimeSeriesChunkCaseName = (typeof RASTER_TIME_SERIES_CHUNK_CASE_NAMES)[number];

/** Closed-form cases shared by the wiring and GPU tests; no expected value uses a GPU route. */
export function createRasterTimeSeriesChunkCase(
  name: RasterTimeSeriesChunkCaseName,
  harness: ChunkCaseHarness
): ChunkCase {
  const {cellCount} = harness;
  const field = harness.field;
  const shared = harness.shared;
  const cells = (entries: number, value: (cell: number, entry: number) => number) =>
    Array.from({length: cellCount * entries}, (_, index) =>
      value(Math.floor(index / entries), index % entries)
    );
  const status = () => field('status', 'uint32', 1);
  const zeros = cells(1, () => 0);
  const values = () => field('values', 'float32', 5, (cell, entry) => cell + entry + 1);
  const mask = () => field('mask', 'uint32', 5, () => 1);
  switch (name) {
    case 'lagged':
      return {
        contributor: new GPURasterLaggedCorrelation({
          id: name,
          cellCount,
          sliceCount: 5,
          values: values(),
          mask: mask(),
          referenceIndex: shared([1, 2, 3, 4, 5]),
          parameters: shared([3]),
          lags: [0],
          output: {
            correlations: field('correlations', 'float32', 1),
            status: status(),
            pairCount: field('pairCount', 'uint32', 1)
          }
        }),
        expected: {correlations: cells(1, () => 1), status: zeros, pairCount: cells(1, () => 5)}
      };
    case 'medoid':
      return {
        contributor: new GPURasterMedoidComposite({
          id: name,
          cellCount,
          sliceCount: 5,
          bandCount: 2,
          values: field(
            'values',
            'float32',
            10,
            (cell, entry) => cell * 10 + [0, 1, 2, 3, 9][Math.floor(entry / 2)]
          ),
          mask: mask(),
          output: {
            composite: field('composite', 'float32', 2),
            sliceIndex: field('sliceIndex', 'uint32', 1),
            status: status()
          }
        }),
        expected: {
          composite: cells(2, cell => cell * 10 + 2),
          sliceIndex: cells(1, () => 2),
          status: zeros
        }
      };
    case 'phenology':
      return {
        contributor: new GPURasterPhenologyMetrics({
          id: name,
          cellCount,
          sliceCount: 5,
          maximumSeasonCount: 1,
          smoothed: field('smoothed', 'float32', 5, (_, entry) => [0, 1, 2, 1, 0][entry]),
          parameters: shared([0.5, 0.5, 0.01, 0.5]),
          output: {
            metrics: field('metrics', 'float32', 10),
            flags: field('flags', 'uint32', 1),
            seasonCount: field('seasonCount', 'uint32', 1),
            status: status()
          }
        }),
        expected: {
          metrics: cells(10, (_, entry) => [1, 3, 2, 2, 2, 0, 2, 1, -1, 3][entry]),
          flags: zeros,
          seasonCount: cells(1, () => 1),
          status: zeros
        }
      };
    case 'rolling':
      return {
        contributor: new GPURasterRollingMean({
          id: name,
          cellCount,
          chunkSliceCount: 5,
          windowLength: 2,
          values: values(),
          mask: mask(),
          parameters: shared([1, 5, 0, 0, 100]),
          ring: field('ring', 'float32', 2),
          seenCount: field('seenCount', 'uint32', 1),
          output: field('rolling', 'float32', 5)
        }),
        expected: {
          rolling: cells(5, (cell, entry) => (entry === 0 ? NaN : cell + entry + 0.5)),
          ring: cells(2, (cell, entry) => cell + (entry === 0 ? 5 : 4)),
          seenCount: cells(1, () => 5)
        }
      };
    case 'rolling-short':
    case 'rolling-short-mask':
      return {
        contributor: new GPURasterRollingMean({
          id: name,
          cellCount,
          chunkSliceCount: 2,
          windowLength: 1,
          // Short logical chunks keep the allocated pitch: cells 0/1 upload [12,pad,22,pad].
          values: field('values', 'float32', 2, (cell, entry) =>
            entry === 0 ? 12 + 10 * cell : -999
          ),
          // Both masks retain that same pitch; padding must never become cell 1's active mask.
          mask: field('mask', 'uint32', 2, (cell, entry) =>
            entry === 0 && (name === 'rolling-short' || cell % 2 === 0) ? 1 : 0
          ),
          parameters: shared(
            getGPURasterStreamingAccumulatorParameterValues(
              {reset: true, activeSliceCount: 1, maximumValueMagnitude: 10000},
              2
            )
          ),
          ring: field('ring', 'float32', 1),
          seenCount: field('seenCount', 'uint32', 1),
          output: field('rolling', 'float32', 2)
        }),
        expected: {
          rolling: cells(2, (cell, entry) =>
            entry === 0 && (name === 'rolling-short' || cell % 2 === 0) ? 12 + 10 * cell : NaN
          ),
          ring: cells(1, cell =>
            name === 'rolling-short' || cell % 2 === 0 ? 12 + 10 * cell : NaN
          ),
          seenCount: cells(1, () => 1)
        }
      };
    case 'standardized':
      return {
        contributor: new GPURasterStandardizedIndex({
          id: name,
          cellCount,
          chunkSliceCount: 5,
          distribution: 'fisk',
          values: field('values', 'float32', 5, getStandardizedCaseValue),
          mask: field('mask', 'uint32', 5, getStandardizedCaseMask),
          fit: field('fit', 'float32', 3, getStandardizedCaseFit),
          parameters: shared([4, 0]),
          output: {index: field('index', 'float32', 5), status: field('status', 'uint32', 5)}
        }),
        expected: {
          index: cells(5, (cell, entry) =>
            getStandardizedCaseStatus(cell, entry, 4) === 0
              ? STANDARDIZED_CASE_QUANTILES[(cell + entry) % 4]
              : NaN
          ),
          status: cells(5, (cell, entry) => getStandardizedCaseStatus(cell, entry, 4))
        }
      };
    case 'calibration':
      return {
        contributor: new GPURasterStandardizedIndexCalibration({
          id: name,
          cellCount,
          chunkSliceCount: 5,
          distribution: 'fisk',
          values: field('values', 'float32', 5, (_, entry) => entry + 1),
          mask: mask(),
          parameters: shared(
            getGPURasterStandardizedIndexCalibrationParameterValues(
              {reset: true, maximumValueMagnitude: 100},
              5
            )
          ),
          state: {counts: field('counts', 'uint32', 3), moments: field('moments', 'uint32', 9)},
          fit: field('fit', 'float32', 3)
        }),
        expected: {
          counts: cells(3, (_, entry) => [5, 0, 5][entry]),
          fit: cells(3, (_, entry) => [(Math.PI * 3) / Math.sqrt(6), 2.7, 0][entry])
        }
      };
    case 'streaming':
      return {
        contributor: new GPURasterStreamingAccumulator({
          id: name,
          cellCount,
          chunkSliceCount: 5,
          values: values(),
          mask: mask(),
          parameters: shared(
            getGPURasterStreamingAccumulatorParameterValues(
              {reset: true, maximumValueMagnitude: cellCount + 10},
              5
            )
          ),
          state: {
            counts: field('counts', 'uint32', 1),
            sums: field('sums', 'uint32', 3),
            moments: field('moments', 'uint32', 3),
            extrema: field('extrema', 'float32', 2)
          }
        }),
        expected: {
          counts: cells(1, () => 5),
          extrema: cells(2, (cell, entry) => cell + (entry === 0 ? 1 : 5))
        }
      };
    case 'resample':
      return {
        contributor: new GPURasterTemporalResample({
          id: name,
          cellCount,
          sliceCount: 5,
          gridCount: 5,
          values: field('values', 'float32', 5, (cell, entry) => cell + [0, 1, 3, 4, 8][entry]),
          mask: mask(),
          parameters: shared([-4, 2, 1e30]),
          dateAxis: harness.dates(),
          output: {values: field('resampled', 'float32', 5), status: field('status', 'uint32', 5)}
        }),
        expected: {
          resampled: cells(5, (cell, entry) => cell + entry * 2),
          status: cells(5, () => 0)
        }
      };
    case 'whittaker':
      return {
        contributor: new GPURasterWhittakerSmoother({
          id: name,
          cellCount,
          sliceCount: 5,
          values: field('values', 'float32', 5, cell => cell + 2),
          mask: mask(),
          qualityWeights: field('weights', 'float32', 5, () => 1),
          parameters: shared([1, 0.01]),
          reweightPassCount: 0,
          dateAxis: harness.dates(),
          output: {smoothed: field('smoothed', 'float32', 5), status: status()}
        }),
        expected: {smoothed: cells(5, cell => cell + 2), status: zeros}
      };
    case 'temporal-mean':
    case 'temporal-median':
    case 'temporal-sg': {
      const smoothing = name === 'temporal-sg';
      const entries = smoothing ? 5 : 1;
      return {
        contributor: new GPURasterTemporal({
          id: name,
          cellCount,
          sliceCount: 5,
          values: smoothing
            ? field('values', 'float32', 5, (cell, entry) => cell + 3 + 2 * [0, 1, 3, 4, 8][entry])
            : values(),
          mask: mask(),
          operation: smoothing ? 'savitzky-golay' : name === 'temporal-mean' ? 'mean' : 'median',
          windowLength: 3,
          polynomialOrder: 1,
          ...(smoothing ? {dateAxis: harness.dates()} : {}),
          output: {
            values: field('temporal', 'float32', entries),
            counts: field('counts', 'uint32', entries),
            chosenSlice: field('chosenSlice', 'uint32', entries),
            status: field('status', 'uint32', entries)
          }
        }),
        expected: {
          temporal: cells(entries, (cell, entry) =>
            smoothing ? cell + 3 + 2 * [0, 1, 3, 4, 8][entry] : cell + 3
          ),
          counts: cells(entries, () => (smoothing ? 3 : 5)),
          chosenSlice: cells(entries, () => (name === 'temporal-median' ? 2 : 0xffffffff)),
          status: cells(entries, () => 0)
        }
      };
    }
    case 'seasonal':
    case 'seasonal-extended':
      return {
        contributor: new GPURasterSeasonalBreak({
          id: name,
          cellCount,
          sliceCount: 12,
          values: field(
            'values',
            'float32',
            12,
            (cell, entry) =>
              cell +
              100 +
              2 * entry +
              3 * Math.cos((Math.PI * entry) / 2) +
              4 * Math.sin((Math.PI * entry) / 2)
          ),
          mask: field('mask', 'uint32', 12, () => 1),
          period: 4,
          minimumSegmentLength: 3,
          minimumBICGain: 1e6,
          ...(name === 'seasonal-extended' ? {fitMethod: 'joint' as const} : {}),
          output: {
            coefficients: field('coefficients', 'float32', 5),
            components: field('components', 'float32', 36),
            breakIndex: field('breakIndex', 'uint32', 1),
            improvement: field('improvement', 'float32', 1),
            status: status()
          }
        }),
        expected: {
          coefficients: cells(5, (cell, entry) => [100 + cell, 24, 3, 4, 0][entry]),
          breakIndex: cells(1, () => 0xffffffff),
          improvement: zeros,
          status: zeros
        }
      };
    case 'change':
      return {
        contributor: new GPUChangeDetection({
          id: name,
          cellCount,
          sliceCount: 5,
          slices: values(),
          observationMask: mask(),
          mask: field('cellMask', 'uint32', 1, () => 1),
          parameters: shared([0, 4, 1e-6, 0.05, 2]),
          output: {
            difference: field('difference', 'float32', 1),
            tStatistic: field('tStatistic', 'float32', 1),
            tDegreesOfFreedom: field('degrees', 'float32', 1),
            senSlope: field('slope', 'float32', 1),
            mannKendallZ: field('mannKendallZ', 'float32', 1)
          }
        }),
        expected: {
          difference: cells(1, () => 4),
          tStatistic: cells(1, () => 2.5 / Math.sqrt(7 / 12)),
          degrees: cells(1, () => 49 / 17),
          slope: cells(1, () => 1),
          mannKendallZ: cells(1, () => 9 / Math.sqrt(50 / 3))
        }
      };
  }
}

// Independent standard-normal CDF values at the chosen quantiles (not the GPU quantile implementation).
export const STANDARDIZED_CASE_QUANTILES = [-1, 0.5, 1.5, -0.75] as const;
const STANDARDIZED_CASE_PROBABILITIES = [
  0.15865525393145707, 0.6914624612740131, 0.9331927987311419, 0.2266273523768682
] as const;
export function getStandardizedCaseFit(cell: number, entry: number): number {
  return entry === 0 ? (cell % 7 === 6 ? NaN : 2 + (cell % 3)) : entry === 1 ? 0.8 + cell * 0.1 : 0;
}
export function getStandardizedCaseValue(cell: number, slice: number): number {
  if (slice % 47 === 2) return NaN;
  const probability = STANDARDIZED_CASE_PROBABILITIES[(cell + slice) % 4];
  return (0.8 + cell * 0.1) * (probability / (1 - probability)) ** (1 / (2 + (cell % 3)));
}
export function getStandardizedCaseMask(_cell: number, slice: number): number {
  return slice % 17 === 1 ? 0 : 1;
}
export function getStandardizedCaseStatus(
  cell: number,
  slice: number,
  activeSliceCount: number
): number {
  if (
    slice >= activeSliceCount ||
    getStandardizedCaseMask(cell, slice) === 0 ||
    !Number.isFinite(getStandardizedCaseValue(cell, slice))
  )
    return 1;
  return cell % 7 === 6 ? 2 : 0;
}
