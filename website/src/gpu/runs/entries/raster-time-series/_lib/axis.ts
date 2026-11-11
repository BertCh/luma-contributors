// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Acquisition dates for the raster time-series runs: the 16 Sentinel-2 scenes behind the Dixie NDVI
// extract, and uploads of an Int64 date axis in the form GPURasterIrregularDateAxis takes.

import {
  getGPURasterCentredFractionalYears,
  GPU_RASTER_SECONDS_PER_YEAR
} from '@luma-contributors/analysis/raster-analysis';
import type {GPUTemporalContext} from '@luma-contributors/analysis/analysis-core';
import type {RunBuilder} from '../../../define-run';

/** Acquisition dates of the 16 scenes of `data/rasters/dixie-ndvi.bin`, in scene order. */
export const DIXIE_SCENE_DATES = [
  '2017-09-02',
  '2018-07-29',
  '2018-08-08',
  '2019-08-23',
  '2020-06-03',
  '2020-08-02',
  '2021-06-03',
  '2021-07-13',
  '2021-09-11',
  '2021-09-21',
  '2022-06-23',
  '2022-08-07',
  '2023-07-13',
  '2023-08-02',
  '2024-06-22',
  '2024-08-01'
] as const;

/** Index of the first scene after the Dixie Fire started on 2021-07-13. */
export const DIXIE_FIRST_POST_FIRE_SCENE = 8;

export const SECONDS_PER_DAY = 86400;

/** Dates as Int64 seconds, the unit of {@link TEMPORAL_CONTEXT}. */
export const TEMPORAL_CONTEXT: GPUTemporalContext = {
  unit: 'second',
  epoch: 0n,
  encoding: 'int64-words',
  axis: {kind: 'irregular'}
};

export function getSceneSeconds(dates: readonly string[]): BigInt64Array {
  return BigInt64Array.from(dates, date => BigInt(Date.parse(`${date}T00:00:00Z`) / 1000));
}

/** Int64 seconds as `(low, high)` words, one `uint32x2` row per date. */
export function getTimeWords(times: BigInt64Array): Uint32Array {
  const words = new Uint32Array(times.length * 2);
  times.forEach((time, index) => {
    const unsigned = BigInt.asUintN(64, time);
    words[index * 2] = Number(unsigned & 0xffffffffn);
    words[index * 2 + 1] = Number(unsigned >> 32n);
  });
  return words;
}

/** The `dateAxis` prop: the acquisition times uploaded as a graph view. */
export function uploadDateAxis(builder: RunBuilder, times: BigInt64Array) {
  return {
    acquisitionTimes: builder.upload('acquisition-times', 'uint32x2', getTimeWords(times)).view,
    temporalContext: TEMPORAL_CONTEXT
  };
}

/** Centred fractional years of the axis, the time every contributor works in. */
export function getAxisYears(times: BigInt64Array): Float32Array {
  return getGPURasterCentredFractionalYears(times, 'second');
}

/** Calendar year with decimals of a time in Int64 seconds, for readouts. */
export function formatDate(seconds: bigint): string {
  return new Date(Number(seconds) * 1000).toISOString().slice(0, 10);
}

export {GPU_RASTER_SECONDS_PER_YEAR};
