// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeSampleTimestampColumn} from './time-runtime';

export const DATASET_ID = 'resident-observations';
export const CATEGORY_LABELS = ['North', 'East', 'South', 'West'];

export type SampleData = {
  longitude: Float32Array;
  latitude: Float32Array;
  positions: Float32Array;
  targetLongitude: Float32Array;
  targetLatitude: Float32Array;
  targetPositions: Float32Array;
  value: Float32Array;
  category: Uint32Array;
  time: Uint32Array;
  timestamp: ReturnType<typeof makeSampleTimestampColumn>;
  colors: Uint8Array;
  rows: number[][];
};

/** Deterministic local data: generated once, never rebuilt by interactions. */
export function createSampleData(rowCount: number): SampleData {
  const longitude = new Float32Array(rowCount);
  const latitude = new Float32Array(rowCount);
  const positions = new Float32Array(rowCount * 2);
  const targetLongitude = new Float32Array(rowCount);
  const targetLatitude = new Float32Array(rowCount);
  const targetPositions = new Float32Array(rowCount * 2);
  const value = new Float32Array(rowCount);
  const category = new Uint32Array(rowCount);
  const time = new Uint32Array(rowCount);
  const colors = new Uint8Array(rowCount * 4);
  const rows: number[][] = [];
  const palette = [
    [86, 203, 249],
    [255, 183, 77],
    [217, 121, 212],
    [116, 216, 161]
  ];
  let randomState = 42;
  const random = () => {
    randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0;
    return randomState / 4294967296;
  };
  for (let rowIndex = 0; rowIndex < rowCount; rowIndex++) {
    const group = rowIndex % 4;
    longitude[rowIndex] = -74.02 + (group % 2) * 0.065 + (random() - 0.5) * 0.055;
    latitude[rowIndex] = 40.72 + Math.floor(group / 2) * 0.065 + (random() - 0.5) * 0.055;
    positions[rowIndex * 2] = longitude[rowIndex];
    positions[rowIndex * 2 + 1] = latitude[rowIndex];
    targetLongitude[rowIndex] = longitude[rowIndex] + 0.025;
    targetLatitude[rowIndex] = latitude[rowIndex] + 0.015;
    targetPositions[rowIndex * 2] = targetLongitude[rowIndex];
    targetPositions[rowIndex * 2 + 1] = targetLatitude[rowIndex];
    value[rowIndex] = Math.fround(random() * 100);
    category[rowIndex] = group;
    time[rowIndex] = Math.floor(random() * 1440);
    colors.set([...palette[group], 200], rowIndex * 4);
    rows.push([
      longitude[rowIndex],
      latitude[rowIndex],
      value[rowIndex],
      group,
      time[rowIndex],
      targetLongitude[rowIndex],
      targetLatitude[rowIndex]
    ]);
  }
  const timestamp = makeSampleTimestampColumn(time);
  rows.forEach((row, index) => row.push(timestamp.values[index]));
  return {
    longitude,
    latitude,
    positions,
    targetLongitude,
    targetLatitude,
    targetPositions,
    value,
    category,
    time,
    timestamp,
    colors,
    rows
  };
}

export const SAMPLE_FIELDS = [
  {name: 'longitude', format: '', type: 'real'},
  {name: 'latitude', format: '', type: 'real'},
  {name: 'value', format: '', type: 'real'},
  {name: 'category', format: '', type: 'integer'},
  {name: 'time', format: '', type: 'integer'},
  {name: 'targetLongitude', format: '', type: 'real'},
  {name: 'targetLatitude', format: '', type: 'real'},
  {name: 'timestamp', format: 'x', type: 'timestamp'}
];
