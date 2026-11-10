// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import dixie from './rasters/dixie-spectral.bin?url&inline';
import {decodeRaster, sampleGrid} from './sample-grid';

let reflectance: Uint16Array | undefined;

/** Sentinel-2 L2A surface reflectance, Feather River watershed on July 13, 2021. */
export function sampleReflectance(columns: number, rows: number) {
  reflectance ??= new Uint16Array(decodeRaster(dixie).buffer);
  const data = reflectance;
  const band = (index: number) =>
    sampleGrid(data, 64, 40, columns, rows, index * 64 * 40).map(value => value / 4095);
  return {
    blue: band(0),
    green: band(1),
    red: band(2),
    nearInfrared: band(3),
    shortwaveInfrared: band(4),
    shortwaveInfrared2: band(5)
  };
}
