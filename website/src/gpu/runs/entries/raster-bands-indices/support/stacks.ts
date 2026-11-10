// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The first twelve Sentinel-2 observations span 2017-09-02 through 2022-08-07.
// The July and September 2021 observations bracket the Dixie Fire (slices 7 and 8).

import {sampleNdviSeries} from '../../../../data/ndvi';
import type {Random} from '../../../inputs';

export function createStackInput(random: Random, aspect: number, rows: number) {
  const columns = Math.max(8, Math.round(rows * aspect));
  const cellCount = columns * rows;
  const dates = Array.from({length: 12}, (_, index) => index);
  const observations = sampleNdviSeries(columns, rows, dates);
  const clouds = Float32Array.from(observations, () => random());
  return {columns, rows, cellCount, sliceCount: dates.length, observations, clouds};
}
