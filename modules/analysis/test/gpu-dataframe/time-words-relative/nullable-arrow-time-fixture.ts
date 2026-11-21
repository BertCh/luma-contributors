// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {prepareArrowTableForGPUAnalytics} from '../../../src/analysis-core/graph-support/arrow-analytics-normalization';

/** Two actual Arrow batches, with a sliced nullable first batch and a genuine epoch-zero row. */
export function prepareNullableArrowTimeFixture() {
  const first = arrow.vectorFromArray([99n, 0n, null], new arrow.Int64()).slice(1);
  const second = arrow.vectorFromArray([1n], new arrow.Int64());
  const table = new arrow.Table({time: first}).concat(new arrow.Table({time: second}));
  const prepared = prepareArrowTableForGPUAnalytics(table, {
    columns: ['time'],
    timeColumns: {time: 'millisecond'}
  });
  return {table, column: prepared.timeColumns['time']};
}
