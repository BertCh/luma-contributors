// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {prepareArrowTableForGPUAnalytics} from '../../../src/analysis-core/graph-support/arrow-analytics-normalization';

/** DPC-123: both batches retain a nonzero slice offset; only the first has nulls/a bitmap. */
export function createArrowIngressFixture() {
  const first = arrow
    .vectorFromArray(
      [new Date(999), new Date(-10), null, new Date(0), new Date(0)],
      new arrow.TimestampMillisecond()
    )
    .slice(1);
  const second = arrow
    .vectorFromArray(
      [new Date(999), new Date(10), new Date(10), new Date(20), new Date(0), new Date(999)],
      new arrow.TimestampMillisecond()
    )
    .slice(1, 5);
  const source = new arrow.Table({time: first}).concat(new arrow.Table({time: second}));
  const prepared = prepareArrowTableForGPUAnalytics(source);
  return {source, column: prepared.timeColumns['time']};
}

// Hand enumeration, not a replay of the contributors:
// source row:        0     1     2    3    4    5    6    7
// time (ms):       -10   null    0    0   10   10   20    0
// cell:             4     4     4    4    8    8    8   missing
// value:            2    999    4    8   10   14  999  999
// period [-10,20):   0   invalid 1    1    2    2   out    1
// surviving rows: [0,2,3,4,5]; keys (replace low 2 cell bits): [4,5,5,10,10].
// groups: key 4 -> [2], key 5 -> [4,8], key 10 -> [10,14], counts [1,2,2].
// dense cell-major bins: [cell4:p0,p1,p2, cell8:p0,p1,p2], sums [2,12,0,0,0,24].
export const ARROW_INGRESS_CELLS = [4n, 4n, 4n, 4n, 8n, 8n, 8n, 0xffffffffffffffffn];
export const ARROW_INGRESS_VALUES = [2, 999, 4, 8, 10, 14, 999, 999];

/** UDF-02: independent nulls, a real zero and a record-batch boundary. */
export function createNullableArrowMeasuresFixture() {
  const first = new arrow.Table({
    key: arrow.vectorFromArray([7, 7], new arrow.Uint32()),
    x: arrow.vectorFromArray([null, 0], new arrow.Float32()),
    y: arrow.vectorFromArray([12, null], new arrow.Float32())
  });
  const last = new arrow.Table({
    key: arrow.vectorFromArray([7], new arrow.Uint32()),
    x: arrow.vectorFromArray([2], new arrow.Float32()),
    y: arrow.vectorFromArray([4], new arrow.Float32())
  });
  return first.concat(last);
}
