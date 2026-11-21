// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {addStopEventGathers} from './stop-event-gathers';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUTrajectoryStaypoints} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-staypoints';
import {GPUPeriodIndex} from '../../../src/dataframe-analysis/period-index/gpu-period-index';
import {GPUPointToCell} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-point-to-cell';
import {GPUCellAggregation} from '../../../src/gpu-spatial-analysis/cell-aggregation/gpu-cell-aggregation';
import {GraphHarness} from './movement-spec-utils';
import {readUint32} from '../../utils/gpu-contributor-test-utils';

it('gathers stop event times and live masks so spare capacity contributes no cell', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new GraphHarness(device, 'stop-gather');
  const positions = harness.input(
    'positions',
    Float32Array.of(12, 30, 12, 30, 12, 30),
    'float32x2',
    3
  );
  const timestamps = harness.input('times', Uint32Array.of(100, 0, 102, 0, 104, 0), 'uint32x2', 3);
  const starts = harness.output('starts', 'uint32', 3);
  const centers = harness.output('centers', 'float32x2', 3);
  const ids = harness.output('ids', 'uint32', 3);
  const count = harness.output('count', 'uint32', 1);
  const overflow = harness.output('overflow', 'uint32', 1);
  harness.graph.add(
    new GPUTrajectoryStaypoints({
      positions,
      timestamps,
      trackOffsets: harness.input('offsets', Uint32Array.of(0, 3), 'uint32', 2),
      parameters: harness.input('parameters', Float32Array.of(1, 1, 0, 1), 'float32', 4),
      output: {
        ids: ids.view,
        startRows: starts.view,
        centroids: centers.view,
        status: {
          count: count.view,
          candidateOverflow: harness.output('candidateOverflow', 'uint32', 1).view,
          overflow: overflow.view
        }
      }
    })
  );
  const eventTimes = harness.output('event-times', 'uint32x2', 3);
  const live = harness.output('live', 'uint32', 3);
  addStopEventGathers(
    harness.graph,
    timestamps,
    harness.input('fix-ones', Uint32Array.of(1, 1, 1), 'uint32', 3),
    starts.view,
    eventTimes.view,
    live.view
  );
  const periods = harness.output('periods', 'uint32', 3);
  harness.graph.add(
    new GPUPeriodIndex({
      timestamps: eventTimes.view,
      mask: live.view,
      temporalContext: {
        encoding: 'int64-words',
        unit: 'second',
        epoch: 'unix',
        axis: {kind: 'irregular'}
      },
      periods: {kind: 'fixed', width: 10, origin: 100n, count: 1},
      output: {periods: periods.view}
    })
  );
  const keys = harness.input('keys', new Uint32Array(6), 'uint32x2', 3);
  harness.graph.add(
    new GPUPointToCell({
      family: 'quadbin',
      resolution: 6,
      positions: centers.view,
      mask: live.view,
      output: {cells: keys}
    })
  );
  const cellKeys = harness.input('cell-keys', new Uint32Array(6), 'uint32x2', 3);
  const cellCounts = harness.output('cell-counts', 'uint32', 3);
  const cellCount = harness.output('cell-count', 'uint32', 1);
  harness.graph.add(
    new GPUCellAggregation({
      family: 'quadbin',
      resolution: 6,
      cells: keys,
      mask: live.view,
      output: {
        cells: cellKeys,
        counts: cellCounts.view,
        status: {count: cellCount.view, overflow: harness.output('cell-overflow', 'uint32', 1).view}
      }
    })
  );
  await harness.run(async () => {
    expect(Array.from(await readUint32(eventTimes.buffer, 6))).toEqual([100, 0, 0, 0, 0, 0]);
    expect(Array.from(await readUint32(starts.buffer, 3))).toEqual([0, 0xffffffff, 0xffffffff]);
    expect(Array.from(await readUint32(live.buffer, 3))).toEqual([1, 0, 0]);
    expect(Array.from(await readUint32(periods.buffer, 3))).toEqual([0, 0xffffffff, 0xffffffff]);
    expect(Array.from(await readUint32(cellCount.buffer, 1))).toEqual([1]);
    expect(Array.from(await readUint32(cellCounts.buffer, 3))).toEqual([1, 0, 0]);
  });
});
