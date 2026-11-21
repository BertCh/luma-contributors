// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUTrajectorySplitOperation} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-split-operation';
import {GPUTrajectoryThinningOperation} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-thinning-operation';
import {GPUTrajectoryTrackRelativeTimes} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-track-relative-times';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {readFloat32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {GraphHarness, importTrajectory} from '../trajectory-staypoints/movement-spec-utils';
import {poisonOperation} from './trajectory-output-oracle-utils';
import {
  OFFSETS,
  POSITIONS,
  SPLIT_TIMES,
  VALUES,
  SPLIT_POSITIONS,
  SPLIT_OUTPUT_TIMES,
  THIN_TIMES,
  THIN_POSITIONS,
  THIN_OUTPUT_TIMES,
  THIN_VALIDITY
} from './cleaning-output-oracles-fixture';

it('relative times subtract each Int64 track origin across word rollover', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new GraphHarness(device, 'relative-output-oracle');
  // Exact word subtraction: the first pair crosses a low-word rollover; the second has a
  // distinct negative origin. All differences are <2^24 and exactly representable.
  const {timestamps, trackOffsets} = importTrajectory(harness, {
    positions: new Float32Array(8),
    times: [4294967290n, 4294967300n, -100000000000n, -99999999970n],
    trackOffsets: [0, 2, 4]
  });
  const relative = harness.output('relative', 'float32', 4);
  relative.buffer.write(Float32Array.of(93, 93, 93, 93));
  harness.graph.add(
    new GPUTrajectoryTrackRelativeTimes({
      timestamps: timestamps as never,
      trackOffsets,
      output: relative.view
    })
  );
  expect(await harness.run(() => readFloat32(relative.buffer, 4))).toEqual([0, 10, 0, 30]);
});

it('split poisoned copies match literal boundary gathers with optional outputs isolated', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const optional of ['positions', 'timestamps', 'none'] as const) {
    const operation = new GPUTrajectorySplitOperation({
      mode: 'value-change',
      positions: 'positions',
      timestamps: 'times',
      values: 'values',
      trackOffsets: 'offsets',
      rowCount: 6,
      segmentCapacity: 3,
      rowCapacity: 7,
      output: {
        segmentOffsets: 'segments',
        segmentTracks: 'tracks',
        rowIds: 'rows',
        ...(optional === 'positions' ? {positions: 'copied'} : {}),
        ...(optional === 'timestamps' ? {timestamps: 'copied'} : {}),
        status: {count: 'count', overflow: 'overflow'}
      }
    });
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: POSITIONS},
        times: {format: 'float32', values: SPLIT_TIMES},
        offsets: {format: 'uint32', values: OFFSETS},
        values: {format: 'uint32', values: VALUES}
      },
      operations: [poisonOperation(operation)],
      outputs: Object.values(operation.outputs).map(output => output.value)
    });
    expect(Array.from(result['rows'])).toEqual([0, 1, 2, 2, 3, 4, 5]);
    expect(Array.from(result['segments'])).toEqual([0, 3, 5, 7]);
    expect(Array.from(result['tracks'])).toEqual([0, 0, 1]);
    expect(Array.from(result['count'])).toEqual([3]);
    expect(Array.from(result['overflow'])).toEqual([0]);
    if (optional !== 'none')
      expect(Array.from(result['copied'])).toEqual(
        optional === 'positions' ? SPLIT_POSITIONS : SPLIT_OUTPUT_TIMES
      );
  }
});

it('thinning poisoned optional ports match hand-selected rows in isolation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const optional of ['sourceSelectionMask', 'positions', 'timestamps', 'none'] as const) {
    const operation = new GPUTrajectoryThinningOperation({
      mode: 'minimum-time',
      positions: 'positions',
      timestamps: 'times',
      trackOffsets: 'offsets',
      parameters: 'parameters',
      rowCount: 6,
      trackCount: 2,
      capacity: 5,
      output: {
        sourceRows: 'rows',
        trackOffsets: 'keptOffsets',
        ...(optional === 'sourceSelectionMask' ? {sourceSelectionMask: 'copied'} : {}),
        ...(optional === 'positions' ? {positions: 'copied'} : {}),
        ...(optional === 'timestamps' ? {timestamps: 'copied'} : {}),
        status: {count: 'count', candidateOverflow: 'candidateOverflow', overflow: 'overflow'}
      }
    });
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: POSITIONS},
        times: {format: 'float32', values: THIN_TIMES},
        offsets: {format: 'uint32', values: OFFSETS},
        parameters: {format: 'float32', values: Float32Array.of(3, 0, 0, 0)}
      },
      operations: [poisonOperation(operation)],
      outputs: Object.values(operation.outputs).map(output => output.value)
    });
    expect(Array.from(result['rows'])).toEqual([0, 2, 3, 4, 5]);
    expect(Array.from(result['keptOffsets'])).toEqual([0, 3, 5]);
    expect(Array.from(result['count'])).toEqual([5]);
    expect(Array.from(result['overflow'])).toEqual([0]);
    if (optional !== 'none')
      expect(Array.from(result['copied'])).toEqual(
        optional === 'sourceSelectionMask'
          ? THIN_VALIDITY
          : optional === 'positions'
            ? THIN_POSITIONS
            : THIN_OUTPUT_TIMES
      );
  }
});
