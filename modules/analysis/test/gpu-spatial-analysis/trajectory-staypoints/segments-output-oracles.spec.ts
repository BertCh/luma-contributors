// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUTrajectorySegmentsOperation} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-segments-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {poisonOperation} from '../trajectory-cleaning/trajectory-output-oracle-utils';
import {
  getOracleSegmentsInputs,
  getOracleSegmentsProps,
  getOracleSegmentOutputNames,
  expectSegmentOracleOutputs,
  SEGMENT_OPTIONALS
} from './segments-output-oracles-fixture';

it('segments poisoned outputs match the hand timeline with each optional record isolated', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const optional of SEGMENT_OPTIONALS) {
    const operation = new GPUTrajectorySegmentsOperation(getOracleSegmentsProps(optional));
    const result = await runOperationGraph(device, {
      inputs: getOracleSegmentsInputs(),
      operations: [poisonOperation(operation)],
      outputs: getOracleSegmentOutputNames(optional)
    });
    // Exact integer identities, row indices, counts and sentinels, including the capacity tail.
    expectSegmentOracleOutputs(result, getOracleSegmentOutputNames(optional));
  }
});

it('segments poisoned count and overflow ports report hand-counted capacity losses', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const operation = new GPUTrajectorySegmentsOperation({
    ...getOracleSegmentsProps('trips-only'),
    triplegCapacity: 3,
    tripCapacity: 1
  });
  const result = await runOperationGraph(device, {
    inputs: getOracleSegmentsInputs(),
    operations: [poisonOperation(operation)],
    outputs: getOracleSegmentOutputNames('trips-only')
  });
  // Five required triplegs retain three; those three form two trips, retaining one.
  expect(Array.from(result['triplegIds'])).toEqual([0, 0, 0]);
  expect(Array.from(result['triplegCount'])).toEqual([3]);
  expect(Array.from(result['triplegOverflow'])).toEqual([1]);
  expect(Array.from(result['tripIds'])).toEqual([0]);
  expect(Array.from(result['tripCount'])).toEqual([1]);
  expect(Array.from(result['tripOverflow'])).toEqual([1]);
  expect(Array.from(result['tripCandidateOverflow'])).toEqual([1]);
});
