// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUModalSplitOperation} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-modal-split-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {poisonOperation} from '../trajectory-cleaning/trajectory-output-oracle-utils';
import {getModalFixture, getModalProps} from './modal-split-operation-fixture';

it('modal poisoned speed bins match hand division with predictions present and absent', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const predictions of [true, false]) {
    const fixture = getModalFixture();
    const props = getModalProps('count', true);
    if (!predictions) delete props.output.predictedModes;
    const operation = new GPUModalSplitOperation(props);
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: fixture.positions},
        timestamps: {format: 'float32', values: fixture.timestamps},
        trackOffsets: {format: 'uint32', values: fixture.trackOffsets},
        triplegTracks: {format: 'uint32', values: fixture.triplegs.tracks},
        triplegStartRows: {format: 'uint32', values: fixture.triplegs.startRows},
        triplegEndRows: {format: 'uint32', values: fixture.triplegs.endRows},
        triplegCount: {format: 'uint32', values: fixture.triplegs.count}
      },
      operations: [poisonOperation(operation)],
      outputs: Object.values(operation.outputs).map(output => output.value)
    });
    // Speeds 1,6,undefined,3,20 map to 0,1,missing,1,2; inactive capacity is zero.
    if (predictions)
      expect(Array.from(result['predictedModes'])).toEqual([0, 1, 0xffffffff, 1, 2, 0]);
    expect(Array.from(result['modalSplit'])).toEqual([1, 1, 0, 0, 1, 1]);
  }
});
