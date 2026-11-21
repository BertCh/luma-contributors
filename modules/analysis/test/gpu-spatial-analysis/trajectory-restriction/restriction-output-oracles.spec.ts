// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUTrajectoryRestrictionOperation} from '../../../src/gpu-spatial-analysis/trajectory-restriction/gpu-trajectory-restriction-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {poisonOperation} from '../trajectory-cleaning/trajectory-output-oracle-utils';
import {getSceneInputs, getSceneProps} from './trajectory-restriction-operation-cases';
import {
  createRestrictionOracleScene,
  expectRestrictionCompact
} from './restriction-output-oracles-fixture';

it('restriction poisoned compact records match GEOS with the compact group present and absent', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  let baseline: Record<string, ArrayLike<number>> | undefined;
  // The constructor requires all five compact destinations together, so isolate the group.
  for (const compact of [true, false]) {
    const scene = createRestrictionOracleScene(compact);
    const props = getSceneProps(scene);
    const operation = new GPUTrajectoryRestrictionOperation(props);
    const result = await runOperationGraph(device, {
      inputs: getSceneInputs(scene),
      operations: [poisonOperation(operation)],
      outputs: Object.values(operation.outputs).map(output => output.value)
    });
    if (compact) {
      expectRestrictionCompact(result);
      baseline = result;
    } else
      for (const port of ['intervals', 'positions', 'status', 'relationships'])
        expect(Array.from(result[port]), port).toEqual(Array.from(baseline![port]));
  }
});
