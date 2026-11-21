// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPUPettittTest} from '../../../src/gpu-spatial-analysis/change-points/gpu-pettitt-test';
import {GPULocationChangePoints} from '../../../src/gpu-spatial-analysis/change-points/gpu-location-change-points';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {expectNumericArrayClose} from '../../../src/testing/expect-numeric-array-close';
import {createInputBuffer, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {AnalysisRig} from '../catchment-accessibility/rig';

it('W2-15 replays captured Pettitt and location-change contributors against the original GPU resources', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  const valuesBuffer = createInputBuffer(device, Float32Array.of(0, 0, 10, 10));
  try {
    const values = importGraphBuffer(rig.graph, 'live-values', valuesBuffer, 'float32', 4);
    const pettittOutput = {
      statistic: rig.output('uint32', 1),
      changeSlice: rig.output('uint32', 1),
      pValues: rig.output('float32', 1)
    };
    const pettittProps = {
      values,
      layout: 'cell-major' as const,
      cellCount: 1,
      sliceCount: 4,
      output: {...pettittOutput}
    };
    const pettitt = new GPUPettittTest(pettittProps);
    const locationOutput = {
      changePointCount: rig.output('uint32', 1),
      changePoints: rig.output('uint32', 2),
      optimalCost: rig.output('float32', 1),
      status: {overflow: rig.output('uint32', 1)}
    };
    const locationProps = {
      values,
      layout: 'cell-major' as const,
      cellCount: 1,
      sliceCount: 4,
      cost: 'mean' as const,
      maximumChangePoints: 2,
      parameters: rig.input(Float32Array.of(1, 1), 'float32'),
      output: {...locationOutput, status: {...locationOutput.status}}
    };
    const location = new GPULocationChangePoints(locationProps);
    pettittProps.sliceCount = 1;
    pettittProps.output.pValues = rig.output('float32', 1);
    locationProps.sliceCount = 1;
    locationProps.output.status.overflow = rig.output('uint32', 1);
    rig.graph.add(pettitt);
    rig.graph.add(location);
    const compiled = rig.graph.compile();
    try {
      for (const constant of [false, true, false]) {
        valuesBuffer.write(constant ? Float32Array.of(2, 2, 2, 2) : Float32Array.of(0, 0, 10, 10));
        submitGraph(device, compiled, undefined);
        // Independent cross-split Pettitt K=4 and p=2exp(-96/80); constant K=0.
        expectNumericArrayClose(await rig.readUint(pettittOutput.statistic), [constant ? 0 : 4]);
        expectNumericArrayClose(await rig.readUint(pettittOutput.changeSlice), [constant ? 1 : 2]);
        expectNumericArrayClose(
          await rig.readFloat(pettittOutput.pValues),
          [constant ? 1 : 0.6023884238244043],
          {absoluteTolerance: 2e-6}
        );
        // Mean cost: no-split SSE=100, split SSE=0 plus penalty=1.
        expectNumericArrayClose(await rig.readUint(locationOutput.changePointCount), [
          constant ? 0 : 1
        ]);
        expectNumericArrayClose(
          await rig.readUint(locationOutput.changePoints),
          constant ? [0xffffffff, 0xffffffff] : [2, 0xffffffff]
        );
        expectNumericArrayClose(
          await rig.readFloat(locationOutput.optimalCost),
          [constant ? 0 : 1],
          {absoluteTolerance: 1e-6}
        );
        expectNumericArrayClose(await rig.readUint(locationOutput.status.overflow), [0]);
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    valuesBuffer.destroy();
    rig.destroy();
  }
});
