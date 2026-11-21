// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUCatchmentAccessibility} from '../../../src/gpu-spatial-analysis/catchment-accessibility/gpu-catchment-accessibility';
import {GPUHuffTradeAreas} from '../../../src/gpu-spatial-analysis/catchment-accessibility/gpu-huff-trade-areas';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

it('keeps a reachable trade area when demandValidity is zero', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const weights = {
    offsets: rig.input(Uint32Array.of(0, 1), 'uint32', 2),
    neighbors: rig.input(Uint32Array.of(0), 'uint32', 1),
    weights: rig.input(Float32Array.of(1), 'float32', 1)
  };
  const demand = rig.input(Float32Array.of(NaN), 'float32', 1);
  const tradeArea = rig.output('uint32', 1),
    huffValidity = rig.output('uint32', 1),
    catchmentValidity = rig.output('uint32', 1),
    expectedDemand = rig.output('float32', 1),
    accessibility = rig.output('float32', 1);
  rig.graph.add(
    new GPUHuffTradeAreas({
      attractiveness: rig.input(Float32Array.of(1), 'float32', 1),
      parameters: rig.input(Float32Array.of(1), 'float32', 1),
      demand,
      demandWeights: weights,
      facilityWeights: weights,
      output: {
        tradeArea: tradeArea.view,
        expectedDemand: expectedDemand.view,
        demandValidity: huffValidity.view
      }
    })
  );
  rig.graph.add(
    new GPUCatchmentAccessibility({
      method: '2sfca',
      supply: rig.input(Float32Array.of(1), 'float32', 1),
      demand,
      facilityWeights: weights,
      output: {accessibility: accessibility.view, demandValidity: catchmentValidity.view}
    })
  );
  const compiled = await rig.graph.compileAsync();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(tradeArea.buffer, 1)).toEqual([0]);
    expect(await readUint32(huffValidity.buffer, 1)).toEqual([0]);
    expect(await readUint32(catchmentValidity.buffer, 1)).toEqual([0]);
    expect(await readFloat32(expectedDemand.buffer, 1)).toEqual([0]);
    expect(await readFloat32(accessibility.buffer, 1)).toEqual([0]);
  } finally {
    compiled.destroy();
    rig.destroy();
    device.destroy();
  }
});
