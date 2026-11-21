// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {addStopValueCubeRecipe} from '../../../src/gpu-spatial-analysis/index';
import {RecipeTestFixture} from './recipe-harness';
import {createStopValueCubeFixture} from './stop-value-cube-fixture';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

it('CCK-01 signed stop times join dictionary rows and suppress every loss cause on replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'stop-inventory');
  let serial = 0;
  try {
    const props = createStopValueCubeFixture((format, values, length) =>
      fixture.input(`input-${serial++}`, values, format, length)
    );
    const losses = Array.from({length: 6}, (_, index) =>
      fixture.output(`loss-${index}`, 'uint32', 1)
    );
    props.stops.status.overflow = losses[0].view;
    props.stops.status.candidateOverflow = losses[1].view;
    props.sliceStatuses = [{overflow: losses[2].view, candidateOverflow: losses[3].view}];
    props.weights.status = {overflow: losses[4].view, candidateOverflow: losses[5].view};
    const requiredCount = fixture.output('weights-required-count', 'uint32', 1);
    requiredCount.buffer.write(Uint32Array.of(2));
    props.weights.status.requiredCount = requiredCount.view;
    const counts = fixture.output('counts', 'uint32', 6);
    const result = addStopValueCubeRecipe(fixture.graph, {...props, output: {counts: counts.view}});
    const status = fixture.output('status', 'uint32', 3);
    fixture.graph.add(
      createWGSLKernelNode(fixture.graph, {
        id: 'status-readback',
        operation: 'stop-cube-inventory',
        invocationCount: 1,
        bindings: [
          {name: 'invalid', view: result.status.invalidCount, type: 'u32', access: 'read'},
          {name: 'overflow', view: result.status.overflow, type: 'u32', access: 'read'},
          {name: 'candidate', view: result.status.candidateOverflow, type: 'u32', access: 'read'},
          {name: 'output', view: status.view, type: 'u32', access: 'read_write'}
        ],
        body: 'output[outputOffset] = invalid[invalidOffset]; output[outputOffset + 1u] = overflow[overflowOffset]; output[outputOffset + 2u] = candidate[candidateOffset];'
      })
    );
    fixture.run();
    // Dictionary order SE,NW; live stop start times 15,-5 fall in periods 2,0.
    expect(await fixture.readUint32(counts, 6)).toEqual([0, 0, 1, 1, 0, 0]);
    expect(await fixture.readUint32(status, 3)).toEqual([0, 0, 0]);
    expect(result.weights.offsets).toBe(props.weights.offsets);
    for (const [index, loss] of losses.entries()) {
      loss.buffer.write(Uint32Array.of(1));
      fixture.run();
      expect(await fixture.readUint32(counts, 6)).toEqual([0, 0, 0, 0, 0, 0]);
      expect(await fixture.readUint32(status, 3)).toEqual(index % 2 === 0 ? [0, 1, 0] : [0, 0, 1]);
      loss.buffer.write(Uint32Array.of(0));
      fixture.run();
      expect(await fixture.readUint32(counts, 6)).toEqual([0, 0, 1, 1, 0, 0]);
      expect(await fixture.readUint32(status, 3)).toEqual([0, 0, 0]);
    }
    // B6: two retained slots cannot certify a requirement of three, even with clear flags.
    for (const required of [2, 3, 2]) {
      requiredCount.buffer.write(Uint32Array.of(required));
      fixture.run();
      expect(await fixture.readUint32(status, 3)).toEqual([0, required === 3 ? 1 : 0, 0]);
      expect(await fixture.readUint32(counts, 6)).toEqual(
        required === 3 ? [0, 0, 0, 0, 0, 0] : [0, 0, 1, 1, 0, 0]
      );
    }
  } finally {
    fixture.destroy();
  }
});
