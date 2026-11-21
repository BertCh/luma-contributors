// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Timing measurement moved out of the correctness spec (TOOL-110): the `-bench` suffix keeps it out
// of `yarn verify` and runs it under `yarn test-benchmarks`.

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {isSoftwareDevice, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createClusteredScene, createFixture} from './edge-bundling-fixture';

it('GPUEdgeBundling measures 10k edges x 16 points x 15 iterations at 256 squared', async context => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  if (isSoftwareDevice(device)) {
    context.skip('edge bundling timing needs a hardware-backed device');
  }
  const scene = createClusteredScene(2000, 10000, 99);
  const fixture = createFixture(device, scene, {withIndices: true});
  await fixture.run();
  const times: number[] = [];
  for (let sample = 0; sample < 7; sample++) {
    const start = performance.now();
    submitGraph(device, fixture.compiled, undefined);
    await readUint32(fixture.drawRecordBuffer!, 4);
    times.push(performance.now() - start);
  }
  times.sort((a, b) => a - b);
  console.log(
    `edge-bundling 10k x 16 x 15 @256^2: median ${times[3].toFixed(1)} ms (min ${times[0].toFixed(1)}, max ${times[6].toFixed(1)}), ${fixture.nodeCount} nodes`
  );
  expect(times[3]).toBeGreaterThan(0);
  fixture.destroy();
}, 120000);
