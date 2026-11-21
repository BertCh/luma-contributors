// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice, submitGraph} from '../../../../analysis/src/testing';
import {
  GPUCostDistancePath,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {addCostDistancePathMembership} from '../../../src/computation-lenses/scenarios/gpu-raster/cost-distance-path.example';
import {createCoverageTrancheOneHarness} from './coverage-tranche-one-harness';

test('W5-16 count mask clears membership after a short path becomes unresolved on one graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createCoverageTrancheOneHarness(device, 'cost-path-membership', 300);
  const {session} = harness;
  const links = new Uint32Array(300).fill(0xfffffffe);
  // Westward D8 links: 2 -> 1 -> 0 (source). Only these three cells contribute once.
  links.set([0, 16, 16]);
  const backLinks = session.input('links', 'uint32', links);
  const path = {
    ids: session.input('ids', 'uint32', new Uint32Array(300)),
    status: {
      count: session.output('count', 'uint32', {}, 1),
      overflow: session.output('overflow', 'uint32', {}, 1)
    }
  };
  const membership = session.output('membership', 'uint32', {});
  harness.graph.add(
    new GPUCostDistancePath({
      grid: makeGPURasterMetadata({width: 300, height: 1}),
      backLinks,
      target: session.input('target', 'uint32', [2]),
      output: {path}
    })
  );
  addCostDistancePathMembership(harness.graph, path, membership, 'membership');
  const compiled = harness.graph.compile();
  try {
    harness.upload();
    submitGraph(device, compiled, undefined);
    expect(await harness.read(compiled, 'count')).toEqual([3]);
    expect(await harness.read(compiled, 'membership')).toEqual([
      1,
      1,
      1,
      ...new Array(297).fill(0)
    ]);
    // The same compiled graph must overwrite the entire row mask, including the former live rows.
    harness.inputs.get('links')!.values[0] = 0xfffffffe;
    harness.upload();
    submitGraph(device, compiled, undefined);
    expect(await harness.read(compiled, 'count')).toEqual([0]);
    expect(await harness.read(compiled, 'membership')).toEqual(new Array(300).fill(0));
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});
