// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  buildRemoteSensingScenario,
  runRemoteSensingLens
} from '../../../src/computation-lenses/scenarios/gpu-raster/remote-sensing-runner';
import {REMOTE_SENSING_SCENARIOS} from '../../../src/computation-lenses/scenarios/gpu-raster/remote-sensing.scenarios';
import {submitGraph} from '@luma-contributors/graph/testing';
import {requireTestDevice} from '@luma-contributors/graph/testing';
import {createCoverageTrancheOneHarness} from './coverage-tranche-one-harness';

it.each(
  REMOTE_SENSING_SCENARIOS
)('$element executable lens has independently checkable GPU outputs', async ({element}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createCoverageTrancheOneHarness(device, element, 16);
  try {
    await buildRemoteSensingScenario(harness.session, element);
    harness.upload();
    const compiled = harness.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const read = (name: string) => harness.read(compiled, name);
      if (element === 'GPURasterConservativeRemapping') {
        expect((await read('remappedValues'))[0]).toBe(2.5);
        expect((await read('coveredArea'))[15]).toBe(0.25);
        expect(await read('status')).toEqual(Array(16).fill(0));
      } else if (element === 'GPURasterTexture') {
        expect((await read('pairCount'))[5]).toBe(12);
        const features = await read('features');
        expect(features[35]).toBeCloseTo(1, 6);
        expect(features[40]).toBeCloseTo(-1, 6);
      } else if (element === 'GPURasterTemporal') {
        const values = await read('smoothedValues');
        expect(values[0]).toBeCloseTo(2, 4);
        expect(values[8]).toBeCloseTo(90, 3);
        expect(await read('status')).toEqual(Array(144).fill(0));
      } else if (element === 'GPURasterRegistration') {
        expect(await read('shift')).toEqual([-1, 0]);
        expect(await read('overlapCount')).toEqual([12]);
        expect(await read('status')).toEqual([0]);
      } else if (element === 'GPURasterRegionGraph') {
        expect(await read('sourceToRoot')).toEqual([0, 0, 2, 3]);
        expect((await read('updatedWeights'))[6]).toBeCloseTo(1.5, 6);
        expect(await read('status')).toEqual([0]);
      } else if (element === 'GPURasterSpectralUnmixing') {
        const values = await read('abundances');
        expect(values[0]).toBeCloseTo(0, 4);
        expect(values[31]).toBeCloseTo(0, 4);
        expect(await read('status')).toEqual(Array(16).fill(0));
      } else {
        expect(await read('breakIndex')).toEqual(Array(16).fill(0xffffffff));
        const diagnostics = await read('diagnostics');
        expect(diagnostics.filter((_, index) => index % 6 === 4)).toEqual([
          ...Array(8).fill(-1),
          ...Array(8).fill(28)
        ]);
        expect((await read('coefficients'))[1]).toBeCloseTo(7.2, 3);
        expect(await read('status')).toEqual(Array(16).fill(0));
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    harness.destroy();
  }
});

it.each(
  REMOTE_SENSING_SCENARIOS
)('$element validates its actual lens record and captured sidecars', async ({element, slug}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runRemoteSensingLens(device, element, `gpu-raster.raster-${slug}`);
  expect(result.captures?.runId).toBe(result.runId);
  expect(result.captures?.snapshots.length).toBeGreaterThan(0);
  expect(result.graphs[0].nodes.length).toBeGreaterThan(0);
});
