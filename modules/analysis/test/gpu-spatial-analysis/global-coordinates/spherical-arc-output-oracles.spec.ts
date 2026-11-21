// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUSphericalArcOverlayOperation} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-arc-overlay-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createOracleFixture} from './output-oracle-fixture';
import {expectOctant, OCTANT_POSITIONS, OCTANT_PROPS} from './spherical-arc-output-oracles-fixture';

it('GPU arc operation writes the independent octant ring over poisoned outputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOracleFixture(device);
  const operation = new GPUSphericalArcOverlayOperation(OCTANT_PROPS);
  const inputs = {
    leftPositions: fixture.input('float32x3', OCTANT_POSITIONS),
    rightPositions: fixture.input('float32x3', OCTANT_POSITIONS),
    leftRingOffsets: fixture.input('uint32', [0, 3]),
    rightRingOffsets: fixture.input('uint32', [0, 3])
  };
  const outputs = {
    positions: fixture.output('float32x3', 5),
    ringOffsets: fixture.output('uint32', 3),
    counts: fixture.output('uint32', 4),
    status: fixture.output('uint32', 1)
  };
  const views = {
    ...inputs,
    ...Object.fromEntries(Object.entries(outputs).map(([name, output]) => [name, output.view]))
  };
  fixture.graph.add(
    operation.createContributor(name => views[name as keyof typeof views] as never)
  );
  const compiled = fixture.graph.compile();
  try {
    for (let submission = 0; submission < 2; submission++) {
      for (const output of Object.values(outputs)) output.poison();
      submitGraph(device, compiled, undefined);
      expectOctant(
        Object.fromEntries(
          await Promise.all(
            Object.entries(outputs).map(async ([name, output]) => [name, await output.read()])
          )
        )
      );
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
