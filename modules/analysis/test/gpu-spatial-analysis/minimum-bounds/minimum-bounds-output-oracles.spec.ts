// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUMinimumBoundsOperation} from '../../../src/gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createOracleFixture, addOracleOperation} from '../global-coordinates/output-oracle-fixture';
import {
  BOUNDS_POSITIONS,
  BOUNDS_LABELS,
  BOUNDS_EXPECTED,
  BOUNDS_OUTPUT_CASES,
  getBoundsOracleOperationProps,
  expectBoundsOutputPorts,
  expectBoundsPort
} from './minimum-bounds-output-oracles-fixture';

it('GPU bounds operation writes every independent shape column, alone and together', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Each shape runs alone. Hull size and group overflow are metadata and require bounds as
  // their sole shape dependency. All combinations share the Node constructor preflight.
  for (const selected of BOUNDS_OUTPUT_CASES) {
    const fixture = createOracleFixture(device);
    const outputs = addOracleOperation(
      fixture,
      new GPUMinimumBoundsOperation(getBoundsOracleOperationProps(selected)),
      {
        positions: fixture.input('float32x2', BOUNDS_POSITIONS),
        labels: fixture.input('uint32', BOUNDS_LABELS)
      }
    );
    expectBoundsOutputPorts(selected, outputs);
    const compiled = fixture.graph.compile();
    try {
      for (let submission = 0; submission < 2; submission++) {
        for (const destination of Object.values(outputs)) destination.poison();
        submitGraph(device, compiled, undefined);
        for (const port of ['overflow', ...selected] as (keyof typeof BOUNDS_EXPECTED)[])
          expectBoundsPort(port, await outputs[port].read());
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
}, 60000);
