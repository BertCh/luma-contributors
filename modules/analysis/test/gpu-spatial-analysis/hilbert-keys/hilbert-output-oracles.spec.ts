// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUHilbertKeys} from '../../../src/gpu-spatial-analysis/hilbert-keys/gpu-hilbert-keys';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createOracleFixture} from '../global-coordinates/output-oracle-fixture';

it('Hilbert sorted keys are literally 0,1,2,3 on the full order-one grid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The order-one U curve visits (0,0),(0,1),(1,1),(1,0). Input rows have keys 2,0,3,1.
  // sortedKeys requires sortedRows; this documented dependency is the only other optional port.
  for (const sortedKeysPresent of [false, true]) {
    const fixture = createOracleFixture(device);
    const keys = fixture.output('uint32', 4),
      rows = fixture.output('uint32', 4),
      sorted = sortedKeysPresent ? fixture.output('uint32', 4) : undefined;
    fixture.graph.add(
      new GPUHilbertKeys({
        order: 1,
        positions: fixture.input('float32x2', [0.75, 0.75, 0.25, 0.25, 0.75, 0.25, 0.25, 0.75]),
        bounds: fixture.input('float32', [0, 0, 1, 1]),
        output: {
          keys: keys.view,
          sortedRows: rows.view,
          ...(sorted ? {sortedKeys: sorted.view} : {})
        }
      })
    );
    const compiled = fixture.graph.compile();
    try {
      for (let submission = 0; submission < 2; submission++) {
        keys.poison();
        rows.poison();
        sorted?.poison();
        submitGraph(device, compiled, undefined);
        expect(await keys.read()).toEqual([2, 0, 3, 1]);
        expect(await rows.read()).toEqual([1, 3, 0, 2]);
        if (sorted) expect(await sorted.read()).toEqual([0, 1, 2, 3]);
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
});
