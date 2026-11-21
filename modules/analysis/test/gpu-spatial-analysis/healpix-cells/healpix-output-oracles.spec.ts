// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUHealpixPointToCellOperation} from '../../../src/gpu-spatial-analysis/healpix-cells/gpu-healpix-point-to-cell-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createOracleFixture, addOracleOperation} from '../global-coordinates/output-oracle-fixture';

it('GPU HEALPix operation writes literal base faces and validity independently', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Independent order-zero base-face derivation is in healpix-output-oracles.node.spec.ts.
  for (const scheme of ['nested', 'uniq'] as const)
    for (const validityPresent of [false, true]) {
      const fixture = createOracleFixture(device);
      const outputs = addOracleOperation(
        fixture,
        new GPUHealpixPointToCellOperation({
          positions: 'positions',
          mask: 'mask',
          rowCount: 5,
          order: 0,
          scheme,
          output: {cells: 'cells', ...(validityPresent ? {validity: 'validity'} : {})}
        }),
        {
          positions: fixture.input('float32x2', [0, 0, 90, 0, 45, 60, NaN, 0, 180, 0]),
          mask: fixture.input('uint32', [1, 1, 1, 1, 0])
        }
      );
      const compiled = fixture.graph.compile();
      try {
        for (let submission = 0; submission < 2; submission++) {
          for (const output of Object.values(outputs)) output.poison();
          submitGraph(device, compiled, undefined);
          expect(await outputs['cells'].read()).toEqual(
            scheme === 'nested' ? [4, 0, 5, 0, 0, 0, 0, 0, 0, 0] : [8, 0, 9, 0, 4, 0, 0, 0, 0, 0]
          );
          if (validityPresent) expect(await outputs['validity'].read()).toEqual([1, 1, 1, 0, 0]);
        }
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    }
});
