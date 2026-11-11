// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUHealpixPointToCellOperation} from '../../../src/gpu-spatial-analysis/healpix-cells/cpu-healpix-point-to-cell-operation';
import {
  GPUHealpixPointToCellOperation,
  type HealpixPointToCellOperationProps
} from '../../../src/gpu-spatial-analysis/healpix-cells/gpu-healpix-point-to-cell-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {getHealpixNestedCenter} from './rts08-healpix-reference';

for (const scheme of ['nested', 'uniq'] as const) {
  for (const masked of [false, true]) {
    it(`HEALPix operations compare every key word and validity: ${scheme}, masked=${masked}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const order = 3;
      const rowCount = 12 * 4 ** order;
      const positions = new Float32Array(rowCount * 2);
      for (let pixel = 0; pixel < rowCount; pixel++) {
        const center = getHealpixNestedCenter(BigInt(pixel), order);
        positions[pixel * 2] = center.longitude;
        positions[pixel * 2 + 1] = center.latitude;
      }
      // All centres are separated from cell boundaries; integer tolerance is exactly zero.
      if (masked) {
        positions[0] = NaN;
        positions[3] = Infinity;
      }
      const props: HealpixPointToCellOperationProps = {
        positions: 'positions',
        mask: 'mask',
        order,
        scheme,
        rowCount,
        output: {cells: 'cells', validity: 'validity'}
      };
      const inputs = {
        positions: {format: 'float32x2' as const, values: positions},
        mask: {
          format: 'uint32' as const,
          values: Uint32Array.from({length: rowCount}, (_, row) =>
            masked && row % 3 === 0 ? 0 : 1
          )
        }
      };
      const outputs = ['cells', 'validity'];
      const gpu = await runOperationGraph(device, {
        inputs,
        operations: [new GPUHealpixPointToCellOperation(props)],
        outputs
      });
      const cpu = await runOperationGraph(device, {
        inputs,
        operations: [new CPUHealpixPointToCellOperation(props)],
        outputs
      });
      for (const output of outputs)
        expect(Array.from(gpu[output])).toEqual(Array.from(cpu[output]));
    });
  }
}
