// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPURasterFractionalZonalStatistics} from '../../src/raster-analysis/gpu-raster-fractional-coverage';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {createParityGraph} from '../oracles/parity-wp2-geodesic-harness';
import {readFloat32, readUint32, submitGraph} from '../utils/gpu-contributor-test-utils';

for (const representation of ['inherited', 'non-enumerable'] as const) {
  it(`RT-09 B4 ${representation} extensive quantity retains total units`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = createParityGraph(device);
    const totals = rig.output('totals', 1, 'float32');
    const means = rig.output('means', 1, 'float32');
    const area = rig.output('area', 1, 'float32');
    const validArea = rig.output('valid-area', 1, 'float32');
    const validity = rig.output('validity', 1, 'uint32');
    const props = {
      grid: {
        width: 1,
        height: 1,
        affine: [2, 0, 0, 0, 3, 0] as const,
        pixelInterpretation: 'area' as const
      },
      input: {
        id: 'input',
        format: 'float32' as const,
        storage: {kind: 'buffer' as const, values: rig.input('values', [5], 'float32').view}
      },
      zoneCount: 1,
      quantity: 'extensive' as const,
      coverage: {
        zoneIds: rig.input('zones', [0], 'uint32').view,
        cellIds: rig.input('cells', [0], 'uint32').view,
        fractions: rig.input('fractions', [1], 'float32').view,
        zoneValidity: rig.input('zone-validity', [1], 'uint32').view,
        status: {
          count: rig.input('count', [1], 'uint32').view,
          requiredCount: rig.input('required', [1], 'uint32').view,
          overflow: rig.input('overflow', [0], 'uint32').view
        }
      },
      output: {
        totals: totals.view,
        means: means.view,
        coveredArea: area.view,
        validArea: validArea.view,
        validity: validity.view
      }
    };
    Reflect.deleteProperty(props, 'quantity');
    if (representation === 'inherited')
      Object.setPrototypeOf(props, {
        get quantity() {
          return 'extensive';
        }
      });
    else Object.defineProperty(props, 'quantity', {get: () => 'extensive', enumerable: false});
    rig.graph.add(new GPURasterFractionalZonalStatistics(props));
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      // An extensive value is already a cell total: 5*coverage(1), regardless of area 6.
      expect(await readFloat32(totals.buffer, 1)).toEqual([5]);
      expect(await readFloat32(means.buffer, 1)).toEqual([5]);
      expect(await readFloat32(area.buffer, 1)).toEqual([6]);
      expect(await readFloat32(validArea.buffer, 1)).toEqual([6]);
      expect(await readUint32(validity.buffer, 1)).toEqual([1]);
    } finally {
      compiled.destroy();
      rig.destroy();
    }
  });
}
