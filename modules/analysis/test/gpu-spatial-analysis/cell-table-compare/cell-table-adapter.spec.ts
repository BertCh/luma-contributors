// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {addCellTableColumnsNode} from '../../../src/gpu-spatial-analysis/recipes/recipe-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {AnalysisRig} from '../catchment-accessibility/rig';

for (const flag of [0, 1]) {
  it(`cell table recipe adapter suppresses lost sums and preserves occupancy, flag ${flag}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    const values = rig.output('float32', 2),
      mask = rig.output('uint32', 2);
    try {
      addCellTableColumnsNode(
        rig.graph,
        'sum-adapter',
        {
          counts: rig.input(Uint32Array.of(2, 0), 'uint32'),
          sumValues: rig.input(Float32Array.of(7, 0), 'float32'),
          status: {sumOverflow: rig.input(Uint32Array.of(flag), 'uint32')}
        },
        {values, mask}
      );
      rig.run();
      expect(await rig.readUint(mask)).toEqual([1, 0]);
      expect(await rig.readFloat(values)).toEqual(flag ? [NaN, 0] : [7, 0]);
    } finally {
      rig.destroy();
    }
  });
}
