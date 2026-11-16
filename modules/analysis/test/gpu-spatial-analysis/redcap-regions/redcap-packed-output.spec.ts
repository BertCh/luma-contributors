// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPURedcapSingleLinkage} from '../../../src/gpu-spatial-analysis/redcap-regions/gpu-redcap-single-linkage';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {requireTestDevice} from '../../utils/require-test-device';

it('writes labels [0,32) and regionCount [32,36) in one buffer across separate dispatches', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const results: number[][] = [];
  for (const packed of [false, true]) {
    const rig = new AnalysisRig(device);
    try {
      const storage = rig.output('uint32', packed ? 9 : 8);
      const labels = packed
        ? rig.graph.createDataView(storage.buffer, {format: 'uint32', length: 8})
        : storage;
      const regionCount = packed
        ? rig.graph.createDataView(storage.buffer, {format: 'uint32', length: 1, byteOffset: 32})
        : rig.output('uint32', 1);
      rig.run(
        new GPURedcapSingleLinkage({
          values: rig.input(Float32Array.of(0, 1, 2, 3, 20, 21, 22, 23), 'float32'),
          weights: rig.weights({
            offsets: [0, 1, 3, 5, 7, 9, 11, 13, 14],
            neighbors: [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
            weights: new Array(14).fill(1)
          }),
          parameters: rig.input(Uint32Array.of(2), 'uint32'),
          output: {labels, regionCount}
        })
      );
      const actual = await rig.readUint(storage);
      const count = packed ? actual[8] : (await rig.readUint(regionCount))[0];
      results.push([...actual.slice(0, 8), count]);
      expect(results[results.length - 1]).toEqual([0, 0, 0, 0, 4, 4, 4, 4, 2]);
    } finally {
      rig.destroy();
    }
  }
  expect(results[1]).toEqual(results[0]);
});
