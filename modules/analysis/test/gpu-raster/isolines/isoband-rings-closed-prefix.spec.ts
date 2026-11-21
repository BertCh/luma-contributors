// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUIsobandRings} from '../../../src/raster-analysis/isolines/gpu-isoband-rings';
import {getGPUIsobandsParameterValues} from '../../../src/raster-analysis/isolines/isobands-parameters';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {AnalysisRig} from '../../gpu-spatial-analysis/catchment-accessibility/rig';

it('B1 a closed retained square cannot certify a truncated two-square boundary', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeGPURasterMetadata({width: 5, height: 2});
  for (const edgeCapacity of [4, 8]) {
    const rig = new AnalysisRig(device);
    try {
      const count = rig.output('uint32', 1);
      const edgeOverflow = rig.output('uint32', 1);
      const complete = rig.output('uint32', 1);
      const candidateOverflow = rig.output('uint32', 1);
      rig.run(
        new GPUIsobandRings({
          grid,
          edgeCapacity,
          values: rig.input(new Float32Array(10).fill(1), 'float32'),
          mask: rig.input(Uint32Array.from([1, 1, 0, 1, 1, 1, 1, 0, 1, 1]), 'uint32'),
          breaks: rig.input(Float32Array.of(0.5), 'float32'),
          parameters: rig.input(
            getGPUIsobandsParameterValues(
              {breakCount: 1, firstBand: 1, lastBand: 1, extent: [0, 0, 5, 2]},
              grid
            ),
            'float32'
          ),
          output: {
            ringOffsets: rig.output('uint32', 3),
            positions: rig.output('float32x2', 10),
            status: {
              count,
              overflow: rig.output('uint32', 1),
              edgeOverflow,
              complete,
              candidateOverflow
            }
          }
        })
      );
      // The masked sample column separates two unit cells. Each emits a closed four-edge square.
      // Four retained edges can form a complete local ring while half the promised domain is lost.
      expect(await rig.readUint(count)).toEqual([edgeCapacity / 4]);
      expect(await rig.readUint(edgeOverflow)).toEqual([edgeCapacity === 4 ? 1 : 0]);
      expect(await rig.readUint(candidateOverflow)).toEqual([edgeCapacity === 4 ? 1 : 0]);
      expect(await rig.readUint(complete)).toEqual([edgeCapacity === 4 ? 0 : 1]);
    } finally {
      rig.destroy();
    }
  }
});
