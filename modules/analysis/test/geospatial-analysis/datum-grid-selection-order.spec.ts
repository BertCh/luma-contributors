// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUDatumGridTransform} from '../../src/geospatial-analysis/datum-grid-transform';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../utils/gpu-contributor-test-utils';
import {createGlobalCoordinateFixture} from '../gpu-spatial-analysis/global-coordinates/global-coordinate-fixture';
import {getDatumScalingGrids} from '../gpu-spatial-analysis/global-coordinates/global-coordinate-scaling-fixtures';
import {expectDatumSelectionPositions} from '../gpu-spatial-analysis/global-coordinates/global-coordinate-output-assertions';

it('ordered overlapping datum grids preserve first coverage, nodata and no coverage as grid count grows', {
  timeout: 300000
}, async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Large grid counts belong to the opt-in benchmark; eight exercise ordered coverage here.
  for (const gridCount of [8]) {
    for (const reverse of [false, true])
      for (const nodata of [false, true]) {
        const fixture = createGlobalCoordinateFixture(device);
        const covering = [0, 4].map(sampleOffset => ({
          width: 2,
          height: 2,
          sampleOffset,
          origin: [0, 0] as const,
          spacing: [1, 1] as const
        }));
        const grids = [
          ...getDatumScalingGrids(gridCount - 2),
          ...(reverse ? covering.reverse() : covering)
        ];
        const positions = fixture.output('float32x3', 2);
        const status = fixture.output('uint32', 2);
        const mask = new Array(8).fill(1);
        if (nodata) mask[reverse ? 4 : 0] = 0;
        fixture.graph.add(
          new GPUDatumGridTransform({
            positions: fixture.input('float32x3', [0.5, 0.5, 10, -2, -2, 10]),
            samples: fixture.input('float32x3', [
              ...Array(4).fill([0.25, -0.125, 2]).flat(),
              ...Array(4).fill([-0.5, 0.5, 9]).flat()
            ]),
            mask: fixture.input('uint32', mask),
            grids,
            operation: 'horizontal',
            output: {positions: positions.view, status: status.view}
          })
        );
        const compiled = fixture.graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          expect(await readUint32(status.buffer, 2)).toEqual([nodata ? 4 : 0, 2]);
          const result = await readFloat32(positions.buffer, 6);
          // Constant bilinear shifts are exact: (.5,.5)+(.25,-.125) or +(-.5,.5).
          expectDatumSelectionPositions(result, reverse, nodata);
        } finally {
          compiled.destroy();
          fixture.destroy();
        }
      }
  }
});
