// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUVoronoiDiagram} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/index';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// A single right triangle has three hull bisector rays. All meet clip box
// [-1,-1,2,2], so requiredCount=3. A distant box misses every ray here: the
// lower/right/left rays respectively have x=.5, x=y, and y=.5; use a box
// entirely above-left of the circumcenter, [-3,2,-2,3].
for (const capacity of [1, 4]) {
  it(`Voronoi overwrites complete status words from three analytic rays (capacity ${capacity})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOutputOracleFixture(device);
    const {graph, input, output} = fixture;
    const clipBounds = input('clip', 'float32x4', Float32Array.of(-1, -1, 2, 2));
    const status = {
      count: output('count', 'uint32', 1),
      requiredCount: output('required', 'uint32', 1),
      overflow: output('overflow', 'uint32', 1),
      invalidCount: output('invalid', 'uint32', 1)
    };
    graph.add(
      new GPUVoronoiDiagram({
        positions: input('positions', 'float32x2', Float32Array.of(0, 0, 1, 0, 0, 1)),
        triangles: input('triangles', 'uint32x3', Uint32Array.of(0, 1, 2)),
        triangleCount: input('triangle-count', 'uint32', Uint32Array.of(1)),
        clipBounds,
        output: {
          segments: output('segments', 'float32x4', capacity),
          siteIds: output('ids', 'uint32x2', capacity),
          status
        }
      })
    );
    const compiled = graph.compile();
    try {
      for (const present of [true, false, true]) {
        fixture.writeInput(
          clipBounds,
          Float32Array.from(present ? [-1, -1, 2, 2] : [-3, 2, -2, 3])
        );
        const actual = await fixture.submitTwice(compiled, () =>
          fixture.readRows([
            status.count,
            status.requiredCount,
            status.overflow,
            status.invalidCount
          ])
        );
        expect(actual).toEqual(
          !present ? [0, 0, 0, 0] : capacity === 1 ? [1, 3, 1, 0] : [3, 3, 0, 0]
        );
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}
