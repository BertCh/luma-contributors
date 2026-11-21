// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUGroupConvexHull} from '../../../src/gpu-spatial-analysis/group-geometry/gpu-group-convex-hull';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createOracleFixture} from '../global-coordinates/output-oracle-fixture';

it('group hull writes every independently defined triangle coordinate and NaN tail', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const positionsPresent of [false, true]) {
    const fixture = createOracleFixture(device);
    // (1,2),(5,2),(1,5) is a CCW triangle; (2,3) lies strictly inside.
    // Lexicographic start is (1,2). Input row order is deliberately different.
    const indices = fixture.output('uint32', 5),
      offsets = fixture.output('uint32', 2),
      counts = fixture.output('uint32', 1),
      overflow = fixture.output('uint32', 1);
    const positions = positionsPresent ? fixture.output('float32x2', 5) : undefined;
    fixture.graph.add(
      new GPUGroupConvexHull({
        positions: fixture.input('float32x2', [5, 2, 2, 3, 1, 5, 1, 2]),
        labels: fixture.input('uint32', [0, 0, 0, 0]),
        groupCount: 1,
        maximumVerticesPerGroup: 4,
        vertexCapacity: 5,
        output: {
          vertexIndices: indices.view,
          offsets: offsets.view,
          counts: counts.view,
          status: {overflow: overflow.view},
          ...(positions ? {vertexPositions: positions.view} : {})
        }
      })
    );
    const compiled = fixture.graph.compile();
    try {
      for (let submission = 0; submission < 2; submission++) {
        positions?.poison();
        submitGraph(device, compiled, undefined);
        expect(await indices.read()).toEqual([3, 0, 2, 0xffffffff, 0xffffffff]);
        expect(await offsets.read()).toEqual([0, 3]);
        expect(await counts.read()).toEqual([3]);
        expect(await overflow.read()).toEqual([0]);
        if (positions) {
          const values = await positions.read();
          expect(values.slice(0, 6)).toEqual([1, 2, 5, 2, 1, 5]);
          for (const value of values.slice(6)) expect(value).toBeNaN();
        }
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
});
