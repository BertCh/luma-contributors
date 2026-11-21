// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUGeometryValidity} from '../../../src/gpu-spatial-analysis/geometry-validity/gpu-geometry-validity';
import {GPU_GEOMETRY_VALIDITY_BIT} from '../../../src/gpu-spatial-analysis/geometry-validity/geometry-validity-types';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createOracleFixture} from '../global-coordinates/output-oracle-fixture';

it('validity counts exactly two separated bowtie crossings independently of optional validity', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const countPresent of [false, true]) {
    const fixture = createOracleFixture(device);
    // Each bowtie has exactly one proper crossing of its nonadjacent diagonals.
    // The rings are separated by x=2..4; adjacent-edge contacts are excluded by definition.
    const issueCode = fixture.output('uint32', 2);
    const overflow = fixture.output('uint32', 1);
    const intersectionCount = countPresent ? fixture.output('uint32', 1) : undefined;
    fixture.graph.add(
      new GPUGeometryValidity({
        polygons: {
          kind: 'polygons',
          positions: fixture.input('float32x2', [0, 0, 2, 2, 0, 2, 2, 0, 4, 0, 6, 2, 4, 2, 6, 0]),
          ringOffsets: fixture.input('uint32', [0, 4, 8]),
          polygonOffsets: fixture.input('uint32', [0, 1, 2]),
          featureOffsets: fixture.input('uint32', [0, 1, 2])
        },
        orientation: 'ignore',
        intersectionCapacity: 8,
        output: {
          issueCode: issueCode.view,
          status: {
            overflow: overflow.view,
            ...(intersectionCount ? {intersectionCount: intersectionCount.view} : {})
          }
        }
      })
    );
    const compiled = fixture.graph.compile();
    try {
      for (let submission = 0; submission < 2; submission++) {
        issueCode.poison();
        overflow.poison();
        intersectionCount?.poison();
        submitGraph(device, compiled, undefined);
        expect(await issueCode.read()).toEqual([
          GPU_GEOMETRY_VALIDITY_BIT.selfIntersection,
          GPU_GEOMETRY_VALIDITY_BIT.selfIntersection
        ]);
        expect(await overflow.read()).toEqual([0]);
        if (intersectionCount) expect(await intersectionCount.read()).toEqual([2]);
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
});
