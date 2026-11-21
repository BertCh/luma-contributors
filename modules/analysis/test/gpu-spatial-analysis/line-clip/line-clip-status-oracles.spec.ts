// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPULineClipByPolygon} from '../../../src/gpu-spatial-analysis/line-clip/gpu-line-clip-by-polygon';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createOracleFixture} from '../global-coordinates/output-oracle-fixture';

it.each([
  'complete',
  'piece capacity',
  'pair capacity'
] as const)('line clip exact crossings overwrite poisoned status: %s', async mode => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOracleFixture(device);
  const pieceCapacity = mode === 'piece capacity' ? 1 : 2;
  const count = fixture.output('uint32', 1);
  const requiredCount = fixture.output('uint32', 1);
  const vertexCount = fixture.output('uint32', 1);
  const overflow = fixture.output('uint32', 1);
  const candidateOverflow = fixture.output('uint32', 1);
  const uncertainCount = fixture.output('uint32', 1);
  // Two horizontal integer-coordinate lines cross the square [0,4]^2 at
  // x=0 and x=4. There are four certified segment pairs, no near-degeneracy,
  // and exactly two two-vertex pieces. Capacity limits do not cause uncertainty.
  fixture.graph.add(
    new GPULineClipByPolygon({
      lines: {
        kind: 'lines',
        positions: fixture.input('float32x2', [-1, 1, 5, 1, -1, 3, 5, 3]),
        lineOffsets: fixture.input('uint32', [0, 2, 4])
      },
      polygons: {
        kind: 'polygons',
        positions: fixture.input('float32x2', [0, 0, 4, 0, 4, 4, 0, 4]),
        ringOffsets: fixture.input('uint32', [0, 4]),
        polygonOffsets: fixture.input('uint32', [0, 1]),
        featureOffsets: fixture.input('uint32', [0, 1])
      },
      intersectionCapacity: mode === 'pair capacity' ? 1 : 8,
      candidateCapacity: 32,
      output: {
        pieces: {
          geometry: {
            kind: 'lines',
            positions: fixture.output('float32x2', 4).view,
            lineOffsets: fixture.output('uint32', pieceCapacity + 1).view
          },
          sourceIds: fixture.output('uint32', pieceCapacity).view,
          status: {
            count: count.view,
            requiredCount: requiredCount.view,
            vertexCount: vertexCount.view,
            overflow: overflow.view,
            candidateOverflow: candidateOverflow.view
          }
        },
        status: {uncertainCount: uncertainCount.view}
      }
    })
  );
  const compiled = fixture.graph.compile();
  try {
    for (let submission = 0; submission < 2; submission++) {
      for (const output of [
        count,
        requiredCount,
        vertexCount,
        overflow,
        candidateOverflow,
        uncertainCount
      ])
        output.poison();
      submitGraph(device, compiled, undefined);
      expect(await uncertainCount.read()).toEqual([0]);
      expect(await overflow.read()).toEqual([Number(mode !== 'complete')]);
      expect(await candidateOverflow.read()).toEqual([Number(mode === 'pair capacity')]);
      if (mode !== 'pair capacity') {
        expect(await count.read()).toEqual([pieceCapacity]);
        expect(await requiredCount.read()).toEqual([2]);
        expect(await vertexCount.read()).toEqual([pieceCapacity * 2]);
      }
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
