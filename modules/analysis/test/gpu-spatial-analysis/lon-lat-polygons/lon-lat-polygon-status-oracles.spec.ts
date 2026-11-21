// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPULonLatPolygonPreparation} from '../../../src/gpu-spatial-analysis/lon-lat-polygons/gpu-lon-lat-polygon-preparation';
import {getGPULonLatPolygonPreparationParameterValues} from '../../../src/gpu-spatial-analysis/lon-lat-polygons/lon-lat-polygon-preparation-parameters';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createOracleFixture} from '../global-coordinates/output-oracle-fixture';

it.each([
  'linear',
  'capacity',
  'disabled great-circle',
  'clamped great-circle',
  'invalid latitude'
] as const)('longitude-latitude preparation overwrites hand-derived diagnostics: %s', async scene => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOracleFixture(device);
  const greatCircle = scene.includes('great-circle');
  const clamped = scene === 'clamped great-circle';
  const invalid = scene === 'invalid latitude';
  const capacity = scene === 'capacity' ? 3 : 8;
  const flags = fixture.output('uint32', 1);
  const status = {
    count: fixture.output('uint32', 1),
    requiredCount: fixture.output('uint32', 1),
    overflow: fixture.output('uint32', 1),
    invalidCount: fixture.output('uint32', 1),
    approximation: fixture.output('uint32', 1)
  };
  // A four-vertex square has no seam, cap, ambiguity or over-wide flag. With
  // maximumPieceCount=1 each two-degree edge exceeds a half-degree step: bit 32.
  // A latitude 91 is invalid: bit 16 and exactly one emptied feature.
  fixture.graph.add(
    new GPULonLatPolygonPreparation({
      positions: fixture.input('float32x2', [0, invalid ? 91 : 0, 2, 0, 2, 2, 0, 2]),
      featureOffsets: fixture.input('uint32', [0, 1]),
      polygonOffsets: fixture.input('uint32', [0, 1]),
      ringOffsets: fixture.input('uint32', [0, 4]),
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'none',
        edgeModel: greatCircle ? 'great-circle' : 'linear'
      },
      maximumPieceCount: 1,
      parameters: fixture.input(
        'float32',
        Array.from(
          getGPULonLatPolygonPreparationParameterValues(greatCircle ? 'great-circle' : 'linear', {
            maximumSegmentDegrees: clamped ? 0.5 : 0
          })
        )
      ),
      output: {
        positions: fixture.output('float32x2', capacity).view,
        featureOffsets: fixture.output('uint32', 2).view,
        polygonOffsets: fixture.output('uint32', 3).view,
        ringOffsets: fixture.output('uint32', 3).view,
        ringFlags: flags.view,
        status: {
          count: status.count.view,
          requiredCount: status.requiredCount.view,
          overflow: status.overflow.view,
          invalidCount: status.invalidCount.view,
          approximation: status.approximation.view
        }
      }
    })
  );
  const compiled = fixture.graph.compile();
  try {
    for (let submission = 0; submission < 2; submission++) {
      for (const output of [flags, ...Object.values(status)]) output.poison();
      submitGraph(device, compiled, undefined);
      expect(await flags.read()).toEqual([invalid ? 16 : clamped ? 32 : 0]);
      expect(await status.overflow.read()).toEqual([Number(scene === 'capacity')]);
      expect(await status.invalidCount.read()).toEqual([Number(invalid)]);
      expect(await status.approximation.read()).toEqual([Number(greatCircle)]);
      if (scene === 'capacity') {
        // Dense input already needs four slots. The documented conservative
        // resize requirement is 8*(denseCount+3*ringCount) = 8*(4+3) = 56.
        expect(await status.requiredCount.read()).toEqual([56]);
      } else {
        expect(await status.count.read()).toEqual([invalid ? 0 : 4]);
        expect(await status.requiredCount.read()).toEqual([invalid ? 0 : 4]);
      }
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
