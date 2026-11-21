// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {checkTerrainGPUStatus} from '../terrain-status-oracles-test-utils';
import {getTerrainProcessingStatusOracles} from './terrain-processing-status-oracles-fixtures';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect} from 'vitest';
import {GPUTerrainTiledFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createTerrainReviewFixture} from './terrain-review-fixture';

for (const oracle of getTerrainProcessingStatusOracles()) {
  it(`GPU terrain processing independent status oracle: ${oracle.name}`, async () => {
    await checkTerrainGPUStatus(oracle);
  });
}

for (const [execution, routing] of [
  ['auto', 'd8'],
  ['resident', 'd8'],
  ['tiled', 'd8'],
  ['auto', 'branching'],
  ['resident', 'branching']
] as const) {
  it(`tiled flow limit overwrites poison and distinguishes a progressing cycle from outlets: ${execution}/${routing}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createTerrainReviewFixture(device);
    const grid = {
      width: 3,
      height: 1,
      affine: [1, 0, 0, 0, -1, 0] as const,
      pixelInterpretation: 'area' as const
    };
    const directions = fixture.input('uint32', [1, 16, 0]);
    const fractions = fixture.input('float32', new Float32Array(24));
    const states = fixture.output('uint32', 3);
    const converged = fixture.output('uint32', 1),
      limit = fixture.output('uint32', 1),
      invalid = fixture.output('uint32', 1);
    for (const output of [states, converged, limit, invalid])
      output.buffer.write(new Uint32Array(output.view.length).fill(17));
    fixture.graph.add(
      new GPUTerrainTiledFlow({
        grid,
        execution,
        routing,
        maximumIterations: 1,
        tiles: [
          {
            column: 0,
            row: 0,
            width: 3,
            height: 1,
            owned: [0, 0, 3, 1],
            flowDirections: directions.view,
            fractions: routing === 'branching' ? fractions.view : undefined,
            runoff: fixture.input('float32', [1, 1, 1]).view,
            streams: fixture.input('uint32', [1, 1, 1]).view,
            output: {
              accumulation: fixture.output('float32', 3).view,
              status: states.view,
              reachIds: fixture.output('uint32', 3).view
            }
          }
        ],
        output: {
          status: {
            converged: converged.view,
            iterationLimitReached: limit.view,
            invalidCount: invalid.view
          }
        }
      })
    );
    const compiled = fixture.graph.compile();
    try {
      // The cycle cannot publish either cell; independent outlet 2 must publish in sweep 1.
      // A progressing last sweep with the named cycle still pending exhausts budget 1.
      // Removing both edges resolves all three outlets, so the limit flag must clear.
      // The explicit tiled D8 route uses bounded graph ranks, not this capped sweep;
      // its iteration-limit flag is 0 even for a rootless cycle.
      for (const cycle of [true, false, true]) {
        directions.buffer.write(new Uint32Array(cycle ? [1, 16, 0] : [0, 0, 0]));
        const weights = new Float32Array(24);
        if (cycle) {
          weights[0] = 1;
          weights[12] = 1;
        }
        fractions.buffer.write(weights);
        fixture.submit(compiled);
        expect(await states.read()).toEqual(cycle ? [2, 2, 1] : [1, 1, 1]);
        expect(await converged.read()).toEqual(cycle ? [0] : [1]);
        expect(await limit.read()).toEqual(cycle && execution !== 'tiled' ? [1] : [0]);
        expect(await invalid.read()).toEqual([0]);
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}
