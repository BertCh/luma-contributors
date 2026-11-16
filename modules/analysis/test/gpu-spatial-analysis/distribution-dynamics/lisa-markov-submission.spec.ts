// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPULISAMarkov} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-lisa-markov';
import {getGPUSpatialAutocorrelationParameterValues} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/spatial-autocorrelation-parameters';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {requireTestDevice} from '../../utils/require-test-device';
import * as giddy from './markov-giddy-fixture';

for (const diagnostics of [false, true]) {
  for (const [name, values, states, counts, ignored] of [
    ['all-period missing', giddy.C1_VALUES, giddy.C1_STATES, giddy.C1_COUNTS, giddy.C1_IGNORED],
    ['one-period missing', giddy.C2_VALUES, giddy.C2_STATES, giddy.C2_COUNTS, giddy.C2_IGNORED]
  ] as const) {
    it(`submits the isolated 37-row giddy LISA chain: ${name}, diagnostics ${diagnostics}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new AnalysisRig(device);
      try {
        const offsets = [0];
        const neighbors: number[] = [];
        const weights: number[] = [];
        for (let row = 0; row < 36; row++) {
          const adjacent = [];
          if (row >= 6) adjacent.push(row - 6);
          if (row % 6 > 0) adjacent.push(row - 1);
          if (row % 6 < 5) adjacent.push(row + 1);
          if (row < 30) adjacent.push(row + 6);
          neighbors.push(...adjacent);
          weights.push(...adjacent.map(() => 1 / adjacent.length));
          offsets.push(neighbors.length);
        }
        offsets.push(neighbors.length); // Row 36 is an island.
        const output = {
          quadrants: rig.output('uint32', 148),
          counts: rig.output('uint32', 25),
          ignored: rig.output('uint32', 1),
          ...(diagnostics
            ? {steadyStates: rig.output('float32', 5), stateSparsity: rig.output('uint32', 2)}
            : {})
        };
        rig.run(
          new GPULISAMarkov({
            values: rig.input(Float32Array.from(values), 'float32'),
            weights: {
              ...rig.weights({offsets, neighbors, weights}),
              information: {rowCount: 37, transform: 'row', symmetric: false}
            },
            periodCount: 4,
            parameters: rig.input(getGPUSpatialAutocorrelationParameterValues({}), 'float32'),
            output
          })
        );
        expect(await rig.readUint(output.quadrants)).toEqual(
          states.map(state => (state < 0 ? 0xffffffff : state))
        );
        expect(await rig.readUint(output.counts)).toEqual(counts);
        expect((await rig.readUint(output.ignored))[0]).toBe(ignored);
      } finally {
        rig.destroy();
      }
    });
  }
}
