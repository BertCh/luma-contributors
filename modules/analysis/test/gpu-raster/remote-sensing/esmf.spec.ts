// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPURasterConservativeRemapping} from '../../../src/raster-analysis/remote-sensing/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRemoteSensingHarness} from './remote-sensing-test-utils';
import reference from '../../oracles/fixtures/raster-esmf.json';

const grid = (source: (typeof reference.cases)[number]['source']) => ({
  width: source.width,
  height: source.height,
  origin: [source.origin[0], source.origin[1]] as const,
  spacing: [source.spacing[0], source.spacing[1]] as const
});

for (const normalization of ['destination-area', 'covered-area', 'extensive'] as const) {
  for (const entry of reference.cases.filter(
    candidate => candidate.normalization === normalization
  )) {
    it(`RS03 direct ESMF 8.9.1 ${entry.name}/${normalization} with prepared reuse`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = createRemoteSensingHarness(device);
      const source = grid(entry.source);
      const destination = grid(entry.destination);
      const count = destination.width * destination.height;
      const capacity =
        count *
        (Math.ceil(destination.spacing[0] / source.spacing[0]) + 1) *
        (Math.ceil(destination.spacing[1] / source.spacing[1]) + 1);
      const preparedWeights = {
        indices: harness.output('indices', 'uint32', capacity),
        areas: harness.output('areas', 'float32', capacity)
      };
      const validity = harness.mask(entry.validity);
      const contributors = entry.dates.map(
        (date, index) =>
          new GPURasterConservativeRemapping({
            id: `date-${index}`,
            source,
            destination,
            normalization,
            values: harness.input(date.values),
            mask: validity,
            preparedWeights,
            weightMode: index === 0 ? 'prepare' : 'reuse',
            output: {
              values: harness.output(`values-${index}`, 'float32', count),
              coveredArea: harness.output(`covered-${index}`, 'float32', count),
              status: harness.output(`status-${index}`, 'uint32', count)
            }
          })
      );
      harness.graph.add(contributors[0]);
      const result = await harness.run(contributors[1]);
      for (const [dateIndex, date] of entry.dates.entries()) {
        for (let row = 0; row < count; row++) {
          expect(result[`covered-${dateIndex}`][row]).toBeCloseTo(entry.coveredArea[row], 5);
          const expected = date.expected[row];
          expect(result[`status-${dateIndex}`][row]).toBe(expected === null ? 1 : 0);
          if (expected === null)
            expect(Number.isNaN(result[`values-${dateIndex}`][row])).toBe(true);
          else expect(result[`values-${dateIndex}`][row]).toBeCloseTo(expected, 5);
        }
      }
    });
  }
}
