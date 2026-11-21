// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOverlayInputSnapshotFixture} from './overlay-input-snapshot-fixture';

for (const packed of [false, true]) {
  for (const ids of [false, true]) {
    it(`preserves tagged triangle provenance after caller operand mutation (packed=${packed}, ids=${ids})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = createOverlayInputSnapshotFixture(device, {mutate: true, packed, ids});
      fixture.graph.add(fixture.nodes);
      try {
        for (let submission = 0; submission < 2; submission++) {
          for (const entry of fixture.columns.filter(entry => entry.output))
            entry.buffer.write(entry.values);
          const actual = await fixture.runGPU();
          expect(actual['boundary-count']).toEqual([3]);
          // Intersection: x>=1, y>=0, x+y<=2. Bottom/diagonal are left ID 101,
          // vertical edge is right ID 202. Representative ownership uses source rows.
          expect(actual['owners'].slice(0, 3)).toEqual(ids ? [101, 202, 101] : [1, 1, 1]);
          const expected = [1, 0, 2, 0, 1, 1, 1, 0, 2, 0, 1, 1];
          const liveEndpoints = actual['boundary'].slice(0, 12);
          expect(liveEndpoints).toHaveLength(12);
          expect(liveEndpoints).toEqual(expected);
          expect(actual['source-rows'][0]).toBe(1);
          expect(actual['source-operands'][0]).toBe(0);
          expect(actual['mixed'][0]).toBe(1);
          expect(actual['overflow']).toEqual([0]);
          expect(actual['candidate']).toEqual([0]);
        }
      } finally {
        fixture.destroy();
      }
    });
  }
}
