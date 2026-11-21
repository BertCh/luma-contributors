// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUBoundsFilter,
  getGPUBoundsFilterParameterValues
} from '../../../src/gpu-spatial-analysis/bounds-filter/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';

// Three nested squares [0,6]^2, [1,5]^2, [2,4]^2. Each mode's boxes below
// select all three, exactly one, then none by closed interval comparisons.
// With two output slots, (count, requiredCount, overflow) is consequently
// (2,3,1), (1,1,0), (0,0,0). No geometry library or implementation output is used.
for (const testCase of [
  {mode: 'intersects' as const, all: [3, 3, 3, 3], one: [0.5, 0.5, 0.5, 0.5], row: 0},
  {mode: 'within' as const, all: [-1, -1, 7, 7], one: [1.5, 1.5, 4.5, 4.5], row: 2},
  {mode: 'contains' as const, all: [2.5, 2.5, 3.5, 3.5], one: [0.25, 0.25, 0.75, 0.75], row: 0}
]) {
  it(`bounds ${testCase.mode} overwrites every status word and clears overflow on reuse`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOutputOracleFixture(device);
    const {graph, input, output} = fixture;
    const valuesFor = ([minX, minY, maxX, maxY]: number[]) =>
      getGPUBoundsFilterParameterValues({minX, minY, maxX, maxY});
    const parameters = input('parameters', 'float32', valuesFor(testCase.all));
    const ids = output('ids', 'uint32', 2);
    const count = output('count', 'uint32', 1);
    const requiredCount = output('required', 'uint32', 1);
    const overflow = output('overflow', 'uint32', 1);
    graph.add(
      new GPUBoundsFilter({
        bounds: input('bounds', 'float32x4', Float32Array.of(0, 0, 6, 6, 1, 1, 5, 5, 2, 2, 4, 4)),
        mode: testCase.mode,
        parameters,
        output: {ids, status: {count, requiredCount, overflow}}
      })
    );
    const compiled = graph.compile();
    try {
      for (const frame of [
        {box: testCase.all, status: [2, 3, 1], ids: [0, 1]},
        {box: testCase.one, status: [1, 1, 0], ids: [testCase.row]},
        {box: [10, 10, 11, 11], status: [0, 0, 0], ids: []},
        {box: testCase.all, status: [2, 3, 1], ids: [0, 1]}
      ]) {
        fixture.writeInput(parameters, valuesFor(frame.box));
        const result = await fixture.submitTwice(compiled, async () => ({
          status: await fixture.readRows([count, requiredCount, overflow]),
          ids: await fixture.read(ids)
        }));
        expect(result.status).toEqual(frame.status);
        expect(result.ids.slice(0, frame.ids.length)).toEqual(frame.ids);
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}
