// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUCellLevelSelection} from '../../../src/gpu-spatial-analysis/cell-aggregation/gpu-cell-level-selection';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const includeRequiredCount of [true, false]) {
  it(`selects literal level counts with requiredCount ${includeRequiredCount}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOutputOracleFixture(device);
    const {graph, input, output} = fixture;
    const count = output('count', 'uint32', 1);
    const requiredCount = includeRequiredCount ? output('required', 'uint32', 1) : undefined;
    graph.add(
      new GPUCellLevelSelection({
        levelCounts: input('counts', 'uint32', new Uint32Array([5, 17, 0])),
        activeLevel: input('active', 'uint32', new Uint32Array([1])),
        output: {status: {count, requiredCount}}
      })
    );
    const compiled = graph.compile();
    try {
      const actual = await fixture.submitTwice(compiled, async () => ({
        count: await fixture.readRows([count]),
        requiredCount: requiredCount ? await fixture.readRows([requiredCount]) : []
      }));
      expect(actual).toEqual({count: [17], requiredCount: includeRequiredCount ? [17] : []});
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}
