// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUBoundsFilter,
  getGPUBoundsFilterParameterValues
} from '../../../src/gpu-spatial-analysis/bounds-filter/index';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const includeCandidateOverflow of [true, false]) {
  it(`writes the exact bounds selection with candidateOverflow ${includeCandidateOverflow}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOutputOracleFixture(device);
    const {graph, input, output} = fixture;
    const ids = output('ids', 'uint32', 1);
    const count = output('count', 'uint32', 1);
    const overflow = output('overflow', 'uint32', 1);
    const candidateOverflow = includeCandidateOverflow
      ? output('candidate-overflow', 'uint32', 1)
      : undefined;
    graph.add(
      new GPUBoundsFilter({
        // Rows 0 and 2 meet the unit query; final capacity drops row 2. There is
        // no bounded candidate stage, so final overflow must not set candidateOverflow.
        bounds: input(
          'bounds',
          'float32x4',
          new Float32Array([0, 0, 1, 1, 3, 3, 4, 4, 0.5, 0.5, 2, 2])
        ),
        parameters: input(
          'parameters',
          'float32',
          getGPUBoundsFilterParameterValues({minX: 0, minY: 0, maxX: 1, maxY: 1})
        ),
        output: {ids, status: {count, overflow, candidateOverflow}}
      })
    );
    const compiled = graph.compile();
    try {
      const actual = await fixture.submitTwice(compiled, async () => ({
        ids: await fixture.readRows([ids]),
        count: await fixture.readRows([count]),
        overflow: await fixture.readRows([overflow]),
        candidateOverflow: candidateOverflow ? await fixture.readRows([candidateOverflow]) : []
      }));
      expect(actual).toEqual({
        ids: [0],
        count: [1],
        overflow: [1],
        candidateOverflow: includeCandidateOverflow ? [0] : []
      });
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}
