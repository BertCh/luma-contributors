// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUDotDensity,
  GPURandomPointsInPolygon,
  GPURandomPointsOnLine,
  getGPUDotDensityParameterValues
} from '../../../src/gpu-spatial-analysis/dot-density/index';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {matchesTriangleBarycentricOracle} from './dot-density-output-oracle-fixtures';

for (const includeCategories of [true, false]) {
  it(`writes exact integer-request category IDs, categories ${includeCategories}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOutputOracleFixture(device);
    const {graph, input, output} = fixture;
    const categories = includeCategories ? output('categories', 'uint32', 5) : undefined;
    const count = output('count', 'uint32', 1),
      overflow = output('overflow', 'uint32', 1),
      ids = output('ids', 'uint32', 5);
    graph.add(
      new GPUDotDensity({
        positions: input('vertices', 'float32x2', new Float32Array([0, 0, 1, 0, 0, 1])),
        featureOffsets: input('features', 'uint32', new Uint32Array([0, 1])),
        polygonOffsets: input('polygons', 'uint32', new Uint32Array([0, 1])),
        ringOffsets: input('rings', 'uint32', new Uint32Array([0, 3])),
        values: input('values', 'float32', new Float32Array([2, 3])),
        categoryCount: 2,
        parameters: input('parameters', 'uint32', getGPUDotDensityParameterValues({seed: 123})),
        output: {
          positions: output('positions', 'float32x2', 5),
          categories,
          dots: {ids, status: {count, overflow}}
        }
      })
    );
    const compiled = graph.compile();
    try {
      const actual = await fixture.submitTwice(compiled, async () => ({
        categories: categories ? await fixture.readRows([categories]) : [],
        count: await fixture.readRows([count]),
        overflow: await fixture.readRows([overflow]),
        ids: await fixture.readRows([ids])
      }));
      // Integer requests are exact and need no Bernoulli remainder decision.
      expect(actual).toEqual({
        categories: includeCategories ? [0, 0, 1, 1, 1] : [],
        count: [5],
        overflow: [0],
        ids: [0, 0, 0, 0, 0]
      });
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}

it('pins triangle samples to independent Python Philox barycentric weights', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOutputOracleFixture(device);
  const {graph, input, output} = fixture;
  const positions = output('positions', 'float32x2', 3),
    count = output('count', 'uint32', 1),
    overflow = output('overflow', 'uint32', 1);
  graph.add(
    new GPURandomPointsInPolygon({
      positions: input('vertices', 'float32x2', new Float32Array([0, 0, 1, 0, 0, 1])),
      featureOffsets: input('features', 'uint32', new Uint32Array([0, 1])),
      polygonOffsets: input('polygons', 'uint32', new Uint32Array([0, 1])),
      ringOffsets: input('rings', 'uint32', new Uint32Array([0, 3])),
      counts: input('requests', 'uint32', new Uint32Array([3])),
      parameters: input('parameters', 'uint32', getGPUDotDensityParameterValues({seed: 123})),
      output: {positions, dots: {ids: output('ids', 'uint32', 3), status: {count, overflow}}}
    })
  );
  const compiled = graph.compile();
  try {
    const actual = await fixture.submitTwice(compiled, () => fixture.readRows([positions]));
    // The triangulator may choose a different vertex start. A single common
    // permutation must explain all draws; per-draw sorting admits a biased sampler.
    const barycentric = Array.from({length: 3}, (_, row) => [
      1 - actual[2 * row] - actual[2 * row + 1],
      actual[2 * row],
      actual[2 * row + 1]
    ]);
    expect(matchesTriangleBarycentricOracle(barycentric), JSON.stringify(barycentric)).toBe(true);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

for (const optional of [undefined, 'fractions', 'candidateOverflow'] as const) {
  for (const requested of optional === 'fractions' ? [3] : [3, 512]) {
    it(`writes independent line fractions and clamp diagnostics: ${optional ?? 'absent'}, request ${requested}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = createOutputOracleFixture(device);
      const {graph, input, output} = fixture;
      const capacity = requested === 512 ? 1 : 3;
      const positions = output('positions', 'float32x2', capacity),
        ids = output('ids', 'uint32', capacity),
        count = output('count', 'uint32', 1),
        overflow = output('overflow', 'uint32', 1);
      const fractions =
        optional === 'fractions' ? output('fractions', 'float32', capacity) : undefined;
      const candidateOverflow =
        optional === 'candidateOverflow' ? output('candidate', 'uint32', 1) : undefined;
      graph.add(
        new GPURandomPointsOnLine({
          positions: input('vertices', 'float32x2', new Float32Array([0, 0, 1, 0])),
          lineOffsets: input('lines', 'uint32', new Uint32Array([0, 2])),
          counts: input('requests', 'uint32', new Uint32Array([requested])),
          parameters: input('parameters', 'uint32', getGPUDotDensityParameterValues({seed: 7})),
          output: {
            positions,
            points: {ids, status: {count, overflow, candidateOverflow}},
            fractions
          }
        })
      );
      const compiled = graph.compile();
      try {
        const actual = await fixture.submitTwice(compiled, async () => ({
          positions: await fixture.readRows([positions]),
          count: await fixture.readRows([count]),
          overflow: await fixture.readRows([overflow]),
          fractions: fractions ? await fixture.readRows([fractions]) : [],
          candidate: candidateOverflow ? await fixture.readRows([candidateOverflow]) : []
        }));
        const expected = [0.20142346620559692, 0.20847898721694946, 0.09203100204467773].slice(
          0,
          capacity
        );
        expect(actual.count).toEqual([capacity]);
        expect(actual.overflow).toEqual([requested === 512 ? 1 : 0]);
        expect(actual.candidate).toEqual(candidateOverflow ? [requested === 512 ? 1 : 0] : []);
        expect(actual.fractions).toEqual(fractions ? expected : []);
        expected.forEach((value, row) => {
          expect(actual.positions[2 * row]).toBeCloseTo(value, 6);
          expect(actual.positions[2 * row + 1]).toBe(0);
        });
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    });
  }
}
