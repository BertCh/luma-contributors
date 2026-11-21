// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPULinearReferencing} from '../../../src/gpu-spatial-analysis/linear-referencing/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {addPrefixLocateConsumers} from './prefix-locate-fixture';

it('B2 location preserves exact endpoints across subtract-and-add cancellation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOutputOracleFixture(device);
  const positions = fixture.input('positions', 'float32x2', Float32Array.of(33554432, 0, 1, 0));
  const offsets = fixture.input('offsets', 'uint32', Uint32Array.of(0, 2));
  // The rounded length of this segment is 2^25. Its terminal coordinate remains exactly 1.
  const located = addPrefixLocateConsumers(fixture, positions, offsets, 2, 33554432);
  let compiled: ReturnType<typeof fixture.graph.compile> | undefined;
  try {
    compiled = fixture.graph.compile();
    const results = await fixture.submitTwice(compiled, async () =>
      Promise.all(
        located.map(async views => ({
          position: await fixture.read(views.position),
          status: await fixture.read(views.statuses)
        }))
      )
    );
    for (const result of results) {
      expect(result.position).toEqual([1, 0]);
      expect(result.status).toEqual([0]);
    }
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
});

it('W5-11 projection and distance/fraction location retain the compensated moving endpoint', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOutputOracleFixture(device);
  const {graph, input, output} = fixture;
  const coordinates = new Float32Array(386);
  for (let row = 1; row < 193; row++) {
    coordinates[row * 2] = Math.min(row, 64) * 16;
    coordinates[row * 2 + 1] = Math.max(row - 64, 0) * 2 ** -20;
  }
  // Independent binary arithmetic: total 64*16 + 128*2^-20 = 1024+2^-13;
  // the last point is (1024,2^-13), last segment fraction and normalized measure both 1.
  const positions = input('positions', 'float32x2', coordinates);
  const lineOffsets = input('offsets', 'uint32', Uint32Array.of(0, 193));
  const footPoints = output('foot', 'float32x2', 1);
  const measures = output('measures', 'float32', 1);
  const normalizedMeasures = output('normalized', 'float32', 1);
  const fractions = output('fractions', 'float32', 1);
  const candidateOverflow = output('candidate', 'uint32', 1);
  const projectedVertexMeasures = output('projected-prefix', 'float32', 193);
  graph.add(
    new GPULinearReferencing({
      positions,
      lineOffsets,
      queryPositions: input('query', 'float32x2', Float32Array.of(1024, 0.0001220703125)),
      radius: input('radius', 'float32', Float32Array.of(0.001)),
      candidateCapacity: 256,
      output: {
        footPoints,
        measures,
        normalizedMeasures,
        fractions,
        vertexMeasures: projectedVertexMeasures,
        status: {candidateOverflow}
      }
    })
  );
  const located = addPrefixLocateConsumers(
    fixture,
    positions,
    lineOffsets,
    193,
    1024.0001220703125
  );
  let compiled: ReturnType<typeof graph.compile> | undefined;
  try {
    compiled = graph.compile();
    const result = await fixture.submitTwice(compiled, async () => ({
      foot: await fixture.read(footPoints),
      measures: await fixture.read(measures),
      normalized: await fixture.read(normalizedMeasures),
      fractions: await fixture.read(fractions),
      candidate: await fixture.read(candidateOverflow),
      prefix: await fixture.read(projectedVertexMeasures),
      located: await Promise.all(
        located.map(async views => ({
          position: await fixture.read(views.position),
          prefix: await fixture.read(views.vertexMeasures),
          status: await fixture.read(views.statuses)
        }))
      )
    }));
    expect(result.foot.every(Number.isFinite)).toBe(true);
    expect(result.foot).toEqual([1024, 0.0001220703125]);
    expect(result.measures).toEqual([1024.0001220703125]);
    expect(result.normalized).toEqual([1]);
    expect(result.fractions).toEqual([1]);
    expect(result.candidate).toEqual([0]);
    expect(result.prefix).toHaveLength(193);
    expect(result.prefix[192]).toBe(1024.0001220703125);
    for (const location of result.located) {
      expect(location.position.every(Number.isFinite)).toBe(true);
      expect(location.position).toEqual([1024, 0.0001220703125]);
      expect(location.prefix).toHaveLength(193);
      expect(location.prefix[192]).toBe(1024.0001220703125);
      expect(location.status).toEqual([0]);
    }
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
});

it('B7 projects the 65-increment endpoint using its published next-row measure', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOutputOracleFixture(device);
  const {graph, input, output} = fixture;
  const coordinates = new Float32Array(130 * 2);
  for (let row = 1; row < 130; row++) {
    coordinates[row * 2] = Math.min(row, 64) * 16;
    coordinates[row * 2 + 1] = Math.max(row - 64, 0) * 2 ** -20;
  }
  // Independent exact length = 1024+65*2^-20, rounded to 1024+2^-13.
  // Previous row 1024+64*2^-20 is a half-ulp tie and rounds to even 1024.
  const vertexMeasures = output('prefix', 'float32', 130),
    measures = output('measure', 'float32', 1),
    normalizedMeasures = output('normalized', 'float32', 1),
    fractions = output('fraction', 'float32', 1),
    footPoints = output('foot', 'float32x2', 1),
    candidateOverflow = output('candidate', 'uint32', 1);
  graph.add(
    new GPULinearReferencing({
      positions: input('positions', 'float32x2', coordinates),
      lineOffsets: input('offsets', 'uint32', Uint32Array.of(0, 130)),
      queryPositions: input('query', 'float32x2', Float32Array.of(1024, 0.00006198883056640625)),
      radius: input('radius', 'float32', Float32Array.of(0.001)),
      candidateCapacity: 256,
      output: {
        vertexMeasures,
        measures,
        normalizedMeasures,
        fractions,
        footPoints,
        status: {candidateOverflow}
      }
    })
  );
  let compiled: ReturnType<typeof graph.compile> | undefined;
  try {
    compiled = graph.compile();
    const result = await fixture.submitTwice(compiled, async () => ({
      prefix: await fixture.read(vertexMeasures),
      measures: await fixture.read(measures),
      normalized: await fixture.read(normalizedMeasures),
      fractions: await fixture.read(fractions),
      foot: await fixture.read(footPoints),
      candidate: await fixture.read(candidateOverflow)
    }));
    expect(result.prefix).toHaveLength(130);
    expect(result.prefix.slice(-2)).toEqual([1024, 1024.0001220703125]);
    expect(result.measures).toEqual([1024.0001220703125]);
    expect(result.normalized).toEqual([1]);
    expect(result.fractions).toEqual([1]);
    expect(result.foot).toEqual([1024, 0.00006198883056640625]);
    expect(result.candidate).toEqual([0]);
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
});
