// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  getGPULineSmoothParameterValues,
  GPULineSmooth
} from '../../../src/gpu-spatial-analysis/line-segmentize/gpu-line-smooth';
import {createFlatLines, createRandom, type FlatLines} from './line-segmentize-oracle';
import {smoothLines} from './line-smooth-oracle';
import {createLineLineFixture, expectLineLineParity} from './line-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

function createSmoothFixture(
  device: Device,
  lines: FlatLines,
  options: {capacity: number; iterationCount: number; closed: boolean}
) {
  return createLineLineFixture(device, {
    inputs: {
      positions: {values: lines.positions, format: 'float32x2'},
      lineOffsets: {values: lines.lineOffsets, format: 'uint32'}
    },
    capacity: options.capacity,
    lineCapacity: lines.lineOffsets.length - 1,
    withVertexColumns: false,
    createContributor: (inputs, parameters, output) =>
      new GPULineSmooth({
        positions: inputs['positions'] as GraphDataView<'float32x2'>,
        lineOffsets: inputs['lineOffsets'] as GraphDataView<'uint32'>,
        iterationCount: options.iterationCount,
        closed: options.closed,
        parameters,
        output
      })
  });
}

function createRandomLines(seed: number, closed: boolean): FlatLines {
  const random = createRandom(seed);
  const lines = Array.from({length: 150}, (_, lineIndex) => {
    const length =
      lineIndex % 23 === 2
        ? 0
        : lineIndex % 19 === 3
          ? 1
          : lineIndex % 17 === 4
            ? 2
            : 3 + Math.floor(random() * 7);
    const vertices = Array.from({length}, () => [random() * 100, random() * 100]);
    // Some closed rings repeat their first vertex.
    if (closed && length >= 3 && lineIndex % 2 === 0) {
      vertices.push(vertices[0]);
    }
    return vertices;
  });
  return createFlatLines(lines);
}

for (const closed of [false, true]) {
  it(`GPULineSmooth matches the Chaikin oracle for ${closed ? 'closed rings' : 'open lines'}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const lines = createRandomLines(closed ? 2 : 1, closed);
    for (const iterationCount of [1, 2, 3]) {
      const capacity = (lines.positions.length / 2) * 2 ** iterationCount + 200;
      const fixture = createSmoothFixture(device, lines, {capacity, iterationCount, closed});
      for (const ratio of [0.25, 0.1, 0.5]) {
        const actual = await fixture.run(getGPULineSmoothParameterValues({ratio}));
        const expected = smoothLines(lines, {iterationCount, ratio, closed});
        expectLineLineParity(actual, expected, capacity, 1e-4, 0);
        expect(actual.overflow).toBe(0);
      }
      expect(fixture.getCompileCount()).toBe(0);
      fixture.destroy();
    }
  });
}

it('GPULineSmooth truncates at capacity with valid offsets', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines = createRandomLines(4, false);
  const capacity = 500;
  const iterationCount = 3;
  const fixture = createSmoothFixture(device, lines, {capacity, iterationCount, closed: false});
  const actual = await fixture.run(getGPULineSmoothParameterValues());
  const expected = smoothLines(lines, {iterationCount, ratio: 0.25, closed: false});
  expect(expected.positions.length).toBeGreaterThan(capacity);
  expectLineLineParity(actual, expected, capacity, 1e-4, 0);
  expect(actual.overflow).toBe(1);
  fixture.destroy();
});

it('GPULineSmooth keeps endpoints and doubles open lines', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines = createFlatLines([
    [
      [0, 0],
      [4, 0],
      [4, 4]
    ],
    [[1, 1]]
  ]);
  const fixture = createSmoothFixture(device, lines, {
    capacity: 16,
    iterationCount: 1,
    closed: false
  });
  const actual = await fixture.run(getGPULineSmoothParameterValues());
  expect(actual.positions).toEqual([
    [0, 0],
    [1, 0],
    [3, 0],
    [4, 1],
    [4, 3],
    [4, 4],
    [1, 1]
  ]);
  expect(actual.lineOffsets).toEqual([0, 6, 7]);
  fixture.destroy();
});

for (const closed of [false, true])
  it(`publishes source rows and chainage with empty lines and repeated encodings (closed=${closed})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const lines = createFlatLines([
      [],
      closed
        ? [
            [0, 0],
            [4, 0],
            [4, 4],
            [0, 4]
          ]
        : [
            [0, 0],
            [4, 0]
          ],
      []
    ]);
    const fixture = createLineLineFixture(device, {
      inputs: {
        positions: {values: lines.positions, format: 'float32x2'},
        lineOffsets: {values: lines.lineOffsets, format: 'uint32'}
      },
      capacity: closed ? 9 : 4,
      lineCapacity: 3,
      withVertexColumns: true,
      createContributor: (inputs, parameters, output) =>
        new GPULineSmooth({
          positions: inputs['positions'] as GraphDataView<'float32x2'>,
          lineOffsets: inputs['lineOffsets'] as GraphDataView<'uint32'>,
          iterationCount: 1,
          closed,
          parameters,
          output
        })
    });
    try {
      for (const ratio of [0.25, 0.5, 0.25]) {
        const actual = await fixture.run(getGPULineSmoothParameterValues({ratio}));
        if (closed) {
          expect(actual.sourceRows).toEqual(
            ratio === 0.5 ? [0, 0, 1, 1, 2, 2, 3, 3, 0] : [0, 0, 1, 1, 2, 2, 3, 3, 0]
          );
          expect(actual.measures).toEqual(
            Array.from(
              {length: 8},
              (_, index) => 4 * Math.floor(index / 2) + 4 * (index % 2 === 0 ? ratio : 1 - ratio)
            ).concat([16 + 4 * ratio])
          );
        } else {
          expect(actual.positions).toEqual([
            [0, 0],
            [4 * ratio, 0],
            [4 * (1 - ratio), 0],
            [4, 0]
          ]);
          expect(actual.sourceRows).toEqual([0, 0, 0, 1]);
          expect(actual.measures).toEqual([0, 4 * ratio, 4 * (1 - ratio), 4]);
        }
        expect(actual.overflow).toBe(0);
      }
    } finally {
      fixture.destroy();
    }
  });

it('keeps ordered source rows and measures for repeated closed rings through multiple truncated iterations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines = createFlatLines([
    [],
    [
      [0, 0],
      [6, 0],
      [6, 2],
      [0, 2],
      [0, 0]
    ],
    []
  ]);
  for (const iterationCount of [2, 3])
    for (const capacity of [1, 7, 4 * 2 ** iterationCount + 1]) {
      const fixture = createLineLineFixture(device, {
        inputs: {
          positions: {values: lines.positions, format: 'float32x2'},
          lineOffsets: {values: lines.lineOffsets, format: 'uint32'}
        },
        capacity,
        lineCapacity: 3,
        withVertexColumns: true,
        createContributor: (inputs, parameters, output) =>
          new GPULineSmooth({
            positions: inputs['positions'] as GraphDataView<'float32x2'>,
            lineOffsets: inputs['lineOffsets'] as GraphDataView<'uint32'>,
            iterationCount,
            closed: true,
            parameters,
            output
          })
      });
      try {
        for (const ratio of [0.25, 0.5, 0.25]) {
          // Independent f64 Chaikin oracle carries an unwrapped source parameter and chainage.
          let vertices = [
            [0, 0, 0, 0],
            [6, 0, 1, 6],
            [6, 2, 2, 8],
            [0, 2, 3, 14]
          ];
          for (let iteration = 0; iteration < iterationCount; iteration++) {
            const next: number[][] = [];
            for (let edge = 0; edge < vertices.length; edge++) {
              const start = vertices[edge];
              const end = [...vertices[(edge + 1) % vertices.length]];
              if (edge + 1 === vertices.length) {
                end[2] += 4;
                end[3] += 16;
              }
              for (const weight of [ratio, 1 - ratio])
                next.push(
                  start.map((value, component) => value * (1 - weight) + end[component] * weight)
                );
            }
            vertices = next;
          }
          const closing = [...vertices[0]];
          closing[2] += 4;
          closing[3] += 16;
          vertices.push(closing);
          const expected = vertices.slice(0, capacity);
          const actual = await fixture.run(getGPULineSmoothParameterValues({ratio}));
          expect(actual.positions).toEqual(expected.map(value => value.slice(0, 2)));
          expect(actual.sourceRows).toEqual(expected.map(value => Math.floor(value[2]) % 4));
          expect(actual.measures).toEqual(expected.map(value => value[3]));
          expect(actual.lineOffsets).toEqual([0, 0, expected.length, expected.length]);
          expect(actual.requiredVertexCount).toBe(vertices.length);
          expect(actual.overflow).toBe(Number(capacity < vertices.length));
        }
      } finally {
        fixture.destroy();
      }
    }
});
