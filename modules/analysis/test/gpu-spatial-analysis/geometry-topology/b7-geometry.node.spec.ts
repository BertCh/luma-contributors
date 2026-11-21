// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {
  createGroupedPruningCase,
  createGroupedRingCase,
  createNodingParameterCase,
  createRingBucketCase,
  createRingExtentCase,
  NODING_PARAMETER_CASES,
  type CreateGeometryFixture
} from './b7-geometry-fixtures';
import {getOracleView, type OracleAllocate} from './noding-polygonize-output-oracles-fixtures';

useCPUExecutorTestHooks();

function runFixture(createFixture: CreateGeometryFixture) {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    const allocate: OracleAllocate = (name, format, length, values) => {
      const view = createTransientView(graph, name, format, length);
      if (values) {
        memory.write(
          view,
          format.startsWith('uint') ? Uint32Array.from(values) : Float32Array.from(values)
        );
      }
      return view;
    };
    const fixture = createFixture(graph, allocate);
    executeCommandNodesOnCPU(fixture.nodes, memory);
    for (const [path, values] of Object.entries(fixture.expected)) {
      const view = getOracleView(fixture.output, path);
      expect((view.length * view.rowByteLength) / 4, path).toBe(values.length);
      const actual = view.format.startsWith('uint')
        ? memory.readUint32(view)
        : memory.readFloat32(view);
      expect(Array.from(actual), path).toEqual(values);
    }
  } finally {
    device.destroy();
  }
}

function runRingFixture(createFixture: CreateGeometryFixture) {
  runFixture((graph, allocate) => {
    const fixture = createFixture(graph, allocate);
    // These outputs are independent of area and winding. The CPU executor
    // cannot synchronize the cooperative area kernel.
    const nodes = fixture.nodes.filter(
      node => !node.id.endsWith('-ring-stats') && !node.id.endsWith('-ring-orientation')
    );
    return {...fixture, nodes};
  });
}

it.each(NODING_PARAMETER_CASES)('line noding preserves %s split parameters', geometry => {
  runFixture((graph, allocate) => createNodingParameterCase(graph, allocate, geometry));
});

const round = Math.fround;
const floatWord = new Float32Array(1);
const uintWord = new Uint32Array(floatWord.buffer);
const zeroProductPower = -512;

function magnitudePower(value: number): number {
  floatWord[0] = value;
  const magnitude = uintWord[0] & 0x7fffffff;
  const exponent = magnitude >>> 23;
  return exponent === 0 ? 31 - Math.clz32(magnitude) - 149 : exponent - 127;
}

function productPower(first: number, second: number): number {
  return first === 0 || second === 0
    ? zeroProductPower
    : magnitudePower(first) + magnitudePower(second);
}

function needsScaling(dx: number, dy: number, offsetX: number, offsetY: number): boolean {
  const edgeMagnitude = Math.max(Math.abs(dx), Math.abs(dy));
  const numeratorPower = Math.max(productPower(dx, offsetX), productPower(dy, offsetY));
  return (
    edgeMagnitude < 2 ** -63 ||
    edgeMagnitude > 2 ** 63 ||
    Math.max(Math.abs(offsetX), Math.abs(offsetY)) > 2 ** 63 ||
    (numeratorPower !== zeroProductPower && numeratorPower < -126)
  );
}

function quotient(numerator: number, denominator: number, reciprocal: boolean): number {
  return reciprocal ? round(numerator * round(1 / denominator)) : round(numerator / denominator);
}

it('pins baseline horizontal bits under direct division and reciprocal multiplication', () => {
  const parameter = (length: number, cut: number, reciprocal: boolean) =>
    quotient(round(cut * length), round(length * length), reciprocal);
  expect(parameter(529, 504.203125, true)).toBe(61 / 64);
  expect(parameter(529, 504.203125, false)).toBe(0.9531249403953552);
  const cuts = [2.750000476837158, 2.7500007152557373];
  const distinct = cuts.map(cut => parameter(11, cut, true));
  expect(distinct).toEqual([0.2500000298023224, 0.2500000596046448]);
  expect(distinct[1] - distinct[0]).toBe(2 ** -25);
  expect(cuts.map(cut => parameter(11, cut, false))).toEqual(new Array(2).fill(distinct[1]));
  expect(parameter(5, 3, false)).toBe(round(3 / 5));
  expect(parameter(5, 3, true)).toBe(0.5999999642372131);
  expect([8.25, 11].map(cut => parameter(11, cut, false))).toEqual([0.75, 1]);
  expect([8.25, 11].map(cut => parameter(11, cut, true))).toEqual([
    0.7499999403953552, 0.9999999403953552
  ]);
});

it('preserves pre-stream float32 parameter bits throughout the ordinary exponent range', () => {
  let compared = 0;
  let differences = 0;
  let scaled = 0;
  const fractions = [
    0,
    1,
    ...Array.from({length: 63}, (_, row) => (row + 1) / 64),
    ...Array.from({length: 64}, (_, row) => 0.25 + row * 2 ** -25)
  ];
  for (let power = -40; power <= 40; power++) {
    for (let row = 0; row < 64; row++) {
      const scale = 2 ** power;
      const length = (1 + row / 16) * scale;
      for (const [edgeX, edgeY] of [
        [length, 0],
        [0, -length],
        [length, -length * 0.3125]
      ]) {
        const startX = row % 2 === 0 ? 0 : 7 * scale;
        const startY = row % 2 === 0 ? 0 : -3 * scale;
        const dx = round(round(startX + edgeX) - startX);
        const dy = round(round(startY + edgeY) - startY);
        for (const fraction of fractions) {
          const offsetX = round(round(startX + dx * fraction) - startX);
          const offsetY = round(round(startY + dy * fraction) - startY);
          const needsScale = needsScaling(dx, dy, offsetX, offsetY);
          if (needsScale) {
            scaled++;
            continue;
          }
          const baselineNumerator = round(round(offsetX * dx) + round(offsetY * dy));
          const baselineDenominator = round(round(dx * dx) + round(dy * dy));
          // The gated path performs the untouched dot products and clamp in either division mode.
          for (const reciprocal of [false, true]) {
            const baseline = Math.min(
              1,
              Math.max(
                0,
                reciprocal
                  ? round(baselineNumerator * round(1 / baselineDenominator))
                  : round(baselineNumerator / baselineDenominator)
              )
            );
            const ordinary = Math.min(
              1,
              Math.max(
                0,
                quotient(
                  round(round(offsetX * dx) + round(offsetY * dy)),
                  round(round(dx * dx) + round(dy * dy)),
                  reciprocal
                )
              )
            );
            if (!Object.is(baseline, ordinary)) differences++;
            compared++;
          }
        }
      }
    }
  }
  expect(compared).toBe(4_012_416);
  expect(differences).toBe(0);
  expect(scaled).toBe(0);
  expect(needsScaling(2 ** -63, 0, 2 ** -88, 0)).toBe(true);
  expect(needsScaling(2 ** -63, 0, 2 ** -63, 0)).toBe(false);
  expect(needsScaling(2 ** -80, 0, 2 ** -81, 0)).toBe(true);
  expect(round(2 ** -88 / 2 ** -63)).toBe(2 ** -25);
});

it.each([
  {squareCount: 8, distant: false},
  {squareCount: 8, distant: true},
  {squareCount: 32, distant: true}
])('polygonize candidate visits stay local for $squareCount squares with distant group=$distant', ({
  squareCount,
  distant
}) => {
  runFixture((graph, allocate) => createGroupedPruningCase(graph, allocate, squareCount, distant));
});

it.each([
  'explicit',
  'derived'
] as const)('ring assembly matches across a cell boundary with %s tolerance', tolerance => {
  runRingFixture((graph, allocate) => createGroupedRingCase(graph, allocate, tolerance));
});

it.each([
  false,
  true
])('ring assembly local hash buckets stay sparse with distant group=%s', distant => {
  runFixture((graph, allocate) => createRingBucketCase(graph, allocate, distant));
});

it('ring assembly reports overflow on the rectangle spanning opposite float32 extremes', () => {
  expect(round(2 ** 127 - -(2 ** 127))).toBe(Infinity);
  runRingFixture(createRingExtentCase);
});
