// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeModalSplitOnCPU,
  CPUModalSplitOperation
} from '../../../src/gpu-spatial-analysis/trajectory-mobility/cpu-modal-split-operation';
import {GPUModalSplitOperation} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-modal-split-operation';
import {getModalFixture, getModalProps} from './modal-split-operation-fixture';

it('counts supplied-mode empty triplegs and gives them zero distance without reading times', () => {
  for (const start of [0, 1, 2]) {
    const input = {
      positions: new Float32Array([0, 0, 1, 0]),
      timestamps: new Float32Array([0, 1]),
      trackOffsets: new Uint32Array([0, 2]),
      triplegs: {
        tracks: Uint32Array.of(0),
        startRows: Uint32Array.of(start),
        endRows: Uint32Array.of(start),
        count: Uint32Array.of(1)
      },
      modes: Uint32Array.of(0)
    };
    // One table row contributes one count; its empty path contains no distance terms.
    expect(
      Array.from(computeModalSplitOnCPU(input, {modeCount: 1, metric: 'count'}).modalSplit)
    ).toEqual([1]);
    expect(
      computeModalSplitOnCPU(input, {modeCount: 1, metric: 'distance'}).modalSplit[0] === 0
    ).toBe(true);
    expect(() => computeModalSplitOnCPU(input, {modeCount: 1, metric: 'duration'})).toThrow(
      /triplegs.startRows/
    );
    expect(() =>
      computeModalSplitOnCPU(input, {modeCount: 1, metric: 'count', modeSpeedBreaks: [1]})
    ).toThrow(/triplegs.startRows/);
  }
});

it('matches hand-counted modes, lengths and durations, excluding a missing speed', () => {
  for (const [metric, predictedExpected, suppliedExpected] of [
    ['count', [1, 1, 0, 0, 1, 1], [1, 0, 1, 0, 2, 0]],
    ['distance', [3, 12, 0, 0, 6, 40], [20, 0, 3, 0, 46, 0]],
    ['duration', [3, 2, 0, 0, 2, 2], [0, 0, 3, 0, 4, 0]]
  ] as const) {
    const input = getModalFixture();
    const predicted = computeModalSplitOnCPU(input, {
      metric,
      modeCount: 3,
      modeSpeedBreaks: [2, 10, Infinity]
    });
    expect(Array.from(predicted.modalSplit)).toEqual(predictedExpected);
    expect(Array.from(predicted.predictedModes)).toEqual([0, 1, 0xffffffff, 1, 2, 0]);
    const supplied = computeModalSplitOnCPU(input, {metric, modeCount: 3});
    expect(Array.from(supplied.modalSplit)).toEqual(suppliedExpected);
    expect(Array.from(supplied.predictedModes)).toEqual(new Array(6).fill(0));
    expect(Array.from(input.modes)).toEqual([9, 0xffffffff, 0, 1, 1, 99]);
  }
});

it('uses exclusive speed breaks and leaves the final speed class unbounded', () => {
  const input = getModalFixture();
  const result = computeModalSplitOnCPU(input, {
    metric: 'count',
    modeCount: 3,
    modeSpeedBreaks: [1, 6, 7]
  });
  expect(Array.from(result.predictedModes)).toEqual([1, 2, 0xffffffff, 1, 2, 0]);
  const empty = computeModalSplitOnCPU(
    {...input, triplegs: {...input.triplegs, count: Uint32Array.of(0)}},
    {metric: 'distance', modeCount: 3, modeSpeedBreaks: [2, 10, Infinity]}
  );
  expect(empty.modalSplit.every(value => value === 0)).toBe(true);
  expect(empty.predictedModes.every(value => value === 0)).toBe(true);
});

it('uses equatorial arc length rather than longitude degrees for a spherical tripleg', () => {
  const result = computeModalSplitOnCPU(
    {
      positions: new Float32Array([179, 0, -179, 0, -177, 0]),
      timestamps: new Float32Array([0, 1, 2]),
      trackOffsets: new Uint32Array([0, 3]),
      triplegs: {
        tracks: Uint32Array.of(0),
        startRows: Uint32Array.of(0),
        endRows: Uint32Array.of(3),
        count: Uint32Array.of(1)
      },
      modes: Uint32Array.of(0)
    },
    {
      modeCount: 1,
      metric: 'distance',
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        units: 'meters',
        sphereRadius: 1
      }
    }
  );
  expect(result.modalSplit[0]).toBe(Math.fround((4 * Math.PI) / 180));
});

it('checks malformed arrays and both operation constructors', () => {
  const input = getModalFixture();
  expect(() =>
    computeModalSplitOnCPU({...input, modes: new Uint32Array(1)}, {modeCount: 3, metric: 'count'})
  ).toThrow(/modes/);
  expect(() =>
    computeModalSplitOnCPU(
      {...input, timestamps: new Float32Array(1)},
      {modeCount: 3, metric: 'count'}
    )
  ).toThrow(/timestamps/);
  expect(() =>
    computeModalSplitOnCPU(
      {...input, triplegs: {...input.triplegs, endRows: Uint32Array.of(1)}},
      {modeCount: 3, metric: 'count'}
    )
  ).toThrow(/triplegs.endRows/);
  expect(() =>
    computeModalSplitOnCPU(
      {...input, triplegs: {...input.triplegs, startRows: new Uint32Array([4, 3, 6, 9, 12, 99])}},
      {modeCount: 3, metric: 'count'}
    )
  ).toThrow(/triplegs.startRows/);
  const props = getModalProps('count', true);
  for (const Operation of [GPUModalSplitOperation, CPUModalSplitOperation]) {
    expect(() => new Operation({...props, modeCount: 0})).toThrow(/^modal-test /);
    expect(
      () => new Operation({...props, output: {...props.output, modalSplit: 'positions'}})
    ).toThrow(/^modal-test /);
  }
  expect(
    () =>
      new CPUModalSplitOperation({...props, modes: undefined, modeSpeedBreaks: [2, 1, Infinity]})
  ).toThrow(/^modal-test modeSpeedBreaks/);
  expect(() => new CPUModalSplitOperation({...props, trackCount: 0})).toThrow(
    /^modal-test trackCount/
  );
});
