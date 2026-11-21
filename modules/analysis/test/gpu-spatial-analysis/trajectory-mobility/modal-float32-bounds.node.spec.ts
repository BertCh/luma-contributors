// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {computeModalSplitOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-mobility/cpu-modal-split-operation';
const input = {
  positions: Float32Array.of(0, 0, 15 / 3.6, 0),
  timestamps: Float32Array.of(0, 1),
  trackOffsets: Uint32Array.of(0, 2),
  triplegs: {
    tracks: Uint32Array.of(0),
    startRows: Uint32Array.of(0),
    endRows: Uint32Array.of(2),
    count: Uint32Array.of(1)
  }
};
it('uses float32 exclusive speed upper bounds at 15 km/hour', () => {
  const result = computeModalSplitOnCPU(input, {
    modeCount: 3,
    metric: 'count',
    modeSpeedUpperBoundsFloat32: [15 / 3.6, 100 / 3.6, Infinity]
  });
  // Stored endpoint is exactly the encoded cutoff, so it belongs to the second bin.
  expect(Array.from(result.predictedModes)).toEqual([1]);
  expect(Array.from(result.modalSplit)).toEqual([0, 1, 0]);
});
it('rejects invalid and collapsed encoded modal tables', () => {
  for (const breaks of [
    [NaN],
    [-Infinity, Infinity],
    [Infinity, Infinity],
    [0.7, 0.7000000001, Infinity]
  ]) {
    expect(() =>
      computeModalSplitOnCPU(input, {
        modeCount: breaks.length,
        metric: 'count',
        modeSpeedUpperBoundsFloat32: breaks
      })
    ).toThrow(/modeSpeed/);
  }
});
it('exposes unknown supplied mode policy and no fictitious prediction', () => {
  for (const unknownModePolicy of ['last-bin', 'exclude'] as const) {
    const result = computeModalSplitOnCPU(
      {...input, modes: Uint32Array.of(7)},
      {modeCount: 2, metric: 'count', unknownModePolicy}
    );
    expect(Array.from(result.modalSplit)).toEqual(
      unknownModePolicy === 'last-bin' ? [0, 1] : [0, 0]
    );
    expect(Array.from(result.predictedModes)).toEqual([0xffffffff]);
  }
});
it('accepts the canonical float32 option and rejects the removed spelling', () => {
  const options = {
    modeCount: 3,
    metric: 'count' as const,
    modeSpeedUpperBoundsFloat32: [15 / 3.6, 100 / 3.6, Infinity]
  };
  expect(Array.from(computeModalSplitOnCPU(input, options).modalSplit)).toEqual([0, 1, 0]);
  const retired = {...options, modeSpeedBreaks: options.modeSpeedUpperBoundsFloat32};
  expect(() => computeModalSplitOnCPU(input, retired)).toThrow(
    /modeSpeedBreaks was removed.*modeSpeedUpperBoundsFloat32/
  );
  expect(() =>
    computeModalSplitOnCPU(input, {...options, unknownModePolicy: 'unrecognized' as never})
  ).toThrow(/last-bin or exclude/);
});
it('B4 classifies an exact non-unit-duration boundary without reciprocal division', () => {
  const result = computeModalSplitOnCPU(
    {...input, positions: Float32Array.of(0, 0, 5, 0), timestamps: Float32Array.of(0, 25)},
    {modeCount: 2, metric: 'count', modeSpeedUpperBoundsFloat32: [Math.fround(0.2), Infinity]}
  );
  // Stored f32(.2) is greater than exact 1/5: 5 < 25*f32(.2), hence bin 0.
  expect(Array.from(result.predictedModes)).toEqual([0]);
});
it('B4 accumulates encoded step lengths in binary32 order on a multistep leg', () => {
  const positions = Float32Array.from({length: 32}, (_, index) =>
    index % 2 === 0 && Math.floor(index / 2) % 2 === 1 ? 0.277777761220932 : 0
  );
  const result = computeModalSplitOnCPU(
    {
      ...input,
      positions,
      timestamps: Float32Array.from({length: 16}, (_, index) => index / 15),
      trackOffsets: Uint32Array.of(0, 16),
      triplegs: {...input.triplegs, endRows: Uint32Array.of(16)}
    },
    {modeCount: 3, metric: 'count', modeSpeedUpperBoundsFloat32: [15 / 3.6, 100 / 3.6, Infinity]}
  );
  // Fifteen stored equal axis steps accumulate to f32 4.166666030883789, below the cutoff.
  expect(Array.from(result.predictedModes)).toEqual([0]);
});
