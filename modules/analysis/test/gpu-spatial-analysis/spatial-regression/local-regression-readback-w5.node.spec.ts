// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {runInNewContext} from 'node:vm';
import {getGPUGeographicallyWeightedRegressionModelResult as readSpatialModel} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-model-result';
import {getGPUGeographicallyTemporallyWeightedRegressionModelResult as readModel} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-model-result';

for (const correctedCriterion of [NaN, -43.50803298407612, -Infinity]) {
  it(`W5-50 reads logical float32 summaries in every representation, AICc ${correctedCriterion}`, () => {
    // n=8, tr(S)=2, RSS=5/11008: llf=27.75401649203806. A true exact fit
    // has RSS=0 and infinite likelihood; an unavailable correction remains null.
    const exact = correctedCriterion === -Infinity;
    const summary = Float32Array.of(
      exact ? 0 : 5 / 11008,
      2,
      correctedCriterion,
      0.9,
      8,
      NaN,
      0,
      exact ? 0 : 5 / 66048,
      NaN,
      NaN,
      exact ? Infinity : 27.75401649203806,
      8
    );
    const expected = readModel({summary});
    expect(expected.rowCount).toBe(8);
    expect(expected.status.code).toBe(0);
    expect(expected.residualSumOfSquares).toBe(exact ? 0 : summary[0]);
    expect(expected.logLikelihood).toBe(summary[10]);
    expect(expected.correctedAkaikeInformationCriterion).toBe(
      Number.isNaN(correctedCriterion) ? null : summary[2]
    );
    const bytes = new Uint8Array(summary.buffer);
    const representations: (ArrayLike<number> | ArrayBuffer)[] = [
      [...summary],
      Float64Array.from(summary),
      bytes,
      summary.buffer
    ];
    for (const offset of [4, 3]) {
      const padded = new Uint8Array(bytes.byteLength + offset + 7).fill(255);
      padded.set(bytes, offset);
      representations.push(padded.subarray(offset, offset + bytes.byteLength));
    }
    for (const representation of representations)
      expect(readModel({summary: representation})).toEqual(expected);
  });
}

it('W5-50 rejects partial float32 words and short layouts', () => {
  for (const summary of [
    new Uint8Array(47),
    new ArrayBuffer(47),
    new Uint8Array(44),
    new ArrayBuffer(44),
    new Float32Array(11),
    new Array<number>(12)
  ])
    expect(() => readModel({summary})).toThrow(/GTWR summary/);
});

it('B6 reads foreign-realm bytes and preserves numeric precision and the 12-word prefix', () => {
  const summary = Float32Array.of(9, 2, NaN, 0.5, 5, 1, 0, 3, NaN, NaN, -7, 5);
  const foreignBytes = runInNewContext('new Uint8Array(words)', {
    words: [...new Uint8Array(summary.buffer)]
  });
  expect(readModel({summary: foreignBytes})).toEqual(readModel({summary}));
  expect(readModel({summary: foreignBytes.buffer})).toEqual(readModel({summary}));
  const numeric = Float64Array.from(summary);
  numeric[0] = 9 + 2 ** -40;
  expect(readModel({summary: numeric}).residualSumOfSquares).toBe(numeric[0]);
  const prefix = [...numeric];
  prefix.length = 13; // The unspecified tail is outside this layout.
  expect(readModel({summary: prefix})).toEqual(readModel({summary: numeric}));
});

it('N1 selectedBandwidth reads two words in numeric or logical-byte representations', () => {
  const summary = Float32Array.of(9, 2, 10, 0.5, 5, 1, 0, 3, NaN, NaN, -7, 5);
  const selected = Float32Array.of(1, 4);
  const padded = new Uint8Array(13);
  padded.set(new Uint8Array(selected.buffer), 3);
  for (const selectedBandwidth of [
    selected,
    [1, 4],
    new Uint8Array(selected.buffer),
    selected.buffer,
    padded.subarray(3, 11)
  ])
    expect(readSpatialModel({summary, selectedBandwidth}).selectedBandwidth).toEqual({
      index: 1,
      value: 4
    });
  for (const selectedBandwidth of [[1], new Uint8Array(7), new ArrayBuffer(4)])
    expect(() => readSpatialModel({summary, selectedBandwidth})).toThrow(/selectedBandwidth/);
});
