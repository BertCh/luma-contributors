// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import * as statistics from '../../../src/gpu-network/network-statistics/gpu-network-statistics';
import * as coarsening from '../../../src/gpu-network/network-coarsening/gpu-network-coarsening';
import * as filter from '../../../src/gpu-network/network-subgraph-filter/network-subgraph-filter-parameters';

it('word readers reject a short count record and decode literal uint32 values', () => {
  expect(() => filter.decodeGPUNetworkSubgraphFilterCountsWords([12])).toThrow(/counts.*4.*1/);
  expect(filter.decodeGPUNetworkSubgraphFilterCountsWords([257, 300, 3, 4])).toEqual({
    liveNodeCount: 257,
    liveSlotCount: 300,
    selfLoopSlotCount: 3,
    liveEdgeCount: 4
  });
});

it('network byte readers honor unaligned offsets, bounds, and numeric words', () => {
  // Little-endian words 257 and 300 deliberately have nonzero high bytes.
  const bytes = new Uint8Array(35);
  const view = new DataView(bytes.buffer, 1, 32);
  [257, 300, 3, 4, 5, 6, 7, 8].forEach((word, index) => view.setUint32(index * 4, word, true));
  const countsBytes = bytes.subarray(1, 17);
  expect(filter.decodeGPUNetworkSubgraphFilterCountsFromBytes(countsBytes).liveSlotCount).toBe(300);
  expect(
    filter.decodeGPUNetworkSubgraphFilterCountsFromBytes({bytes: countsBytes}).liveNodeCount
  ).toBe(257);
  expect(
    coarsening.decodeGPUNetworkCoarseningSummaryFromBytes(bytes.subarray(1, 33)).superedgeCount
  ).toBe(8);
  expect(() => filter.decodeGPUNetworkSubgraphFilterCountsFromBytes(bytes.subarray(1, 16))).toThrow(
    /4-byte/
  );
  expect(() =>
    coarsening.decodeGPUNetworkCoarseningSummaryFromBytes(bytes.subarray(1, 32))
  ).toThrow(/4-byte/);
  expect(() =>
    coarsening.decodeGPUNetworkCoarseningSummaryFromBytes(bytes.subarray(1, 29))
  ).toThrow(/8.*7/);
  expect(() => filter.decodeGPUNetworkSubgraphFilterCountsFromBytes(bytes.subarray(1, 13))).toThrow(
    /counts.*4.*3/
  );
  expect(
    coarsening.decodeGPUNetworkCoarseningSummaryWords([257, 300, 3, 4, 5, 6, 7, 8]).liveNodeCount
  ).toBe(257);
});

it('byte readers refuse untyped numeric word payloads', () => {
  for (const readBytes of [
    filter.decodeGPUNetworkSubgraphFilterCountsFromBytes,
    coarsening.decodeGPUNetworkCoarseningSummaryFromBytes,
    statistics.decodeGPUNetworkStatisticsFromBytes
  ]) {
    const words = new Uint32Array(19);
    for (const input of [Array.from(words), words, {bytes: words}, null]) {
      expect(() => readBytes(input as never)).toThrow(/expected Uint8Array or ArrayBuffer/);
    }
  }
});

it('statistics infer all three histogram widths on exact logical records', () => {
  const words = new Uint32Array(136);
  words[0] = 257;
  words.set(
    Array.from({length: 120}, (_, index) => 100 + index),
    16
  );
  const decoded = statistics.decodeGPUNetworkStatisticsWords(words);
  expect(decoded.outDegreeHistogram).toEqual(Array.from({length: 40}, (_, index) => 100 + index));
  expect(decoded.inDegreeHistogram).toEqual(Array.from({length: 40}, (_, index) => 140 + index));
  expect(decoded.totalDegreeHistogram).toEqual(Array.from({length: 40}, (_, index) => 180 + index));
  expect(
    statistics.decodeGPUNetworkStatisticsFromBytes({bytes: new Uint8Array(words.buffer)})
      .liveNodeCount
  ).toBe(257);
  expect(statistics.decodeGPUNetworkStatisticsFromBytes(words.buffer).liveNodeCount).toBe(257);
  const offsetBytes = new Uint8Array(words.byteLength + 3);
  offsetBytes.set(new Uint8Array(words.buffer), 1);
  expect(
    statistics.decodeGPUNetworkStatisticsFromBytes(offsetBytes.subarray(1, 1 + words.byteLength))
      .inDegreeHistogram[0]
  ).toBe(140);
  expect(() =>
    statistics.decodeGPUNetworkStatisticsFromBytes(offsetBytes.subarray(1, words.byteLength))
  ).toThrow(/4-byte/);
  expect(() => statistics.decodeGPUNetworkStatisticsWords(words, {degreeBinCount: 32})).toThrow(
    /length/
  );
  for (const degreeBinCount of [NaN, -6, 0, 1.5, 4097]) {
    expect(() => statistics.getGPUNetworkStatisticsLength(degreeBinCount)).toThrow(
      /degreeBinCount/
    );
    expect(() => statistics.decodeGPUNetworkStatisticsWords(words, {degreeBinCount})).toThrow(
      /degreeBinCount/
    );
    expect(() =>
      statistics.decodeGPUNetworkStatisticsWords(new Uint32Array(), {degreeBinCount})
    ).toThrow(/degreeBinCount/);
  }
  expect(() => statistics.decodeGPUNetworkStatisticsWords(new Uint32Array(18))).toThrow();
  const shortValidLayout = new Uint32Array(19);
  expect(statistics.decodeGPUNetworkStatisticsWords(shortValidLayout).componentsConverged).toBe(
    false
  );
  const padded = new Uint32Array(140);
  padded.set(words);
  expect(() => statistics.decodeGPUNetworkStatisticsWords(padded, {degreeBinCount: 40})).toThrow(
    /logical length/
  );
  expect(
    statistics.decodeGPUNetworkStatisticsWords(padded.subarray(0, 136)).inDegreeHistogram[0]
  ).toBe(140);
});

it('64-bin statistics preserve the 34-node directed star in every histogram', () => {
  // 0 -> 1..33: out-degrees 33, 0 x 33; in-degrees 0, 1 x 33; totals 33, 1 x 33.
  const words = new Uint32Array(16 + 3 * 64);
  words[0] = 34;
  words[16] = 33;
  words[16 + 33] = 1;
  words[80] = 1;
  words[81] = 33;
  words[144 + 1] = 33;
  words[144 + 33] = 1;
  const decoded = statistics.decodeGPUNetworkStatisticsWords(words);
  expect(decoded.inDegreeHistogram.slice(0, 2)).toEqual([1, 33]);
  for (const histogram of [
    decoded.outDegreeHistogram,
    decoded.inDegreeHistogram,
    decoded.totalDegreeHistogram
  ]) {
    expect(histogram).toHaveLength(64);
    expect(histogram.reduce((sum, count) => sum + count, 0)).toBe(34);
  }
});

it('every numeric-word reader rejects ambiguous one-byte views (B4)', () => {
  // All views share the literal little-endian uint32 65537 = bytes [1,0,1,0].
  const words = new Uint32Array(19);
  words[0] = 65537;
  for (const bytes of [
    new Int8Array(words.buffer),
    new Uint8ClampedArray(words.buffer),
    new Uint8Array(words.buffer)
  ]) {
    for (const reader of [
      filter.decodeGPUNetworkSubgraphFilterCountsWords,
      coarsening.decodeGPUNetworkCoarseningSummaryWords,
      statistics.decodeGPUNetworkStatisticsWords
    ]) {
      // Dynamic consumers must be refused too; the new signatures exclude these statically.
      expect(() => reader(bytes as never)).toThrow(/use the FromBytes reader/);
    }
  }
  expect(filter.decodeGPUNetworkSubgraphFilterCountsWords(words).liveNodeCount).toBe(65537);
  expect(coarsening.decodeGPUNetworkCoarseningSummaryWords(Array.from(words)).liveNodeCount).toBe(
    65537
  );
  expect(
    statistics.decodeGPUNetworkStatisticsFromBytes(new Uint8Array(words.buffer)).liveNodeCount
  ).toBe(65537);
});
