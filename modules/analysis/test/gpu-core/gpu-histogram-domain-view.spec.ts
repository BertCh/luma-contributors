// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Published `GPUHistogram` (`@luma.gl/graph` 10.0.0-alpha.12, public entry point) run twice on the
 * same float32 values: with the domain as a literal pair and with the same two numbers in a
 * GPU-resident view. See `dev-docs/upstream-proposals.md` (GPUHistogram).
 *
 * The values are the bin edges themselves: `binCount` bins of width `binWidth` from `minimum`, and
 * one value on each of the `binCount + 1` edges. Every edge and the width of the domain are
 * integers or dyadic fractions, so they are exact in float32 and "on the edge" is not a matter of
 * rounding. The expected counts are `numpy.histogram(values, bins=binCount, range=(lo, hi))`
 * semantics: bins are half-open, the last one is closed, so an edge value belongs to the bin above
 * it and the maximum joins the last bin. That is one count per bin and two in the last. NumPy 2.5.4
 * on float32 input returns exactly that for every case below.
 *
 * Result on the test GPU: neither form is right for every input, and the two forms differ. The
 * published bin is `u32((value - minimum) / (maximum - minimum) * f32(BIN_COUNT))`, a float32
 * quotient scaled back up, and how the quotient is rounded depends on whether the divisor is a
 * shader constant. In the cases pinned below the literal form behaves as one multiplication by
 * the float32 constant `BIN_COUNT / range`, and the view form as a multiplication by the float32
 * reciprocal of the range followed by one by `BIN_COUNT` (both replicated in NumPy float32; this
 * is read off the counts, not off generated code, and does not fit every other case tried). An
 * IEEE division and multiplication in the written order would be wrong on a third set (edge 27 of
 * 49 bins of width 1).
 *
 * The cases the class gets wrong assert the observed counts, so the suite passes. When a luma.gl
 * or browser upgrade places these edge values correctly the assertions fail and must be flipped to
 * the empty list, and section 6 of the proposal retired. A different adapter may misplace
 * different edges.
 */

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph, GPUHistogram} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

type DomainForm = 'literal' | 'view';
const DOMAIN_FORMS: readonly DomainForm[] = ['literal', 'view'];

async function runPublishedHistogram(
  device: Device,
  values: Float32Array,
  domain: readonly [number, number],
  binCount: number,
  domainForm: DomainForm
): Promise<number[]> {
  const graph = new GPUCommandGraph(device, {id: `published-histogram-${domainForm}`});
  const inputBuffer = createInputBuffer(device, values);
  const domainBuffer = createInputBuffer(device, Float32Array.from(domain));
  const outputBuffer = createOutputBuffer(device, binCount);
  graph.add(
    new GPUHistogram({
      id: 'published-histogram',
      input: importGraphBuffer(graph, 'histogram-input', inputBuffer, 'float32', values.length),
      output: importGraphBuffer(graph, 'histogram-output', outputBuffer, 'uint32', binCount),
      domain:
        domainForm === 'literal'
          ? domain
          : importGraphBuffer(graph, 'histogram-domain', domainBuffer, 'float32', 2)
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return Array.from(await readUint32(outputBuffer, binCount));
  } finally {
    compiled.destroy();
    inputBuffer.destroy();
    domainBuffer.destroy();
    outputBuffer.destroy();
  }
}

type EdgeCase = {binCount: number; binWidth: number; minimum: number};

/** One value on every edge of the case, the minimum and the maximum included. */
function getEdgeValues({binCount, binWidth, minimum}: EdgeCase): Float32Array {
  const values = Float32Array.from({length: binCount + 1}, (_, edge) => minimum + edge * binWidth);
  // Every edge is exact in float32, or the case does not test what it says.
  values.forEach((value, edge) => expect(value).toBe(minimum + edge * binWidth));
  return values;
}

/** `numpy.histogram` of the edge values: one per bin, and the maximum joins the last bin. */
function getExpectedCounts(binCount: number): number[] {
  const counts = new Array<number>(binCount).fill(1);
  counts[binCount - 1] = 2;
  return counts;
}

/**
 * The interior edges whose value was counted one bin too low, read off the counts. Throws when the
 * counts are not explained by values in their own bin or the one below.
 */
function getEdgesCountedOneBinLow(counts: number[]): number[] {
  const binCount = counts.length;
  const lowEdges: number[] = [];
  // `below` is 1 when the value on the lower edge of the current bin went to the bin below.
  let below = 0;
  for (let bin = 0; bin < binCount; bin++) {
    const isLast = bin === binCount - 1;
    const fromAbove = counts[bin] - (1 - below) - (isLast ? 1 : 0);
    if (fromAbove !== 0 && (fromAbove !== 1 || isLast)) {
      throw new Error(`bin ${bin} holds ${counts[bin]} values`);
    }
    if (fromAbove === 1) lowEdges.push(bin + 1);
    below = fromAbove;
  }
  return lowEdges;
}

const getDomain = ({binCount, binWidth, minimum}: EdgeCase) =>
  [minimum, minimum + binCount * binWidth] as const;

// Cases both forms count as NumPy does. 1000 bins take the global-atomics path (above 256 bins).
const RIGHT_IN_BOTH_FORMS: EdgeCase[] = [
  {binCount: 7, binWidth: 1, minimum: 0},
  {binCount: 12, binWidth: 0.25, minimum: -16},
  {binCount: 30, binWidth: 8, minimum: 1024},
  {binCount: 255, binWidth: 1, minimum: 0},
  {binCount: 1000, binWidth: 1, minimum: 0}
];

for (const edgeCase of RIGHT_IN_BOTH_FORMS) {
  const {binCount, binWidth, minimum} = edgeCase;
  it(`published GPUHistogram puts edge values in the upper bin: ${binCount} bins of ${binWidth} from ${minimum}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const values = getEdgeValues(edgeCase);
    for (const domainForm of DOMAIN_FORMS) {
      const counts = await runPublishedHistogram(
        device,
        values,
        getDomain(edgeCase),
        binCount,
        domainForm
      );
      expect(counts, domainForm).toEqual(getExpectedCounts(binCount));
    }
  });
}

// Proposal section 6. Edges whose value the published class counts one bin too low on the test
// GPU; NumPy counts none low. `[]` means that form is right for the case.
const COUNTED_ONE_BIN_LOW: (EdgeCase & Record<DomainForm, number[]>)[] = [
  // Bins of width 1: the literal form is right, the view form is wrong for 7 of 50 values.
  {binCount: 49, binWidth: 1, minimum: 0, literal: [], view: [7, 14, 27, 28, 35, 41, 47]},
  // The same seven after a shift and a scaling by a power of two.
  {binCount: 49, binWidth: 8, minimum: 1024, literal: [], view: [7, 14, 27, 28, 35, 41, 47]},
  // Bins of width 41 and 83: the literal form is wrong too, and not always on the same values.
  {binCount: 3, binWidth: 41, minimum: 0, literal: [1, 2], view: [1, 2]},
  {binCount: 6, binWidth: 41, minimum: 0, literal: [1, 2, 4], view: [1, 2, 3, 4]},
  {binCount: 12, binWidth: 41, minimum: 0, literal: [1, 2, 4, 7, 8], view: [1, 2, 3, 4, 6, 8]},
  {binCount: 7, binWidth: 83, minimum: 0, literal: [1, 2, 4], view: [1, 2, 4]}
];

for (const edgeCase of COUNTED_ONE_BIN_LOW) {
  const {binCount, binWidth, minimum} = edgeCase;
  it(`published GPUHistogram counts some edge values one bin low: ${binCount} bins of ${binWidth} from ${minimum}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const values = getEdgeValues(edgeCase);
    const observed: Record<DomainForm, number[]> = {literal: [], view: []};
    for (const domainForm of DOMAIN_FORMS) {
      const counts = await runPublishedHistogram(
        device,
        values,
        getDomain(edgeCase),
        binCount,
        domainForm
      );
      expect(counts.reduce((total, count) => total + count, 0)).toBe(binCount + 1);
      observed[domainForm] = getEdgesCountedOneBinLow(counts);
    }
    // Observed, not correct: the correct lists are empty (see the file comment).
    expect(observed).toEqual({literal: edgeCase.literal, view: edgeCase.view});
  });
}

it('published GPUHistogram with a domain view near the float32 range counts every value in bin 0', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // 16 bins over [-1e38, 1e38]: (v + 1e38) / 2e38 * 16 puts 0 and 1 in bin 8, -5e37 in bin 4 and
  // 5e37 in bin 12, which is what numpy.histogram returns.
  const values = Float32Array.of(0, 1, -5e37, 5e37);
  const domain = [-1e38, 1e38] as const;
  const expected = new Array<number>(16).fill(0);
  expected[4] = 1;
  expected[8] = 2;
  expected[12] = 1;
  expect(await runPublishedHistogram(device, values, domain, 16, 'literal')).toEqual(expected);
  // Proposal section 6. Observed, not correct: with the domain in a view all four land in bin 0.
  const observed = new Array<number>(16).fill(0);
  observed[0] = 4;
  expect(await runPublishedHistogram(device, values, domain, 16, 'view')).toEqual(observed);
});
