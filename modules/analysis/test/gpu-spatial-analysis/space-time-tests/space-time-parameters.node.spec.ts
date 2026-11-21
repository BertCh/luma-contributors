// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  decodeGPUSpaceTimeInferenceMetadata,
  getGPUSpaceTimeParameterValues,
  getGPUSpaceTimeRelativeTimes,
  getKnoxPoissonPValue,
  computeMantelTestOnCPU,
  GPUKnoxTest,
  GPUMantelTest,
  GPUJacquezTest
} from '../../../src/gpu-spatial-analysis/space-time-tests/index';
import {getGPUSpaceTimeInferenceMetadata} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-parameters';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

describe('getGPUSpaceTimeRelativeTimes', () => {
  const epoch = 1_700_000_000;

  it('keeps one-second resolution that raw float32 epoch seconds lose', () => {
    const times = [epoch, epoch + 1, epoch + 2, epoch + 3600];
    // Raw float32 epoch seconds collapse neighbors (spacing 128 s).
    const raw = Float32Array.from(times);
    expect(raw[0]).toBe(raw[1]);
    const relative = getGPUSpaceTimeRelativeTimes(times);
    expect(relative.origin).toBe(epoch);
    expect(Array.from(relative.times)).toEqual([0, 1, 2, 3600]);
    // Spacing at a 3600 s span: 2^(floor(log2 3600) - 23) = 2^-12.
    expect(relative.resolution).toBe(2 ** -12);
  });

  it('uses an explicit origin and handles a constant series', () => {
    const relative = getGPUSpaceTimeRelativeTimes([epoch + 5, epoch + 7], epoch);
    expect(Array.from(relative.times)).toEqual([5, 7]);
    expect(getGPUSpaceTimeRelativeTimes([epoch, epoch]).resolution).toBe(0);
  });

  it('rejects non-finite times and spans coarser than the required resolution', () => {
    expect(() => getGPUSpaceTimeRelativeTimes([0, Number.NaN])).toThrow();
    expect(() => getGPUSpaceTimeRelativeTimes([0, Infinity])).toThrow();
    // 4.6 months of seconds (1.2e7 s) resolve to 1 s, not 1 ms.
    expect(() => getGPUSpaceTimeRelativeTimes([0, 1.2e7], 0, 1e-3)).toThrow(/resolution/);
    expect(() => getGPUSpaceTimeRelativeTimes([0, 1.2e7], 0, 1)).not.toThrow();
  });
});

describe('getGPUSpaceTimeInferenceMetadata', () => {
  it('does not describe a stopped ten-draw result as the requested 150 draws', () => {
    const metadata = getGPUSpaceTimeInferenceMetadata(
      'knox',
      {seed: 5, permutationCount: 150},
      191,
      10
    );
    expect(metadata.permutationCount).toBeUndefined();
    expect(metadata.requestedPermutationCount).toBe(150);
    expect(metadata.stoppingRule).toEqual({method: 'besag-clifford', stopAfterExceedances: 10});
    expect(decodeGPUSpaceTimeInferenceMetadata(metadata, 10)).toMatchObject({
      permutationCount: 10,
      requestedPermutationCount: 150
    });
    expect(metadata.permutationCount).toBeUndefined();
    expect(decodeGPUSpaceTimeInferenceMetadata(metadata, 150).permutationCount).toBe(150);
    expect(decodeGPUSpaceTimeInferenceMetadata(metadata, 0)).toMatchObject({
      permutationCount: 0,
      available: false
    });
    expect(decodeGPUSpaceTimeInferenceMetadata(metadata, 10)).toMatchObject({available: true});
    for (const invalidCount of [1, 9, -1, 151, 1.5, NaN, Infinity]) {
      expect(() => decodeGPUSpaceTimeInferenceMetadata(metadata, invalidCount)).toThrow(
        /space-time.*permutationCount/
      );
    }
    const shortRequest = getGPUSpaceTimeInferenceMetadata(
      'knox',
      {seed: 5, permutationCount: 5},
      191,
      10
    );
    expect(() => decodeGPUSpaceTimeInferenceMetadata(shortRequest, 1)).toThrow(/permutationCount/);
    expect(() => decodeGPUSpaceTimeInferenceMetadata(shortRequest, 4)).toThrow(/permutationCount/);
    expect(decodeGPUSpaceTimeInferenceMetadata(shortRequest, 5).permutationCount).toBe(5);
  });

  it('states the one-sided Monte Carlo alternative from the packed settings', () => {
    expect(getGPUSpaceTimeInferenceMetadata('knox', {seed: 7, permutationCount: 99}, 999)).toEqual({
      alternative: 'greater',
      pValueConstruction: 'upper-tail-count',
      permutationCount: 99,
      requestedPermutationCount: 99,
      stoppingRule: {method: 'fixed-count'},
      seed: 7,
      multipleTesting: 'none',
      testedRowsDefinition:
        'all unordered event pairs listed in the spatial pair list, counted once'
    });
  });

  it('carries each contributor stopping rule and the CPU twin executed count into final metadata', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
      createTransientView(graph, `inference-${serial++}`, format, length);
    const pairs = {
      offsets: view('uint32', 4),
      neighbors: view('uint32', 3),
      weights: view('float32', 3),
      distances: view('float32', 3)
    };
    const common = {
      times: view('float32', 3),
      parameters: view('uint32', 4),
      maximumPermutations: 191,
      stopAfterExceedances: 10
    };
    const output = {
      statistics: view('uint32', 192),
      summary: view('float32', 12),
      permutationCount: view('uint32', 1)
    };
    const callerProps = [
      {...common, pairs, output},
      {...common, pairs, output: {...output, statistics: view('float32', 192)}},
      {...common, neighbors: pairs, neighborCount: 1, output}
    ] as const;
    const contributors = [
      new GPUKnoxTest(callerProps[0]),
      new GPUMantelTest(callerProps[1]),
      new GPUJacquezTest(callerProps[2])
    ];
    const settings = {seed: 5, permutationCount: 150};
    const result = computeMantelTestOnCPU(
      {
        offsets: Uint32Array.of(0, 2, 3, 3),
        neighbors: Uint32Array.of(1, 2, 2),
        weights: Float32Array.of(1, 1, 1),
        distances: Float32Array.of(1, 2, 3),
        times: Float32Array.of(0, 1, 3),
        parameters: getGPUSpaceTimeParameterValues(settings)
      },
      common
    );
    expect(result.permutationCount[0]).toBeGreaterThanOrEqual(10);
    expect(result.permutationCount[0]).toBeLessThanOrEqual(150);
    for (const [index, contributor] of contributors.entries()) {
      const nodes = contributor.getCommandNodes(graph);
      const sources = getGPUCommandNodeKernelSources(nodes)
        .map(kernel => kernel.source)
        .join('\n');
      expect(sources).toContain('10u');
      delete (callerProps[index] as {stopAfterExceedances?: number}).stopAfterExceedances;
      const metadata = contributor.getInferenceMetadata(settings);
      expect(metadata.permutationCount).toBeUndefined();
      expect(metadata.stoppingRule).toEqual({method: 'besag-clifford', stopAfterExceedances: 10});
      expect(
        decodeGPUSpaceTimeInferenceMetadata(
          metadata,
          result.permutationCount[0],
          result.usablePermutationCount[0]
        )
      ).toMatchObject({
        permutationCount: result.permutationCount[0],
        requestedPermutationCount: 150,
        alternative: 'greater',
        seed: 5
      });
      (callerProps[index] as {maximumPermutations: number}).maximumPermutations = 1;
      expect(contributor.getInferenceMetadata(settings)).toEqual(metadata);
    }
  });

  it('clamps the count as the kernels do and validates as the packer does', () => {
    const settings = {seed: 1, permutationCount: 500};
    expect(
      getGPUSpaceTimeInferenceMetadata('mantel', settings, 64).permutationCount
    ).toBeUndefined();
    expect(getGPUSpaceTimeInferenceMetadata('mantel', settings, 64).requestedPermutationCount).toBe(
      64
    );
    expect(getGPUSpaceTimeParameterValues(settings)[2]).toBe(500);
    for (const invalid of [
      {seed: -1, permutationCount: 9},
      {seed: 1, permutationCount: 0},
      {seed: 1, permutationCount: 9, timeThreshold: -1}
    ]) {
      expect(() => getGPUSpaceTimeParameterValues(invalid)).toThrow();
      expect(() => getGPUSpaceTimeInferenceMetadata('knox', invalid, 64)).toThrow();
    }
  });
});

describe('getKnoxPoissonPValue', () => {
  // [observed, expected, P(X >= observed)] from SciPy 1.13.1, `scipy.stats.poisson.sf(observed - 1,
  // expected)` with /usr/bin/python3. Small tails used to return exactly 0 because the helper
  // computed one minus the lower cumulative probability.
  const cases: [number, number, number][] = [
    [1, 1, 0.6321205588285577],
    [3, 3, 0.5768099188731565],
    [5, 10, 0.9707473119230389],
    [1000, 1000, 0.5042052441802155],
    [1050, 1000, 0.05962832876847707],
    [20, 3, 8.314423588191718e-11],
    [10, 0.5, 1.709670029348906e-10],
    [1200, 1000, 4.684203855872308e-10],
    [100000, 99000, 0.0007574199211747684],
    [101500, 100000, 1.1193749619997961e-6],
    [36, 5, 3.045720787991376e-19],
    [1290, 1000, 9.40696576548782e-19],
    [20, 1, 1.5875276010732616e-19],
    [30, 1, 1.4330814167223303e-33],
    [40, 1, 4.621445840379973e-49],
    [150, 10, 8.509523870867125e-118],
    [500, 100, 3.8087542289263496e-178],
    [239, 5, 4.5956159366136846e-302],
    [2384, 1000, 5.17185159407604e-301]
  ];

  it('matches the SciPy upper tail to a small relative error, including tails near 1e-300', () => {
    for (const [observed, expected, reference] of cases) {
      const value = getKnoxPoissonPValue(observed, expected);
      expect(Number.isFinite(value), `${observed}, ${expected}`).toBe(true);
      expect(value, `${observed}, ${expected}`).toBeGreaterThan(0);
      expect(Math.abs(value - reference) / reference, `${observed}, ${expected}`).toBeLessThan(
        1e-9
      );
    }
  });

  it('keeps the edge cases: zero count, zero mean, nonfinite inputs and an underflowing tail', () => {
    expect(getKnoxPoissonPValue(0, 3)).toBe(1);
    expect(getKnoxPoissonPValue(4, 0)).toBe(0);
    expect(getKnoxPoissonPValue(Number.NaN, 3)).toBeNaN();
    // SciPy also returns 0.0 for sf(199, 1.5): the tail (about 1e-340) is below the float64 range.
    expect(getKnoxPoissonPValue(200, 1.5)).toBe(0);
  });
});

it('preserves a representable Poisson upper tail when count / mean overflows', () => {
  // P(X >= 1) = 1-exp(-mean), evaluated without cancellation, including binary64 subnormals.
  const expected = -Math.expm1(-1e-310);
  const actual = getKnoxPoissonPValue(1, 1e-310);
  expect(Number.isFinite(actual)).toBe(true);
  expect(actual).toBeGreaterThan(0);
  expect(Math.abs(actual / expected - 1)).toBeLessThan(1e-12);
  // SciPy sf(23, 3.25), reviewer ordinary control, independent of the log-domain repair.
  expect(Math.abs(getKnoxPoissonPValue(24, 3.25) / 1.38405834309809e-13 - 1)).toBeLessThan(1e-12);
});
