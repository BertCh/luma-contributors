// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  decodeGPUSpaceTimeInferenceMetadata,
  getGPUSpaceTimeParameterValues,
  getGPUSpaceTimeRelativeTimes,
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
      {...common, neighbors: pairs, k: 1, output}
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
        times: Float32Array.of(2, 2, 2),
        parameters: getGPUSpaceTimeParameterValues(settings)
      },
      common
    );
    expect(result.permutationCount[0]).toBe(10);
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
        decodeGPUSpaceTimeInferenceMetadata(metadata, result.permutationCount[0])
      ).toMatchObject({
        permutationCount: 10,
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
    expect(getGPUSpaceTimeInferenceMetadata('mantel', settings, 64).permutationCount).toBe(64);
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
