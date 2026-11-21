// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPULocalPermutationTest} from '../../../src/gpu-spatial-analysis/permutation-inference/gpu-local-permutation-test';
import {GPUGlobalPermutationTest} from '../../../src/gpu-spatial-analysis/permutation-inference/gpu-global-permutation-test';
import {GPUQuantileLISA} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-quantile-lisa';
import {getGPUPermutationParameterValues} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

function createFixture(device: Device, rows = 4) {
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createInputBuffer>[] = [];
  const input = (name: string, values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(
        graph,
        name,
        buffer,
        values instanceof Float32Array ? 'float32' : 'uint32',
        values.length
      )
    };
  };
  const uint = (name: string, length = 1) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
  };
  const float = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'float32', length)};
  };
  // Separate typed input constructors preserve precise view formats at call sites.
  const uintInput = (name: string, values: Uint32Array) => {
    const resource = input(name, values);
    return {
      ...resource,
      view: graph.createDataView(resource.view.buffer, {format: 'uint32', length: values.length})
    };
  };
  const floatInput = (name: string, values: Float32Array) => {
    const resource = input(name, values);
    return {
      ...resource,
      view: graph.createDataView(resource.view.buffer, {format: 'float32', length: values.length})
    };
  };
  const pathOffsets = new Uint32Array(rows + 1);
  const pathNeighbors = new Uint32Array(3 * rows);
  let slot = 0;
  for (let row = 0; row < rows; row++) {
    pathOffsets[row] = slot;
    if (row > 0) pathNeighbors[slot++] = row - 1;
    if (row + 1 < rows) pathNeighbors[slot++] = row + 1;
  }
  pathOffsets[rows] = slot;
  const offsets = uintInput('offsets', pathOffsets);
  const neighbors = uintInput('neighbors', pathNeighbors);
  const weights = floatInput('weights', new Float32Array(3 * rows).fill(1));
  const values = floatInput(
    'values',
    Float32Array.from({length: rows}, (_, row) => 2 ** row)
  );
  const finalLoss = uintInput('final-loss', Uint32Array.of(0));
  const candidateLoss = uintInput('candidate-loss', Uint32Array.of(0));
  const parameters = uintInput(
    'parameters',
    getGPUPermutationParameterValues({seed: 7, permutationCount: 9})
  );
  return {
    graph,
    uint,
    float,
    uintInput,
    floatInput,
    offsets,
    neighbors,
    values,
    finalLoss,
    candidateLoss,
    parameters,
    weights: {
      offsets: offsets.view,
      neighbors: neighbors.view,
      weights: weights.view,
      status: {overflow: finalLoss.view, candidateOverflow: candidateLoss.view}
    },
    destroy: () => {
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

it('SSR-113 publishes independent local and global causes and clears them on resubmission', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scope of ['local', 'global'] as const) {
    const fixture = createFixture(device);
    const {graph, uint, float} = fixture;
    const causes = {
      weightsOverflow: uint('cause-final'),
      candidateOverflow: uint('cause-candidate'),
      invalidInput: uint('cause-settings')
    };
    const neighbor = scope === 'local' ? uint('cause-neighbors') : undefined;
    const status = {
      weightsOverflow: causes.weightsOverflow.view,
      candidateOverflow: causes.candidateOverflow.view,
      invalidInput: causes.invalidInput.view
    };
    const props = {
      weights: fixture.weights,
      values: fixture.values.view,
      parameters: fixture.parameters.view,
      maximumPermutations: 9
    };
    graph.add(
      scope === 'local'
        ? new GPULocalPermutationTest({
            ...props,
            statistic: 'localMoran',
            maximumNeighbors: 2,
            output: {
              exceedances: uint('exceedances', 4).view,
              pseudoPValues: float('p-values', 4).view,
              status: {
                ...status,
                overflow: uint('aggregate').view,
                neighborOverflow: neighbor!.view
              }
            }
          })
        : new GPUGlobalPermutationTest({
            ...props,
            statistic: 'moran',
            output: {summary: float('summary', 12).view, status}
          })
    );
    const compiled = graph.compile();
    const check = async (expected: number[], expectedNeighbor = 0) => {
      submitGraph(device, compiled, undefined);
      const actual = [];
      for (const resource of Object.values(causes))
        actual.push((await readUint32(resource.buffer, 1))[0]);
      expect(actual).toEqual(expected);
      if (neighbor) expect((await readUint32(neighbor.buffer, 1))[0]).toBe(expectedNeighbor);
    };
    try {
      await check([0, 0, 0]);
      fixture.finalLoss.buffer.write(Uint32Array.of(1));
      await check([1, 0, 0]);
      fixture.finalLoss.buffer.write(Uint32Array.of(0));
      fixture.candidateLoss.buffer.write(Uint32Array.of(1));
      await check([0, 1, 0]);
      fixture.candidateLoss.buffer.write(Uint32Array.of(0));
      const invalid = getGPUPermutationParameterValues({seed: 7, permutationCount: 9});
      invalid[2] = 0;
      fixture.parameters.buffer.write(invalid);
      await check([0, 0, 1]);
      fixture.parameters.buffer.write(
        getGPUPermutationParameterValues({seed: 7, permutationCount: 9})
      );
      if (neighbor) {
        fixture.offsets.buffer.write(Uint32Array.of(0, 3, 6, 9, 12));
        fixture.neighbors.buffer.write(Uint32Array.of(1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2));
        await check([0, 0, 0], 1);
        fixture.offsets.buffer.write(Uint32Array.of(0, 1, 3, 5, 6));
        fixture.neighbors.buffer.write(Uint32Array.of(1, 0, 2, 1, 3, 2, 0, 0, 0, 0, 0, 0));
      }
      await check([0, 0, 0]);
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
});

it('W5-10 legacy Quantile LISA rejects 1/25 at packed .04 and includes exact 1/32 equality', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device);
  const {graph, uint, float} = fixture;
  fixture.values.buffer.write(new Float32Array(4).fill(1));
  const significant = uint('significant', 4),
    classes = uint('classes', 4),
    probabilities = float('p', 4);
  graph.add(
    new GPUQuantileLISA({
      alternative: 'geoda-legacy',
      weights: fixture.weights,
      values: fixture.values.view,
      parameters: fixture.parameters.view,
      quantileParameters: fixture.uintInput('quantiles', Uint32Array.of(2, 2)).view,
      maximumPermutations: 31,
      output: {
        membership: uint('membership', 4).view,
        boundaries: float('boundaries', 2).view,
        validCount: uint('count').view,
        invalidQuantile: uint('invalid', 1).view,
        classes: classes.view,
        significant: significant.view,
        exceedances: uint('exceedances', 4).view,
        pseudoPValues: probabilities.view,
        status: {overflow: uint('overflow').view}
      }
    })
  );
  const compiled = graph.compile();
  try {
    // Every draw ties, so the strict-complement legacy count is zero independently of seed.
    for (const {draws, level, expected} of [
      {draws: 24, level: 0.04, expected: 0},
      {draws: 31, level: 0.03125, expected: 1}
    ]) {
      fixture.parameters.buffer.write(
        getGPUPermutationParameterValues({
          seed: 7,
          permutationCount: draws,
          significanceLevel: level
        })
      );
      submitGraph(device, compiled, undefined);
      expect(await readUint32(significant.buffer, 4)).toEqual(new Array(4).fill(expected));
      expect(await readUint32(classes.buffer, 4)).toEqual(new Array(4).fill(expected));
      const actual = await readFloat32(probabilities.buffer, 4);
      expect(actual.every(Number.isFinite)).toBe(true);
      expect(actual).toEqual(new Array(4).fill(Math.fround(1 / (draws + 1))));
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('W5-11 root and child complete certificates suppress inference independently of overflow and recover', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scope of ['local', 'global'] as const)
    for (const placement of ['whole', 'root', 'child'] as const) {
      const fixture = createFixture(device);
      const {graph, uint, float} = fixture;
      const complete = fixture.uintInput('complete', Uint32Array.of(1));
      const status = {...fixture.weights.status, complete: complete.view};
      const weights =
        placement === 'whole'
          ? {...fixture.weights, status}
          : {
              rowCount: 4,
              status: placement === 'root' ? status : undefined,
              chunks: [
                {
                  rowStart: 0,
                  rowCount: 4,
                  weights: {
                    ...fixture.weights,
                    status: placement === 'child' ? status : fixture.weights.status
                  }
                }
              ]
            };
      const analysisCount = uint('analysis-count'),
        finalLoss = uint('final-cause'),
        candidateLoss = uint('candidate-cause');
      const probabilities = float('p', 4),
        summary = float('summary', 12),
        overflow = uint('aggregate');
      const props = {
        weights,
        values: fixture.values.view,
        parameters: fixture.parameters.view,
        maximumPermutations: 9
      };
      const causes = {weightsOverflow: finalLoss.view, candidateOverflow: candidateLoss.view};
      graph.add(
        scope === 'local'
          ? new GPULocalPermutationTest({
              ...props,
              statistic: 'localMoran',
              output: {
                analysisCount: analysisCount.view,
                exceedances: uint('exceedances', 4).view,
                pseudoPValues: probabilities.view,
                status: {...causes, overflow: overflow.view}
              }
            })
          : new GPUGlobalPermutationTest({
              ...props,
              statistic: 'moran',
              output: {analysisCount: analysisCount.view, summary: summary.view, status: causes}
            })
      );
      const compiled = graph.compile();
      try {
        for (const certificate of [1, 0, 1]) {
          complete.buffer.write(Uint32Array.of(certificate));
          submitGraph(device, compiled, undefined);
          expect(await readUint32(analysisCount.buffer, 1)).toEqual([certificate ? 4 : 0]);
          expect(await readUint32(finalLoss.buffer, 1)).toEqual([0]);
          expect(await readUint32(candidateLoss.buffer, 1)).toEqual([0]);
          if (scope === 'local') {
            expect(await readUint32(overflow.buffer, 1)).toEqual([certificate ? 0 : 1]);
            const actual = await readFloat32(probabilities.buffer, 4);
            expect(actual.every(certificate ? Number.isFinite : Number.isNaN)).toBe(true);
          } else {
            const actual = await readFloat32(summary.buffer, 12);
            for (const index of [0, 1]) {
              if (certificate) expect(Number.isFinite(actual[index])).toBe(true);
              else expect(actual[index]).toBeNaN();
            }
          }
        }
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    }
});

it('DEF-01 inclusive Quantile LISA retains tied draws and SSR-113 marks untested rows on reuse', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const alternative of ['greater', 'two-sided', 'geoda-legacy'] as const) {
    const fixture = createFixture(device);
    const {graph, uint, float} = fixture;
    fixture.values.buffer.write(new Float32Array(4).fill(1));
    fixture.offsets.buffer.write(Uint32Array.of(0, 2, 4, 6, 8));
    fixture.neighbors.buffer.write(Uint32Array.of(1, 3, 0, 2, 1, 3, 0, 2, 0, 0, 0, 0));
    const probabilities = float('p', 4);
    const tested = uint('tested', 4);
    graph.add(
      new GPUQuantileLISA({
        weights: fixture.weights,
        values: fixture.values.view,
        parameters: fixture.parameters.view,
        quantileParameters: fixture.uintInput('quantile', Uint32Array.of(2, 2)).view,
        maximumPermutations: 99,
        alternative,
        output: {
          membership: uint('membership', 4).view,
          boundaries: float('boundaries', 2).view,
          validCount: uint('count').view,
          classes: uint('classes', 4).view,
          invalidQuantile: uint('invalid-quantile').view,
          exceedances: uint('exceedances', 4).view,
          pseudoPValues: probabilities.view,
          tested: tested.view,
          status: {overflow: uint('overflow').view}
        }
      })
    );
    const compiled = graph.compile();
    try {
      for (const count of [9, 99]) {
        fixture.parameters.buffer.write(
          getGPUPermutationParameterValues({seed: 7, permutationCount: count})
        );
        submitGraph(device, compiled, undefined);
        const readback = await readFloat32(probabilities.buffer, 4);
        expect(readback).toHaveLength(4);
        for (const value of readback) {
          expect(Number.isFinite(value)).toBe(true);
          expect(value).toBeCloseTo(alternative === 'geoda-legacy' ? 1 / (count + 1) : 1, 6);
        }
        expect(await readUint32(tested.buffer, 4)).toEqual([1, 1, 1, 1]);
      }
      fixture.finalLoss.buffer.write(Uint32Array.of(1));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(tested.buffer, 4)).toEqual([0, 0, 0, 0]);
      fixture.finalLoss.buffer.write(Uint32Array.of(0));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(tested.buffer, 4)).toEqual([1, 1, 1, 1]);
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
});

it('SSR-114 records fixed and stopped draws and untested local rows on a reused graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // With binary ones every simulated join count and global G ties the observed statistic.
  // Inclusive greater tails reach h=2 on exactly draw 2. A one-draw request cannot reach h.
  for (const scope of ['local', 'global'] as const)
    for (const stopAfterExceedances of [undefined, 2])
      for (const permutationBatchSize of scope === 'global' ? [undefined, 3] : [undefined]) {
        const fixture = createFixture(device);
        const {graph, uint, float} = fixture;
        fixture.values.buffer.write(new Float32Array(4).fill(1));
        const mask = fixture.uintInput('mask', Uint32Array.of(1, 1, 1, 0));
        const counts = uint('counts', 4);
        const summary = float('summary', 12);
        const props = {
          weights: fixture.weights,
          values: fixture.values.view,
          mask: mask.view,
          parameters: fixture.parameters.view,
          maximumPermutations: 9,
          stopAfterExceedances,
          alternative: 'greater' as const
        };
        graph.add(
          scope === 'local'
            ? new GPULocalPermutationTest({
                ...props,
                statistic: 'localJoinCount',
                output: {
                  exceedances: uint('exceedances', 4).view,
                  pseudoPValues: float('p', 4).view,
                  permutationCounts: counts.view,
                  status: {overflow: uint('overflow').view}
                }
              })
            : new GPUGlobalPermutationTest({
                ...props,
                permutationBatchSize,
                statistic: 'getisOrdG',
                output: {summary: summary.view}
              })
        );
        const compiled = graph.compile();
        try {
          for (const requested of [9, 1, 9]) {
            fixture.parameters.buffer.write(
              getGPUPermutationParameterValues({seed: 7, permutationCount: requested})
            );
            submitGraph(device, compiled, undefined);
            const expected =
              stopAfterExceedances === undefined ? requested : Math.min(requested, 2);
            if (scope === 'local')
              expect(await readUint32(counts.buffer, 4)).toEqual([
                expected,
                expected,
                expected,
                0xffffffff
              ]);
            else expect((await readFloat32(summary.buffer, 12))[3]).toBe(expected);
          }
          mask.buffer.write(new Uint32Array(4));
          submitGraph(device, compiled, undefined);
          if (scope === 'local')
            expect(await readUint32(counts.buffer, 4)).toEqual(new Array(4).fill(0xffffffff));
          mask.buffer.write(Uint32Array.of(1, 1, 1, 0));
          submitGraph(device, compiled, undefined);
          if (scope === 'local')
            expect(await readUint32(counts.buffer, 4)).toEqual([
              stopAfterExceedances ?? 9,
              stopAfterExceedances ?? 9,
              stopAfterExceedances ?? 9,
              0xffffffff
            ]);
        } finally {
          compiled.destroy();
          fixture.destroy();
        }
      }
});

it('B3 Quantile LISA inclusive significance uses the exact resident cutoff and distinct tails', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const alternative of ['greater', 'two-sided'] as const) {
    const fixture = createFixture(device, 32);
    const {graph, uint, float} = fixture;
    fixture.offsets.buffer.write(Uint32Array.from({length: 33}, (_, index) => 2 * index));
    fixture.neighbors.buffer.write(
      Uint32Array.from({length: 96}, (_, slot) => {
        if (slot >= 64) return 0;
        const row = Math.floor(slot / 2),
          previous = (row + 31) % 32,
          next = (row + 1) % 32;
        return slot % 2 === 0 ? Math.min(previous, next) : Math.max(previous, next);
      })
    );
    fixture.values.buffer.write(Float32Array.from({length: 32}, (_, row) => (row < 3 ? 1 : 0)));
    const probabilities = float('p', 32),
      significant = uint('significant', 32),
      classes = uint('classes', 32);
    graph.add(
      new GPUQuantileLISA({
        weights: fixture.weights,
        values: fixture.values.view,
        parameters: fixture.parameters.view,
        quantileParameters: fixture.uintInput('quantiles', Uint32Array.of(16, 16)).view,
        maximumPermutations: 99,
        alternative,
        output: {
          membership: uint('membership', 32).view,
          boundaries: float('boundaries', 2).view,
          validCount: uint('count').view,
          invalidQuantile: uint('invalid-quantile').view,
          classes: classes.view,
          exceedances: uint('exceedances', 32).view,
          pseudoPValues: probabilities.view,
          significant: significant.view,
          status: {overflow: uint('overflow').view}
        }
      })
    );
    const compiled = graph.compile();
    try {
      fixture.parameters.buffer.write(
        getGPUPermutationParameterValues({
          seed: 13,
          permutationCount: 99,
          significanceLevel: alternative === 'greater' ? 0.01 : 0.02
        })
      );
      submitGraph(device, compiled, undefined);
      const probability = await readFloat32(probabilities.buffer, 32),
        classification = await readUint32(classes.buffer, 32),
        significance = await readUint32(significant.buffer, 32);
      expect(probability).toHaveLength(32);
      expect(classification).toHaveLength(32);
      expect(significance).toHaveLength(32);
      expect(Number.isFinite(probability[1])).toBe(true);
      expect(probability[1]).toBeCloseTo(alternative === 'greater' ? 0.01 : 0.02, 7);
      expect(significance[1]).toBe(0);
      expect(classification[1]).toBe(0);
      fixture.values.buffer.write(Float32Array.from({length: 32}, (_, row) => (row < 3 ? 0 : 1)));
      fixture.parameters.buffer.write(
        getGPUPermutationParameterValues({seed: 13, permutationCount: 19})
      );
      submitGraph(device, compiled, undefined);
      const differentTails = await readFloat32(probabilities.buffer, 32);
      expect(differentTails).toHaveLength(32);
      expect(Number.isFinite(differentTails[3])).toBe(true);
      expect(differentTails[3]).toBeCloseTo(alternative === 'greater' ? 1 : 0.2, 6);
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
});

it('S3 local inference rejects invalid resident cutoffs and clears the cause on clean resubmission', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device);
  const {graph, uint, float} = fixture;
  const invalidInput = uint('invalid-input'),
    overflow = uint('overflow'),
    probabilities = float('p', 4),
    significant = uint('significant', 4);
  graph.add(
    new GPULocalPermutationTest({
      weights: fixture.weights,
      values: fixture.values.view,
      parameters: fixture.parameters.view,
      maximumPermutations: 9,
      statistic: 'localMoran',
      output: {
        exceedances: uint('exceedances', 4).view,
        pseudoPValues: probabilities.view,
        significant: significant.view,
        status: {overflow: overflow.view, invalidInput: invalidInput.view}
      }
    })
  );
  const compiled = graph.compile();
  try {
    for (const cutoff of [1, 0, NaN, Infinity, -0.1]) {
      const parameters = getGPUPermutationParameterValues({seed: 7, permutationCount: 9});
      parameters[3] = new Uint32Array(Float32Array.of(cutoff).buffer)[0];
      fixture.parameters.buffer.write(parameters);
      submitGraph(device, compiled, undefined);
      expect(await readUint32(invalidInput.buffer, 1)).toEqual([1]);
      expect(await readUint32(overflow.buffer, 1)).toEqual([1]);
      const values = await readFloat32(probabilities.buffer, 4);
      expect(values).toHaveLength(4);
      expect(values.every(Number.isNaN)).toBe(true);
      expect(await readUint32(significant.buffer, 4)).toEqual([0, 0, 0, 0]);
      fixture.parameters.buffer.write(
        getGPUPermutationParameterValues({seed: 7, permutationCount: 9})
      );
      submitGraph(device, compiled, undefined);
      expect(await readUint32(invalidInput.buffer, 1)).toEqual([0]);
      expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
      const clean = await readFloat32(probabilities.buffer, 4);
      expect(clean).toHaveLength(4);
      expect(clean.every(Number.isFinite)).toBe(true);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
