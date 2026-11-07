// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUConditionalGaussianSimulation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation';
import {InterpolationBenchmarkScope} from './interpolation-performance-harness';
import {requireTestDevice} from '../../utils/require-test-device';
import {readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';

for (const blocked of [false, true]) {
  it(`retained ${blocked ? 'blocked' : 'cooperative'} simulation preserves preparation and stream identity`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scope = new InterpolationBenchmarkScope(device, 'retained-simulation-replay');
    const observationCount = blocked ? 129 : 4;
    const queryCount = blocked ? 129 : 3;
    const originalPositions = blocked
      ? Float32Array.from({length: observationCount * 2}, (_, component) =>
          component % 2 === 0
            ? (Math.floor(component / 2) % 16) / 4
            : Math.floor(component / 32) / 4
        )
      : new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]);
    const originalObservations = Float32Array.from(
      {length: observationCount * 4},
      (_, component) =>
        component % 4 === 0
          ? 1 + (Math.floor(component / 4) % 7)
          : component % 4 === 1
            ? 0.1
            : component % 4 === 3
              ? 1
              : 0
    );
    const queries = blocked
      ? Float32Array.from({length: queryCount * 2}, (_, component) =>
          component % 2 === 0
            ? ((Math.floor(component / 2) % 16) + 0.3) / 4
            : (Math.floor(component / 32) + 0.2) / 4
        )
      : new Float32Array([0.2, 0.4, 0.2, 0.4, 0.8, 0.7]);
    queries[2] = queries[0];
    queries[3] = queries[1];
    const positions = originalPositions.slice();
    const observations = originalObservations.slice();
    const inputs = {
      positions: scope.input('positions', positions, 'float32x2'),
      observations: scope.input('observations', observations, 'float32x4'),
      queries: scope.input('queries', queries, 'float32x2'),
      covariance: {model: 'exponential' as const, sill: 1, range: 1.3}
    };
    const createOutput = (name: string, replicateCount: number) => ({
      values: scope.output(`${name}-values`, replicateCount * queryCount, 'float32'),
      moments: scope.output(`${name}-moments`, queryCount, 'float32x2'),
      status: scope.output(`${name}-status`, replicateCount, 'uint32')
    });
    const retained = new GPUConditionalGaussianSimulation({
      ...inputs,
      id: 'retained',
      mean: 0,
      seed: 71,
      replicateCount: 16,
      reusePreparation: true,
      output: createOutput('retained', 16)
    });
    scope.graph.add(retained);
    for (const [name, seed, replicateOffset, replicateCount, mean] of [
      ['fresh', 71, 0, 16, 0],
      ['batch', 99, 17, 8, 0],
      ['mean', 99, 17, 8, 2.5]
    ] as const) {
      scope.graph.add(
        new GPUConditionalGaussianSimulation({
          ...inputs,
          id: name,
          seed,
          replicateOffset,
          replicateCount,
          mean,
          output: createOutput(name, replicateCount)
        })
      );
    }
    const compiled = scope.graph.compile();
    const statisticsBuffer = retained.preparationStatistics?.buffer.defaultBuffer;
    if (!(statisticsBuffer instanceof Buffer))
      throw new Error('Missing retained statistics buffer');
    const readStatistics = async () => (await readUint32(statisticsBuffer, 7)).slice(4, 6);
    const discardEncoding = () => {
      const encoder = device.createCommandEncoder();
      compiled.encode(encoder, {parameters: undefined});
      encoder.finish();
    };
    const compareOutputs = async (reference = 'fresh', replicateCount = 16) => {
      submitGraph(device, compiled, undefined);
      const actual = await scope.read();
      expect(actual['retained-status'].slice(0, replicateCount)).toEqual(
        actual[`${reference}-status`]
      );
      for (const name of ['moments', 'values']) {
        const expected = actual[`${reference}-${name}`];
        for (const [index, value] of expected.entries()) {
          const observed = actual[`retained-${name}`][index];
          if (Number.isNaN(value)) expect(Number.isNaN(observed)).toBe(true);
          else
            expect(Math.abs(observed - value)).toBeLessThanOrEqual(
              1e-5 * Math.max(1, Math.abs(value))
            );
        }
      }
      if (replicateCount < 16) {
        expect(actual['retained-status'].slice(replicateCount)).toEqual(
          Array(16 - replicateCount).fill(1)
        );
        expect(
          actual['retained-values'].slice(replicateCount * queryCount).every(Number.isNaN)
        ).toBe(true);
      }
      return actual;
    };
    const uploadInputs = () => {
      scope.initialWrites[0].buffer.write(positions);
      scope.initialWrites[1].buffer.write(observations);
    };
    try {
      for (const write of scope.initialWrites) write.buffer.write(write.values);
      discardEncoding();
      expect(await readStatistics()).toEqual([0, 0]);
      const initial = await compareOutputs();
      expect(await readStatistics()).toEqual([1, 1]);
      await compareOutputs();
      expect(await readStatistics()).toEqual([1, 1]);

      for (let row = 0; row < observationCount; row++) observations[row * 4] += 3;
      uploadInputs();
      retained.invalidateObservations();
      await compareOutputs();
      expect(await readStatistics()).toEqual([1, 2]);

      observations[1] += 0.2;
      uploadInputs();
      retained.invalidateObservations();
      await compareOutputs();
      expect(await readStatistics()).toEqual([2, 3]);

      positions[0] += 0.3;
      uploadInputs();
      retained.invalidatePreparation();
      await compareOutputs();
      expect(await readStatistics()).toEqual([3, 4]);

      observations[0] = NaN;
      observations[3] = 0;
      uploadInputs();
      retained.invalidateObservations();
      await compareOutputs();
      expect(await readStatistics()).toEqual([4, 5]);

      observations[4] = NaN;
      uploadInputs();
      retained.invalidateObservations();
      const invalid = await compareOutputs();
      expect(invalid['retained-status']).toEqual(Array(16).fill(1));
      expect(await readStatistics()).toEqual([4, 5]);
      await compareOutputs();
      expect(await readStatistics()).toEqual([4, 5]);

      observations[4] = 5;
      uploadInputs();
      retained.invalidateObservations();
      await compareOutputs();
      expect(await readStatistics()).toEqual([5, 6]);

      positions.set(originalPositions);
      observations.set(originalObservations);
      positions[2] = positions[0];
      positions[3] = positions[1];
      observations[1] = observations[5] = 0;
      uploadInputs();
      retained.invalidatePreparation();
      const singular = await compareOutputs();
      expect(singular['retained-status']).toEqual(Array(16).fill(2));
      expect(await readStatistics()).toEqual([6, 6]);

      positions.set(originalPositions);
      observations.set(originalObservations);
      uploadInputs();
      retained.invalidatePreparation();
      await compareOutputs();
      expect(await readStatistics()).toEqual([7, 7]);

      retained.setDrawParameters({seed: 99, replicateOffset: 17, replicateCount: 8});
      await compareOutputs('batch', 8);
      expect(await readStatistics()).toEqual([7, 7]);
      retained.setMean(2.5);
      await compareOutputs('mean', 8);
      expect(await readStatistics()).toEqual([7, 8]);

      retained.setMean(0);
      retained.setDrawParameters({seed: 71, replicateOffset: 0, replicateCount: 16});
      const restored = await compareOutputs();
      expect(restored['retained-values']).toEqual(initial['retained-values']);
      expect(await readStatistics()).toEqual([7, 9]);
      retained.invalidatePreparation();
      discardEncoding();
      expect(await readStatistics()).toEqual([7, 9]);
      await compareOutputs();
      expect(await readStatistics()).toEqual([8, 10]);

      expect(() => retained.setMean(NaN)).toThrow();
      expect(() => retained.setDrawParameters({replicateCount: 17})).toThrow();
      expect(() =>
        retained.setDrawParameters({replicateOffset: 0xffffffff, replicateCount: 2})
      ).toThrow();
      await compareOutputs();
      expect(await readStatistics()).toEqual([8, 10]);
    } finally {
      compiled.destroy();
      retained.destroy();
      expect(statisticsBuffer.destroyed).toBe(true);
      for (const write of scope.initialWrites) expect(write.buffer.destroyed).toBe(false);
      scope.destroy();
    }
    expect(() => retained.invalidatePreparation()).toThrow();
  });
}
