// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {Buffer, type Device} from '@luma.gl/core';
import {expect} from 'vitest';
import {GPUVariogramModelFit} from '../../../src/dataframe-analysis/pair-statistics/gpu-variogram-model-fit';
import {fitVariogramModel} from '../../../src/dataframe-analysis/pair-statistics/variogram-model';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {InterpolationBenchmarkScope} from './interpolation-performance-harness';
import {getBenchmarkDistribution, getBenchmarkSettings} from '../../utils/benchmark-settings';

/** Measure the actual readback/CPU-fit/reupload alternative, not a CPU time inferred from GPU phases. */
export async function measureResidentFitComparison(
  device: Device,
  binCount: number,
  settings = getBenchmarkSettings()
) {
  const scope = new InterpolationBenchmarkScope(device, `resident-fit-comparison-${binCount}`);
  const lags = Float32Array.from(
    {length: binCount},
    (_, index) => 0.1 + (index * 3.9) / (binCount - 1)
  );
  const semivariances = Float32Array.from(
    lags,
    distance => 0.2 + 1.5 * (1 - Math.exp((-3 * distance) / 2.2))
  );
  const counts = Uint32Array.from({length: binCount}, () => 20);
  const distances = scope.input('lags', lags, 'float32');
  const gamma = scope.input('semivariances', semivariances, 'float32');
  const pairs = scope.input('pairs', counts, 'uint32');
  scope.graph.add(
    new GPUVariogramModelFit({
      distances,
      semivariances: gamma,
      pairCounts: pairs,
      weighting: 'pairs',
      rangeSteps: 200,
      maximumIterations: 60,
      output: {
        parameters: scope.output('parameters', 3, 'float32x4'),
        diagnostics: scope.output('diagnostics', 3, 'float32x4'),
        status: scope.output('status', 3, 'uint32')
      }
    })
  );
  const queue = (device.handle as GPUDevice).queue;
  const cpuUpload = device.createBuffer({byteLength: 48, usage: Buffer.STORAGE | Buffer.COPY_DST});
  let compiled: ReturnType<typeof scope.graph.compile> | undefined;
  try {
    for (const write of scope.initialWrites) {
      write.buffer.write(write.values);
    }
    await queue.onSubmittedWorkDone();
    compiled = scope.graph.compile();
    submitGraph(device, compiled, undefined);
    await queue.onSubmittedWorkDone();
    const samples = [];
    for (let sample = 0; sample < settings.warmupCount + settings.sampleCount; sample++) {
      const scale = 1 + sample * 0.03;
      scope.initialWrites[1].buffer.write(Float32Array.from(semivariances, value => value * scale));
      await queue.onSubmittedWorkDone();
      const gpuStart = performance.now();
      submitGraph(device, compiled, undefined);
      await queue.onSubmittedWorkDone();
      const residentMilliseconds = performance.now() - gpuStart;
      const readStart = performance.now();
      const [readLags, readGamma, readCounts] = await Promise.all([
        readFloat32(scope.initialWrites[0].buffer, binCount),
        readFloat32(scope.initialWrites[1].buffer, binCount),
        readUint32(scope.initialWrites[2].buffer, binCount)
      ]);
      const readbackMilliseconds = performance.now() - readStart;
      const cpuStart = performance.now();
      const fits = (['spherical', 'exponential', 'gaussian'] as const).map(model =>
        fitVariogramModel(
          {distances: readLags, semivariances: readGamma, pairCounts: readCounts},
          {model, weighting: 'pairs'}
        )
      );
      const cpuFitMilliseconds = performance.now() - cpuStart;
      const uploadStart = performance.now();
      cpuUpload.write(
        Float32Array.from(fits.flatMap(fit => [fit.nugget, fit.sill, fit.range, fit.residual]))
      );
      await queue.onSubmittedWorkDone();
      const uploadMilliseconds = performance.now() - uploadStart;
      const cpuRoundTripMilliseconds = performance.now() - readStart;
      const result = await scope.read();
      expect(result['status'].every(value => value === 0 || value === 3)).toBe(true);
      expect(result['parameters'][4]).toBeCloseTo(fits[1].nugget, 3);
      expect(result['parameters'][5]).toBeCloseTo(fits[1].sill, 3);
      expect(result['parameters'][6]).toBeCloseTo(fits[1].range, 3);
      if (sample >= settings.warmupCount) {
        samples.push({
          residentMilliseconds,
          readbackMilliseconds,
          cpuFitMilliseconds,
          uploadMilliseconds,
          cpuRoundTripMilliseconds
        });
      }
    }
    return {
      workload: `variogram-repeated-fit-comparison-${binCount}`,
      binCount,
      models: 3,
      rangeSteps: 200,
      maximumIterations: 60,
      weights: 'pairs',
      samples,
      medians: {
        residentMilliseconds: getBenchmarkDistribution(
          samples.map(sample => sample.residentMilliseconds)
        ).median,
        cpuRoundTripMilliseconds: getBenchmarkDistribution(
          samples.map(sample => sample.cpuRoundTripMilliseconds)
        ).median
      },
      warmupCount: settings.warmupCount,
      correctnessStatus: 'passed',
      method:
        'Updated resident empirical bins; GPU encode/submit/completion versus actual concurrent three-buffer readback + CPU three-model fit + fitted parameters upload/completion. CPU fixed60 refinements; GPU convergence may stop earlier. Validation output readback excluded from both. Both use same device and warmed caches.'
    };
  } finally {
    compiled?.destroy();
    cpuUpload.destroy();
    scope.destroy();
  }
}
