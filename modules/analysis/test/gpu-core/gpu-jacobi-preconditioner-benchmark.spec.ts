// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {Device} from '@luma.gl/core';
import {GPUJacobiPreconditioner} from '@luma.gl/graph/gpu-core';
import {BatchConformanceFixture} from './batch-conformance-utils';
import {getBenchmarkDistribution, getBenchmarkSettings} from '../utils/benchmark-settings';
import {requireTestDevice} from '../utils/require-test-device';
import {submitGraph} from '../utils/gpu-contributor-test-utils';

const environment = (import.meta as unknown as {env?: Record<string, string | undefined>}).env;
type Workload = {
  name: string;
  rows: number;
  shortLength: number;
  longLength?: number;
  hub?: boolean;
  signed?: boolean;
  fragmented?: boolean;
};

async function measureDiagonal(
  device: Device,
  workload: Workload,
  settings: ReturnType<typeof getBenchmarkSettings>
) {
  const offsets = [0];
  const columns: number[] = [];
  const values: number[] = [];
  const expected: number[] = [];
  const referenceStarted = performance.now();
  for (let row = 0; row < workload.rows; row++) {
    const length =
      workload.longLength !== undefined && (!workload.hub || row === 0)
        ? workload.longLength
        : workload.shortLength;
    let diagonal = 0;
    for (let entry = 0; entry < length; entry++) {
      const diagonalEntry = workload.signed ? entry < 3 : entry % 3 === 0;
      const column = diagonalEntry ? row : (row + 1) % workload.rows;
      const value = workload.signed ? [1e8, -1e8, 1][entry % 3] : 0.125;
      columns.push(column);
      values.push(value);
      if (column === row) diagonal += value;
    }
    offsets.push(columns.length);
    expected.push(diagonal === 0 ? 0 : 1 / diagonal);
  }
  const referenceMilliseconds = performance.now() - referenceStarted;
  const setupStarted = performance.now();
  const fixture = new BatchConformanceFixture(device);
  try {
    const rowOffsets = fixture.column(
      'offsets',
      'uint32',
      offsets,
      workload.fragmented ? [1, 0, offsets.length - 1] : [offsets.length]
    );
    const columnIndices = fixture.column(
      'columns',
      'uint32',
      columns,
      workload.fragmented ? [0, 4097, columns.length - 4097] : [columns.length]
    );
    const matrixValues = fixture.column(
      'values',
      'float32',
      values,
      workload.fragmented ? [1, 4096, 0, values.length - 4097] : [values.length]
    );
    const inverseDiagonal = fixture.column(
      'inverse',
      'float32',
      new Array(workload.rows).fill(77),
      workload.fragmented ? [64, 0, workload.rows - 64] : [workload.rows]
    );
    fixture.graph.add(
      new GPUJacobiPreconditioner({
        rowOffsets,
        columnIndices,
        values: matrixValues,
        inverseDiagonal
      })
    );
    const setupAndUploadMilliseconds = performance.now() - setupStarted;
    const compileStarted = performance.now();
    const compiled = fixture.graph.compile();
    const compileMilliseconds = performance.now() - compileStarted;
    const execute = async () => {
      const started = performance.now();
      submitGraph(device, compiled, undefined);
      await (device.handle as GPUDevice).queue.onSubmittedWorkDone();
      return performance.now() - started;
    };
    const validate = async () => {
      const actual = await fixture.read(inverseDiagonal);
      expect(actual.length).toBe(expected.length);
      for (let row = 0; row < actual.length; row++) {
        expect(Number.isFinite(actual[row])).toBe(true);
        expect(Math.abs(actual[row] - expected[row])).toBeLessThanOrEqual(
          1e-8 + Math.abs(expected[row]) * 1e-5
        );
      }
    };
    try {
      const firstExecutionMilliseconds = await execute();
      await validate();
      for (let warmup = 0; warmup < settings.warmupCount; warmup++) await execute();
      const wallSamples: number[] = [];
      for (let sample = 0; sample < settings.sampleCount; sample++)
        wallSamples.push(await execute());
      await validate();
      return {
        workload: workload.name,
        specification: workload,
        entries: values.length,
        referenceMilliseconds,
        setupAndUploadMilliseconds,
        compileMilliseconds,
        firstExecutionMilliseconds,
        wallSamples,
        wallDistribution: getBenchmarkDistribution(wallSamples),
        declaredNodes: compiled.stats.nodeOrder.length,
        cooperativeNodes: compiled.stats.nodeOrder.filter(identifier =>
          identifier.endsWith('-cooperative')
        ).length,
        physicalTransientBytes: compiled.stats.physicalTransientResourceBytes,
        logicalTransientBytes: compiled.stats.logicalTransientBytes,
        correctnessStatus: 'passed'
      };
    } finally {
      compiled.destroy();
      expect(fixture.buffers.every(buffer => !buffer.destroyed)).toBe(true);
    }
  } finally {
    fixture.destroy();
  }
}

it.skipIf(environment?.['VITE_JACOBI_ROW_BENCHMARK'] !== 'true')(
  'Jacobi row shape and chunk performance controls',
  {timeout: 900000},
  async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const settings = getBenchmarkSettings(environment);
    console.log(
      `JACOBI_ROW_METADATA ${JSON.stringify({settings, device: device.info, userAgent: navigator.userAgent, timing: 'Warm encode/submit/queue completion for diagonal construction. Allocation/upload, graph compile, first use, reference and full-output readback are separate. No GPU timestamp instrumentation.'})}`
    );
    const cases: Workload[] = [
      {name: 'short-4096', rows: 4096, shortLength: 8},
      {name: 'short-65538', rows: 65538, shortLength: 8},
      {name: 'hub-65538', rows: 65538, shortLength: 3, longLength: 131072, hub: true},
      {name: 'empty-hub-65538', rows: 65538, shortLength: 0, longLength: 65536, hub: true},
      {name: 'all-long-64', rows: 64, shortLength: 0, longLength: 4096},
      {name: 'all-long-1024', rows: 1024, shortLength: 0, longLength: 512},
      {name: 'row-boundary-256', rows: 32, shortLength: 256},
      {name: 'row-boundary-257', rows: 32, shortLength: 257},
      {name: 'entry-boundary-4095', rows: 1, shortLength: 4095},
      {name: 'entry-boundary-4096', rows: 1, shortLength: 4096},
      {
        name: 'signed-hub-4096',
        rows: 4096,
        shortLength: 3,
        longLength: 65536,
        hub: true,
        signed: true
      },
      {
        name: 'fragmented-hub-65538',
        rows: 65538,
        shortLength: 3,
        longLength: 131072,
        hub: true,
        fragmented: true
      },
      {name: 'empty-321', rows: 321, shortLength: 0}
    ];
    let measured = 0;
    for (const workload of cases) {
      if (!workload.name.includes(settings.workloadFilter)) continue;
      measured++;
      console.log(
        `JACOBI_ROW_MEASUREMENT ${JSON.stringify(await measureDiagonal(device, workload, settings))}`
      );
    }
    expect(measured).toBeGreaterThan(0);
  }
);
