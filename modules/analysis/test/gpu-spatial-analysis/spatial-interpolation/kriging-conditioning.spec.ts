// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPULocalKriging} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {getEquilibratedSolveWGSL} from '../../../src/gpu-spatial-analysis/spatial-interpolation/equilibrated-solve-wgsl';
import {getWGSLFloatLiteral} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {KRIGING_PIVOT_RATIO_TOLERANCE} from '../../../src/gpu-spatial-analysis/spatial-interpolation/kriging-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {KRIGING_ROBUSTNESS_REFERENCE} from './kriging-robustness-reference';

const {positions, values} = KRIGING_ROBUSTNESS_REFERENCE.illConditioned;

/** Solves one ordinary-kriging system on the clustered samples; returns status and pivot ratio. */
async function solveLocalKriging(nugget: number): Promise<{status: number; pivotRatio: number}> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const positionBuffer = createInputBuffer(device, new Float32Array(positions));
  const observationBuffer = createInputBuffer(
    device,
    new Float32Array(values.flatMap(value => [value, 0, 0, 1]))
  );
  const queryBuffer = createInputBuffer(device, new Float32Array([75, 75]));
  const valueBuffer = createOutputBuffer(device, 4);
  const statusBuffer = createOutputBuffer(device, 1);
  graph.add(
    new GPULocalKriging({
      positions: importGraphBuffer(graph, 'positions', positionBuffer, 'float32x2', values.length),
      observations: importGraphBuffer(
        graph,
        'observations',
        observationBuffer,
        'float32x4',
        values.length
      ),
      supports: importGraphBuffer(graph, 'queries', queryBuffer, 'float32x2', 1),
      mode: 'ordinary',
      covariance: {model: 'gaussian', range: 500, sill: 30, nugget},
      neighborCount: 12,
      output: {
        values: importGraphBuffer(graph, 'values', valueBuffer, 'float32x4', 1),
        status: importGraphBuffer(graph, 'status', statusBuffer, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    status: (await readUint32(statusBuffer, 1))[0],
    pivotRatio: (await readFloat32(valueBuffer, 4))[2]
  };
  compiled.destroy();
  return result;
}

it('GPULocalKriging rejects systems GPUKriging rejects, with a pivot ratio between the old and shared limits', async () => {
  // Gaussian model, nugget 3e-4, 12 clustered samples: the equilibrated pivot ratio is 4.2e-5,
  // above the old local limit of 1e-5 (status 0) and below the shared limit of 1e-4. GPUKriging
  // measured 5% weight errors at 3e-5, so the system is unreliable.
  const result = await solveLocalKriging(3e-4);
  expect(result.pivotRatio).toBeGreaterThan(1e-5);
  expect(result.pivotRatio).toBeLessThan(KRIGING_PIVOT_RATIO_TOLERANCE);
  expect(result.status).toBe(2);
  // A well conditioned nugget stays accepted.
  const accepted = await solveLocalKriging(0.1);
  expect(accepted.pivotRatio).toBeGreaterThan(KRIGING_PIVOT_RATIO_TOLERANCE);
  expect(accepted.status).toBe(0);
});

it('the shared local solver of GPULocalKriging and GPULocalRBF uses the GPUKriging pivot ratio limit', () => {
  const source = getEquilibratedSolveWGSL(8);
  expect(source).toContain(
    `const SYSTEM_PIVOT_RATIO_TOLERANCE: f32 = ${getWGSLFloatLiteral(KRIGING_PIVOT_RATIO_TOLERANCE)};`
  );
  expect(KRIGING_PIVOT_RATIO_TOLERANCE).toBe(1e-4);
});
