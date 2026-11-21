// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUCanonicalGeometryEquality} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-canonical-geometry-equality';
import {requireTestDevice} from '../../utils/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {ORACLE_CASES, expectOracleRows} from './refinement-output-oracles-fixtures';
import {
  createRefinementOracleHarness,
  executeGPUOracleOperation,
  executeCPUOracleOperation
} from './refinement-output-oracle-utils';

it('oracle harness reads uploaded literals and nonzero poison before any graph submission', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRefinementOracleHarness(device);
  try {
    // A literal upload/readback oracle isolates buffer-usage mistakes from geometry kernels.
    // Every number is exactly representable in f32; no numeric tolerance is needed.
    expect(await harness.read(harness.view('float32x2', 2))).toEqual([913, 913, 913, 913]);
    expect(await harness.read(harness.view('uint32x2', 2))).toEqual([913, 913, 913, 913]);
    expect(await harness.read(harness.view('float32x2', 2, [0.5, -2, 4, 8]))).toEqual([
      0.5, -2, 4, 8
    ]);
    expect(await harness.read(harness.view('uint32', 2, [71, 4093]))).toEqual([71, 4093]);
  } finally {
    harness.destroy();
  }
});

for (const createCase of ORACLE_CASES) {
  const fixture = createCase();
  it(`independent GPU output oracle: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const operation = fixture.gpu();
    const gpuExpected = 'gpuExpected' in fixture ? fixture.gpuExpected : fixture.expected;
    const contributor = await executeGPUOracleOperation(device, operation, fixture.inputs, rows =>
      expectOracleRows(rows, gpuExpected)
    );
    expectOracleRows(contributor, gpuExpected);
    const lowered = await runOperationGraph(device, {
      inputs: fixture.inputs,
      operations: [fixture.gpu()],
      outputs: Object.values(operation.outputs).map(output => output.value)
    });
    expectOracleRows(lowered, gpuExpected);
    // Retain an explicit twin comparison as well as independent expectations for both routes.
    const cpu = await executeCPUOracleOperation(fixture.cpu(), fixture.inputs);
    expectOracleRows(cpu, fixture.expected);
    expectOracleRows(
      lowered,
      Object.fromEntries(
        Object.entries(cpu).map(([name, values]) => [
          name,
          name === 'circles' && 'gpuExpected' in fixture
            ? fixture.gpuExpected.circles
            : Array.from(values)
        ])
      )
    );
  });
}

it('canonical equality distinguishes literal equal, coordinate-different, kind-different and invalid trees', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRefinementOracleHarness(device);
  const {graph, view, read} = harness;
  // Left point (1,2) versus equal point with a different ID, point (1,3), line (1,2), NaN point.
  // Geometry equality ignores IDs, distinguishes kinds and coordinates, and diagnoses nonfinite.
  const output = view('uint32', 4);
  graph.add(
    new GPUCanonicalGeometryEquality({
      left: {
        positions: view('float32x2', 1, [1, 2]),
        nodes: view('uint32x4', 1, [0, 0, 1, 91]),
        children: view('uint32', 1, [0])
      },
      right: {
        positions: view('float32x2', 4, [1, 2, 1, 3, 1, 2, NaN, 2]),
        nodes: view('uint32x4', 4, [0, 0, 1, 17, 0, 1, 1, 18, 2, 2, 1, 19, 0, 3, 1, 20]),
        children: view('uint32', 1, [0])
      },
      pairs: view('uint32x2', 4, [0, 0, 0, 1, 0, 2, 0, 3]),
      output
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    expect(await read(output)).toEqual([1, 0, 0, 2]);
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});
