// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUCoregionalizationFitOperation,
  type CoregionalizationFitOperationProps
} from '../../../src/dataframe-analysis/pair-statistics/gpu-coregionalization-fit-operation';
import {
  CPUCoregionalizationFitOperation,
  fitLinearModelOfCoregionalization
} from '../../../src/dataframe-analysis/pair-statistics/cpu-coregionalization-fit-operation';
import {GPUCoregionalizationFit} from '../../../src/dataframe-analysis/pair-statistics/gpu-coregionalization-fit';
import {getGPUCoregionalizationFitParameterValues} from '../../../src/dataframe-analysis/pair-statistics/coregionalization-parameters';
import {GPULocalKriging} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {runOperationGraph, type OperationGraphRows} from '../../utils/mixed-execution-test-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixtures from './fixtures/coregionalization-numpy.json';

function getProps(
  variableCount: number,
  structureCount: number,
  maximumIterations: number
): CoregionalizationFitOperationProps {
  return {
    variableCount,
    structureCount,
    maximumIterations,
    distances: 'distances',
    semivariances: 'semivariances',
    pairCounts: 'pairCounts',
    parameters: 'parameters',
    output: {
      matrices: 'matrices',
      loadings: 'loadings',
      diagnostics: 'diagnostics',
      status: {
        validity: 'validity',
        iterationCount: 'iterationCount',
        converged: 'converged',
        iterationLimitReached: 'iterationLimitReached',
        invalidCount: 'invalidCount'
      }
    }
  };
}

/** Factor signs/bases are arbitrary. Compare their Gram matrices (all other outputs directly). */
function reconstructLoadings(values: OperationGraphRows, variableCount: number): number[] {
  const output: number[] = [];
  for (let structure = 0; structure < values.length / variableCount ** 2; structure++) {
    for (let row = 0; row < variableCount; row++) {
      for (let column = 0; column < variableCount; column++) {
        let value = 0;
        for (let component = 0; component < variableCount; component++)
          value +=
            values[structure * variableCount ** 2 + row * variableCount + component] *
            values[structure * variableCount ** 2 + column * variableCount + component];
        output.push(value);
      }
    }
  }
  return output;
}

for (const fixture of fixtures) {
  it(`GPU/CPU operations agree with NumPy, including every output: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props = getProps(
      fixture.variableCount,
      fixture.structureCount,
      fixture.maximumIterations
    );
    // Tiny-lag NumPy/Decimal fixtures settle Gaussian and both Matérn families at h/range = 1e-8 and 1e-9.
    const coefficientTolerance = fixture.name.includes('tiny-lag') ? 2e-5 : 3e-3;
    const inputs = {
      distances: {format: 'float32' as const, values: Float32Array.from(fixture.distances)},
      semivariances: {format: 'float32' as const, values: Float32Array.from(fixture.semivariances)},
      pairCounts: {format: 'uint32' as const, values: Uint32Array.from(fixture.pairCounts)},
      parameters: {format: 'float32' as const, values: Float32Array.from(fixture.parameters)}
    };
    const outputs = ['matrices', 'loadings', 'diagnostics', ...Object.keys(props.output.status)];
    const gpuRows = await runOperationGraph(device, {
      inputs,
      operations: [new GPUCoregionalizationFitOperation(props)],
      outputs
    });
    const cpuRows = await runOperationGraph(device, {
      inputs,
      operations: [new CPUCoregionalizationFitOperation(props)],
      outputs
    });
    expect(gpuRows['validity'][0]).toBe(fixture.validity);
    for (const name of outputs) {
      if (name === 'loadings') continue;
      gpuRows[name].forEach((value, index) => {
        const expected = cpuRows[name][index];
        if (Number.isNaN(expected)) expect(Number.isNaN(value), `${name}[${index}]`).toBe(true);
        // Convergence counts can move a few sweeps at the float32 threshold. The zero-budget,
        // one-structure and short fixed-budget cases have exact counts.
        else if (name === 'iterationCount' && fixture.structureCount > 1 && fixture.converged)
          expect(Math.abs(value - expected)).toBeLessThanOrEqual(3);
        else if (name in props.output.status) expect(value, name).toBe(expected);
        // Long correlated-structure fits allow float32 accumulation and convergence error.
        else
          expect(Math.abs(value - expected), `${name}[${index}]`).toBeLessThanOrEqual(
            coefficientTolerance * Math.max(1, Math.abs(expected))
          );
      });
    }
    const reconstructedGPU = reconstructLoadings(gpuRows['loadings'], fixture.variableCount);
    const reconstructedCPU = reconstructLoadings(cpuRows['loadings'], fixture.variableCount);
    reconstructedGPU.forEach((value, index) => {
      if (!fixture.validity) expect(Number.isNaN(value)).toBe(true);
      else {
        expect(Math.abs(value - reconstructedCPU[index])).toBeLessThanOrEqual(
          coefficientTolerance * Math.max(1, Math.abs(reconstructedCPU[index]))
        );
        expect(Math.abs(value - fixture.matrices![index])).toBeLessThanOrEqual(
          coefficientTolerance * Math.max(1, Math.abs(fixture.matrices![index]))
        );
      }
    });
  });
}

it('a fitted supplied LMC predicts the hand-derived two-observation co-kriging result', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fit = fitLinearModelOfCoregionalization(
    {
      distances: new Float32Array([1]),
      semivariances: new Float32Array([1.375, -0.6875, 2.0625]),
      pairCounts: new Uint32Array([20])
    },
    {variableCount: 2, structures: [{model: 'spherical', range: 2, sill: 1}]}
  );
  expect(fit.validity[0]).toBe(1);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32x2' | 'float32x4'>(
    values: number[],
    format: Format,
    components: number
  ) => {
    const buffer = createInputBuffer(device, new Float32Array(values));
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `input-${buffers.length}`,
      buffer,
      format,
      values.length / components
    );
  };
  const prediction = createOutputBuffer(device, 4);
  const status = createOutputBuffer(device, 1);
  buffers.push(prediction, status);
  graph.add(
    new GPULocalKriging({
      positions: input([0, 0, 1, 0], 'float32x2', 2),
      observations: input([1, 0, 0, 1, 2, 0, 1, 1], 'float32x4', 4),
      supports: input([0.5, 0], 'float32x2', 2),
      mode: 'co-kriging',
      coKrigingMean: 'known',
      variableCount: 2,
      neighborCount: 2,
      covariance: fit.coregionalizationModels[0].covariance,
      coregionalizationModels: fit.coregionalizationModels,
      output: {
        values: importGraphBuffer(graph, 'prediction', prediction, 'float32x4', 1),
        status: importGraphBuffer(graph, 'status', status, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const values = await readFloat32(prediction, 4);
    expect(Array.from(await readUint32(status, 1))).toEqual([0]);
    // C = [[2,-0.3125],[-0.3125,3]], k = [2,-1]*0.6328125.
    const determinant = 6 - 0.3125 ** 2;
    const firstCovariance = 2 * 0.6328125;
    const secondCovariance = -0.6328125;
    const firstWeight = (3 * firstCovariance + 0.3125 * secondCovariance) / determinant;
    const secondWeight = (0.3125 * firstCovariance + 2 * secondCovariance) / determinant;
    expect(values[0]).toBeCloseTo(firstWeight + 2 * secondWeight, 4);
    expect(values[1]).toBeCloseTo(
      2 - firstWeight * firstCovariance - secondWeight * secondCovariance,
      4
    );
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

it('replay rewrites statuses after nonfinite bins and per-frame models, preserving trailing output capacity', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const bind = <Format extends 'float32' | 'float32x4' | 'uint32'>(
    values: Float32Array | Uint32Array,
    format: Format,
    components = 1
  ) => {
    // Outputs below are read back, so these caller-owned buffers need COPY_SRC as well.
    const buffer = createOutputBuffer(device, values.length);
    buffer.write(values);
    expect(buffer.usage & Buffer.COPY_SRC).toBe(Buffer.COPY_SRC);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(
        graph,
        `view-${buffers.length}`,
        buffer,
        format,
        values.length / components
      )
    };
  };
  const distances = bind(new Float32Array([1]), 'float32');
  const semivariances = bind(new Float32Array([2, -1, 3]), 'float32');
  const pairCounts = bind(new Uint32Array([10]), 'uint32');
  const initialParameters = getGPUCoregionalizationFitParameterValues([
    {model: 'spherical', range: 1, sill: 1}
  ]);
  const parameters = bind(initialParameters, 'float32');
  const matrices = bind(new Float32Array(8).fill(-777), 'float32');
  const loadings = bind(new Float32Array(8).fill(-777), 'float32');
  const diagnostics = bind(new Float32Array(8).fill(-777), 'float32x4', 4);
  const status = {
    validity: bind(new Uint32Array([777, 777]), 'uint32'),
    iterationCount: bind(new Uint32Array([777, 777]), 'uint32'),
    converged: bind(new Uint32Array([777, 777]), 'uint32'),
    iterationLimitReached: bind(new Uint32Array([777, 777]), 'uint32'),
    invalidCount: bind(new Uint32Array([777, 777]), 'uint32')
  };
  graph.add(
    new GPUCoregionalizationFit({
      variableCount: 2,
      structureCount: 1,
      maximumIterations: 4,
      distances: distances.view,
      semivariances: semivariances.view,
      pairCounts: pairCounts.view,
      parameters: parameters.view,
      output: {
        matrices: matrices.view,
        loadings: loadings.view,
        diagnostics: diagnostics.view,
        status: {
          validity: status.validity.view,
          iterationCount: status.iterationCount.view,
          converged: status.converged.view,
          iterationLimitReached: status.iterationLimitReached.view,
          invalidCount: status.invalidCount.view
        }
      }
    })
  );
  const compiled = graph.compile();
  try {
    for (const scenario of [
      'initial',
      'nonfinite',
      'invalid-model',
      'changed-model',
      'initial'
    ] as const) {
      semivariances.buffer.write(
        new Float32Array(scenario === 'nonfinite' ? [2, NaN, 3] : [2, -1, 3])
      );
      parameters.buffer.write(
        scenario === 'invalid-model'
          ? new Float32Array([1e-5, 9, 0, 1, 1])
          : scenario === 'changed-model'
            ? getGPUCoregionalizationFitParameterValues([{model: 'spherical', range: 1, sill: 2}])
            : initialParameters
      );
      submitGraph(device, compiled, undefined);
      const valid = scenario !== 'nonfinite' && scenario !== 'invalid-model';
      const fitted = await readFloat32(matrices.buffer, 8);
      for (let entry = 0; entry < 4; entry++) {
        if (valid)
          expect(fitted[entry], `${scenario}: matrices[${entry}]`).toBeCloseTo(
            [2, -1, -1, 3][entry] / (scenario === 'changed-model' ? 2 : 1),
            4
          );
        else expect(Number.isNaN(fitted[entry])).toBe(true);
      }
      expect(Array.from(fitted.slice(4))).toEqual([-777, -777, -777, -777]);
      expect(Array.from((await readFloat32(loadings.buffer, 8)).slice(4))).toEqual([
        -777, -777, -777, -777
      ]);
      expect(Array.from((await readFloat32(diagnostics.buffer, 8)).slice(4))).toEqual([
        -777, -777, -777, -777
      ]);
      for (const [name, output] of Object.entries(status)) {
        const values = await readUint32(output.buffer, 2);
        expect(values[1]).toBe(777);
        const expected =
          name === 'validity' || name === 'converged'
            ? Number(valid)
            : name === 'iterationCount'
              ? valid
                ? 2
                : 0
              : name === 'invalidCount'
                ? Number(scenario === 'nonfinite')
                : 0;
        expect(values[0], `${scenario}: ${name}`).toBe(expected);
      }
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
