// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {getPanelWithinNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/panel-within-nodes';
import {GPUSpatialPanelErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

async function createContext() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const input = <Format extends 'float32' | 'uint32'>(
    format: Format,
    values: Float32Array | Uint32Array
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, `f-reg-input-${serial++}`, buffer, format, values.length);
  };
  const output = <Format extends 'float32' | 'uint32'>(format: Format, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `f-reg-output-${serial++}`, buffer, format, length)
    };
  };
  let compiled: ReturnType<typeof graph.compile> | undefined;
  return {
    graph,
    input,
    output,
    run() {
      compiled = graph.compile();
      submitGraph(device, compiled, undefined);
    },
    destroy() {
      compiled?.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

for (const periodCount of [8, 1025, 32769]) {
  it(`shifted tiled panel means at ${periodCount} periods preserve small within deviations`, async () => {
    const entityCount = 2;
    const predictorCount = 2;
    const rows = entityCount * periodCount;
    const predictors = new Float32Array(rows * predictorCount);
    const response = new Float32Array(rows);
    const sums = new Float64Array(entityCount * 3);
    for (let period = 0; period < periodCount; period++) {
      for (let entity = 0; entity < entityCount; entity++) {
        const row = period * entityCount + entity;
        response[row] = (entity === 0 ? 3000000 : -3000000) + ((period % 7) - 3) * 0.25;
        predictors[row * 2] = 10000 + ((period % 5) - 2) * 0.03125;
        predictors[row * 2 + 1] = entity + ((period % 11) - 5) * 0.125;
        sums[entity * 3] += response[row];
        sums[entity * 3 + 1] += predictors[row * 2];
        sums[entity * 3 + 2] += predictors[row * 2 + 1];
      }
    }
    const context = await createContext();
    try {
      const centeredPredictors = context.output('float32', predictors.length);
      const centeredResponse = context.output('float32', rows);
      const nodes = getPanelWithinNodes(context.graph, {
        id: 'f-reg-within',
        entityCount,
        periodCount,
        predictorCount,
        predictors: context.input('float32', predictors),
        response: context.input('float32', response),
        output: {predictors: centeredPredictors.view, response: centeredResponse.view}
      });
      context.graph.add({getCommandNodes: () => nodes});
      context.run();
      const actualResponse = await readFloat32(centeredResponse.buffer, rows);
      const actualPredictors = await readFloat32(centeredPredictors.buffer, predictors.length);
      for (let row = 0; row < rows; row++) {
        const entity = row % entityCount;
        expect(
          Math.abs(actualResponse[row] - (response[row] - sums[entity * 3] / periodCount))
        ).toBeLessThan(1e-6);
        for (let column = 0; column < predictorCount; column++) {
          expect(
            Math.abs(
              actualPredictors[row * 2 + column] -
                (predictors[row * 2 + column] - sums[entity * 3 + column + 1] / periodCount)
            )
          ).toBeLessThan(1e-6);
        }
      }
    } finally {
      context.destroy();
    }
  });
}

it('panel means avoid overflow for finite observations of opposite signs', async () => {
  const context = await createContext();
  try {
    const values = Float32Array.of(3e38, -3e38, 3e38, -3e38);
    const predictors = context.output('float32', 4);
    const response = context.output('float32', 4);
    const nodes = getPanelWithinNodes(context.graph, {
      id: 'f-reg-large-within',
      entityCount: 1,
      periodCount: 4,
      predictorCount: 1,
      predictors: context.input('float32', values),
      response: context.input('float32', values),
      output: {predictors: predictors.view, response: response.view}
    });
    context.graph.add({getCommandNodes: () => nodes});
    context.run();
    // The exact mean is zero; every within deviation is the unchanged finite observation.
    expect(Array.from(await readFloat32(predictors.buffer, 4))).toEqual(Array.from(values));
    expect(Array.from(await readFloat32(response.buffer, 4))).toEqual(Array.from(values));
  } finally {
    context.destroy();
  }
});

for (const outlierPeriod of [0, 512, 1023]) {
  it(`two-part panel means preserve unit differences with an outlier at period ${outlierPeriod}`, async () => {
    const context = await createContext();
    try {
      const periodCount = 1024;
      const rows = periodCount * 2;
      const values = Float32Array.from({length: rows}, (_, row) =>
        Math.floor(row / 2) === outlierPeriod ? 1e8 : 1 + (Math.floor(row / 2) % 2)
      );
      const means = [0, 0];
      for (let row = 0; row < rows; row++) means[row % 2] += values[row] / periodCount;
      const predictors = context.output('float32', rows);
      const response = context.output('float32', rows);
      const nodes = getPanelWithinNodes(context.graph, {
        id: 'outlier-panel',
        entityCount: 2,
        periodCount,
        predictorCount: 1,
        predictors: context.input('float32', values),
        response: context.input('float32', values),
        output: {predictors: predictors.view, response: response.view}
      });
      context.graph.add({getCommandNodes: () => nodes});
      context.run();
      for (const output of [predictors, response]) {
        const actual = await readFloat32(output.buffer, rows);
        for (let row = 0; row < rows; row++) {
          const expected = Math.fround(values[row] - means[row % 2]);
          expect(Math.abs(actual[row] - expected)).toBeLessThanOrEqual(
            Math.max(1e-5, Math.abs(expected) * 2 ** -22)
          );
        }
        const first = outlierPeriod <= 1 ? 2 : 0;
        expect(actual[(first + 1) * 2] - actual[first * 2]).toBeCloseTo(1, 2);
      }
    } finally {
      context.destroy();
    }
  });
}

for (const periodCount of [1025, 32768]) {
  for (const overflow of [false, true]) {
    it(`complete panel-error fit and diagnostics at ${periodCount} periods, overflow=${overflow}`, {
      timeout: 120000
    }, async () => {
      const context = await createContext();
      try {
        const rows = periodCount * 2;
        const predictors = new Float32Array(rows);
        const response = new Float32Array(rows);
        const expectedResiduals = new Float32Array(rows);
        const completePeriods = 4 * Math.floor(periodCount / 4);
        for (let period = 0; period < periodCount; period++) {
          const included = period < completePeriods;
          const value = included ? [-1, 1, -1, 1][period % 4] : 0;
          const firstError = included ? [1, 1, -1, -1][period % 4] : 0;
          const secondError = included ? [1, -1, -1, 1][period % 4] : 0;
          predictors[period * 2] = value;
          predictors[period * 2 + 1] = value * 2;
          response[period * 2] = 3 + 2 * value + firstError;
          response[period * 2 + 1] = 3 + 4 * value + secondError;
          expectedResiduals[period * 2] = firstError;
          expectedResiduals[period * 2 + 1] = secondError;
        }
        const table = context.output('float32', 12);
        const summary = context.output('float32', 22);
        const residuals = context.output('float32', rows);
        const status = context.output('uint32', 1);
        const diagnostics = context.output('float32', 6);
        const diagnosticStatus = context.output('uint32', 1);
        context.graph.add(
          new GPUSpatialPanelErrorMaximumLikelihood({
            periodCount,
            predictorCount: 1,
            predictors: context.input('float32', predictors),
            response: context.input('float32', response),
            maximumPower: 16,
            traceMode: 'stochastic',
            traceProbeCount: 64,
            weights: {
              offsets: context.input('uint32', Uint32Array.of(0, 1, 2)),
              neighbors: context.input('uint32', Uint32Array.of(1, 0)),
              weights: context.input('float32', Float32Array.of(1, 1)),
              status: {overflow: context.input('uint32', Uint32Array.of(Number(overflow)))}
            },
            output: {
              table: table.view,
              summary: summary.view,
              residuals: residuals.view,
              status: {code: status.view}
            },
            diagnostics: {tests: diagnostics.view, status: diagnosticStatus.view}
          })
        );
        context.run();
        expect(Array.from(await readUint32(status.buffer, 1))).toEqual([overflow ? 3 : 0]);
        const estimates = await readFloat32(table.buffer, 12);
        const actualResiduals = await readFloat32(residuals.buffer, rows);
        if (overflow) {
          expect(estimates.every(Number.isNaN)).toBe(true);
          expect(actualResiduals.every(Number.isNaN)).toBe(true);
          expect(Array.from(await readUint32(diagnosticStatus.buffer, 1))).toEqual([1]);
        } else {
          expect(estimates[0]).toBeNaN();
          expect(estimates[4]).toBeCloseTo(2, 3);
          // The exact optimum is lambda=0 and sigma2=completePeriods/periodCount. The large-row
          // trace route is stochastic, so allow its documented small sampling displacement.
          expect(Math.abs(estimates[8])).toBeLessThan(0.025);
          const values = await readFloat32(summary.buffer, 22);
          expect(Math.abs(values[2] - completePeriods / periodCount)).toBeLessThan(0.01);
          let maximumResidualError = 0;
          for (let row = 0; row < rows; row++)
            maximumResidualError = Math.max(
              maximumResidualError,
              Math.abs(actualResiduals[row] - expectedResiduals[row])
            );
          expect(maximumResidualError).toBeLessThan(0.01);
          expect(Array.from(await readUint32(diagnosticStatus.buffer, 1))).toEqual([0]);
          const tests = await readFloat32(diagnostics.buffer, 6);
          for (const index of [0, 2, 4]) {
            expect(Math.abs(tests[index])).toBeLessThan(1e-4);
            expect(tests[index + 1]).toBeCloseTo(1, 3);
          }
        }
      } finally {
        context.destroy();
      }
    });
  }
}
