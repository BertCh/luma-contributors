// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it, expect} from 'vitest';
import type {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {GPUSpatialPanelRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-regression';
import reference from '../../oracles/fixtures/spatial-panel-fixed-effects.json';

async function setup() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    data: number[]
  ) => {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data)
    );
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `input-${serial++}`,
      buffer,
      format,
      data.length / (format === 'float32x2' ? 2 : 1)
    );
  };
  const output = <Format extends 'float32' | 'uint32'>(format: Format, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, `output-${serial++}`, buffer, format, length)};
  };
  let compiled: ReturnType<typeof graph.compile> | undefined;
  return {
    graph,
    input,
    output,
    run: () => {
      compiled = graph.compile();
      submitGraph(device, compiled, undefined);
    },
    destroy: () => {
      compiled?.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}
function close(actual: ArrayLike<number>, expected: number[], tolerance = 2e-4) {
  expected.forEach((value, index) =>
    Number.isNaN(value)
      ? expect(actual[index]).toBeNaN()
      : expect(
          Math.abs(actual[index] - value),
          `element ${index}: ${actual[index]} vs ${value}`
        ).toBeLessThan(tolerance)
  );
}

for (const scenario of ['reference', 'shifted', 'singular']) {
  const singular = scenario === 'singular';
  it(`unit panel matches independent dummy-variable IV: scenario=${scenario}`, async () => {
    const context = await setup();
    const {graph, input, output} = context;
    try {
      const table = output('float32', 12);
      const summary = output('float32', 16);
      const residuals = output('float32', reference.response.length);
      const status = output('uint32', 1);
      graph.add(
        new GPUSpatialPanelRegression({
          effects: 'unit',
          weights: {
            offsets: input('uint32', reference.offsets),
            neighbors: input('uint32', reference.neighbors),
            weights: input('float32', reference.weights)
          },
          periodCount: reference.periods,
          predictorCount: 1,
          predictors: input(
            'float32',
            singular
              ? reference.predictors.map((_, row) => row % reference.entities)
              : reference.predictors
          ),
          response: input(
            'float32',
            reference.response.map(
              (value, row) => value + (scenario === 'shifted' ? 20 * (row % reference.entities) : 0)
            )
          ),
          output: {
            table: table.view,
            summary: summary.view,
            residuals: residuals.view,
            status: {code: status.view}
          }
        })
      );
      context.run();
      expect(Array.from(await readUint32(status.buffer, 1))).toEqual([singular ? 1 : 0]);
      const result = await readFloat32(table.buffer, 12);
      if (singular) {
        expect(Array.from(result).every(Number.isNaN)).toBe(true);
      } else {
        close(result.slice(0, 4), [NaN, NaN, NaN, NaN]);
        close([result[4], result[8]], reference.coefficients, 1e-3);
        close([result[5], result[9]], reference.standardErrors, 1e-3);
        close(
          await readFloat32(residuals.buffer, reference.response.length),
          reference.residuals,
          1e-3
        );
        const diagnostics = await readFloat32(summary.buffer, 16);
        expect(diagnostics[1]).toBeCloseTo(reference.variance, 3);
        close(diagnostics.slice(4, 7), [NaN, NaN, NaN]);
      }
    } finally {
      context.destroy();
    }
  });
}
