// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it, expect} from 'vitest';
import type {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {GPUInstrumentalVariablesCovariance} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-instrumental-variables-covariance';
import {GPUSpatialTwoStageLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';
import {GPUSpatialCrossValidation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation';
import {GPURankMarkov} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-rank-markov';
import {GPUIslandAttachment} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-island-attachment';
import reference from '../../oracles/fixtures/statistics-coverage.json';

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

for (const method of ['HC0', 'HC1', 'HAC'] as const) {
  it(`IV covariance matches independent dense projection: ${method}`, async () => {
    const context = await setup();
    const {graph, input, output} = context;
    try {
      const covariance = output('float32', 9);
      const status = output('uint32', 1);
      graph.add(
        new GPUInstrumentalVariablesCovariance({
          design: input('float32', reference.design),
          coefficientCount: 2,
          instruments: input('float32', reference.instruments),
          instrumentCount: 2,
          residuals: input('float32', reference.residuals),
          method,
          ...(method === 'HAC'
            ? {
                kernel: {
                  offsets: input('uint32', reference.kernel.offsets),
                  neighbors: input('uint32', reference.kernel.neighbors),
                  weights: input('float32', reference.kernel.weights)
                }
              }
            : {}),
          output: {covariance: covariance.view, status: status.view}
        })
      );
      context.run();
      expect(Array.from(await readUint32(status.buffer, 1))).toEqual([0]);
      close(await readFloat32(covariance.buffer, 9), reference.covariances[method], 3e-4);
    } finally {
      context.destroy();
    }
  });
  it(`spatial TSLS retains estimates and replaces robust standard errors: ${method}`, async () => {
    const context = await setup();
    const {graph, input, output} = context;
    try {
      const covariance = output('float32', 9);
      const robustStatus = output('uint32', 1);
      const status = output('uint32', 1);
      const table = output('float32', 12);
      const summary = output('float32', 16);
      graph.add(
        new GPUSpatialTwoStageLeastSquares({
          predictors: input('float32', reference.predictors),
          predictorCount: 1,
          response: input('float32', reference.response),
          weights: {
            offsets: input('uint32', reference.weights.offsets),
            neighbors: input('uint32', reference.weights.neighbors),
            weights: input('float32', reference.weights.weights)
          },
          robust: {
            method,
            ...(method === 'HAC'
              ? {
                  kernel: {
                    offsets: input('uint32', reference.kernel.offsets),
                    neighbors: input('uint32', reference.kernel.neighbors),
                    weights: input('float32', reference.kernel.weights)
                  }
                }
              : {}),
            output: {covariance: covariance.view, status: robustStatus.view}
          },
          output: {table: table.view, summary: summary.view, status: {code: status.view}}
        })
      );
      context.run();
      expect(Array.from(await readUint32(status.buffer, 1))).toEqual([0]);
      expect(Array.from(await readUint32(robustStatus.buffer, 1))).toEqual([0]);
      const result = await readFloat32(table.buffer, 12);
      close([result[0], result[4], result[8]], reference.coefficients, 3e-4);
      close(
        [result[1], result[5], result[9]],
        [0, 4, 8].map(slot => Math.sqrt(reference.covariances[method][slot])),
        3e-4
      );
    } finally {
      context.destroy();
    }
  });
}
for (const scenario of ['singular', 'nonfinite', 'invalid-kernel', 'insufficient'] as const) {
  it(`IV covariance exposes ${scenario} without fabricated uncertainty`, async () => {
    const context = await setup();
    const {graph, input, output} = context;
    try {
      const rows = scenario === 'insufficient' ? 3 : 12;
      const covariance = output('float32', 9);
      const status = output('uint32', 1);
      graph.add(
        new GPUInstrumentalVariablesCovariance({
          design: input('float32', reference.design.slice(0, rows * 2)),
          coefficientCount: 2,
          instruments: input(
            'float32',
            scenario === 'singular'
              ? Array(rows * 2).fill(1)
              : reference.instruments.slice(0, rows * 2)
          ),
          instrumentCount: 2,
          residuals: input(
            'float32',
            scenario === 'nonfinite'
              ? reference.residuals.map((value, row) => (row === 2 ? NaN : value))
              : reference.residuals.slice(0, rows)
          ),
          method: scenario === 'invalid-kernel' ? 'HAC' : 'HC1',
          ...(scenario === 'invalid-kernel'
            ? {
                kernel: {
                  offsets: input('uint32', reference.kernel.offsets),
                  neighbors: input('uint32', reference.kernel.neighbors),
                  weights: input(
                    'float32',
                    reference.kernel.weights.map(value => (value === 1 ? 0.5 : value))
                  )
                }
              }
            : {}),
          output: {covariance: covariance.view, status: status.view}
        })
      );
      context.run();
      expect(Array.from(await readUint32(status.buffer, 1))).toEqual([
        {singular: 1, nonfinite: 3, 'invalid-kernel': 4, insufficient: 2}[scenario]
      ]);
      expect(Array.from(await readFloat32(covariance.buffer, 9)).every(Number.isNaN)).toBe(true);
    } finally {
      context.destroy();
    }
  });
}
it('spatial folds provide complete test coverage, block cohesion and strict buffered separation', async () => {
  const context = await setup();
  const {graph, input, output} = context;
  const positions = [0, 0, 0.1, 0, 2, 0, 2.1, 0, 4, 0, 4.1, 0, NaN, 0];
  try {
    const folds = output('uint32', 7);
    const training = output('uint32', 21);
    const testing = output('uint32', 21);
    const counts = output('uint32', 9);
    const status = output('uint32', 3);
    graph.add(
      new GPUSpatialCrossValidation({
        positions: input('float32x2', positions),
        blocks: input('uint32', [0, 0, 1, 1, 2, 2, 3]),
        foldCount: 3,
        bufferDistance: 2,
        seed: 17,
        output: {
          folds: folds.view,
          training: training.view,
          testing: testing.view,
          counts: counts.view,
          status: status.view
        }
      })
    );
    context.run();
    const assigned = await readUint32(folds.buffer, 7);
    const train = await readUint32(training.buffer, 21);
    const test = await readUint32(testing.buffer, 21);
    for (let row = 0; row < 6; row++) {
      expect([0, 1, 2].reduce((sum, fold) => sum + test[fold * 7 + row], 0)).toBe(1);
      if (row % 2 === 0) expect(assigned[row]).toBe(assigned[row + 1]);
    }
    expect(assigned[6]).toBe(0xffffffff);
    for (let fold = 0; fold < 3; fold++)
      for (let row = 0; row < 7; row++) {
        expect(train[fold * 7 + row] * test[fold * 7 + row]).toBe(0);
        if (train[fold * 7 + row])
          for (let other = 0; other < 7; other++)
            if (test[fold * 7 + other])
              expect(Math.abs(positions[2 * row] - positions[2 * other])).toBeGreaterThan(2);
      }
    const totals = await readUint32(counts.buffer, 9);
    const codes = await readUint32(status.buffer, 3);
    for (let fold = 0; fold < 3; fold++)
      expect(codes[fold]).toBe(
        (totals[3 * fold] === 0 ? 1 : 0) | (totals[3 * fold + 1] === 0 ? 2 : 0)
      );
  } finally {
    context.destroy();
  }
});
for (const mode of ['full-rank', 'geographic-rank'] as const) {
  it(`rank Markov ${mode} resolves ties and uses common complete panels`, async () => {
    const context = await setup();
    const {graph, input, output} = context;
    try {
      const ranks = output('uint32', 12);
      const counts = output('uint32', 16);
      const mobility = output('float32', 2);
      graph.add(
        new GPURankMarkov({
          tieOrder: 'row-id',
          mode,
          values: input('float32', [4, 4, 1, 8, 1, 5, 2, NaN, 5, 2, 1, 3]),
          periodCount: 3,
          output: {ranks: ranks.view, counts: counts.view, mobility: mobility.view}
        })
      );
      context.run();
      expect(Array.from(await readUint32(ranks.buffer, 12))).toEqual([
        0, 1, 2, 0xffffffff, 2, 0, 1, 0xffffffff, 0, 1, 2, 0xffffffff
      ]);
      close(await readFloat32(mobility.buffer, 2), [4 / 3, 4 / 3]);
      const expected = Array(16).fill(0);
      const rankRows = [
        [0, 1, 2],
        [2, 0, 1],
        [0, 1, 2]
      ];
      for (let period = 0; period < 2; period++)
        for (let row = 0; row < 3; row++) {
          const first =
            mode === 'full-rank' ? rankRows[period][row] : rankRows[period].indexOf(row);
          const second =
            mode === 'full-rank' ? rankRows[period + 1][row] : rankRows[period + 1].indexOf(row);
          expected[first * 4 + second]++;
        }
      expect(Array.from(await readUint32(counts.buffer, 16))).toEqual(expected);
    } finally {
      context.destroy();
    }
  });
}
// Powers of two preserve the fixture's exact ties after float32 upload.
for (const scale of [1, 2 ** -90, 2 ** 90, 2 ** 126])
  for (const capacity of [2, 8])
    it(`island attachment preserves nearest ordering and provenance at scale ${scale}, capacity ${capacity}`, async () => {
      const context = await setup();
      const {graph, input, output} = context;
      try {
        const offsets = output('uint32', 5);
        const neighbors = output('uint32', capacity);
        const weights = output('float32', capacity);
        const attachment = output('uint32', 4);
        const added = output('uint32', capacity);
        const overflow = output('uint32', 1);
        const required = output('uint32', 1);
        graph.add(
          new GPUIslandAttachment({
            weights: {
              offsets: input('uint32', [0, 1, 2, 2, 2]),
              neighbors: input('uint32', [1, 0]),
              weights: input('float32', [0.5, 0.5])
            },
            positions: input(
              'float32x2',
              [-3, 0, 3, 0, 2, 0, 2, 1].map(value => value * scale)
            ),
            output: {
              offsets: offsets.view,
              neighbors: neighbors.view,
              weights: weights.view,
              attachment: attachment.view,
              added: added.view,
              status: {overflow: overflow.view, requiredCount: required.view}
            }
          })
        );
        context.run();
        expect(Array.from(await readUint32(attachment.buffer, 4))).toEqual([
          0xffffffff, 0xffffffff, 1, 2
        ]);
        expect(Array.from(await readUint32(required.buffer, 1))).toEqual([6]);
        expect(Array.from(await readUint32(overflow.buffer, 1))).toEqual([capacity < 6 ? 1 : 0]);
        const written = Math.min(capacity, 6);
        expect(Array.from(await readUint32(neighbors.buffer, written))).toEqual(
          [1, 0, 2, 1, 3, 2].slice(0, written)
        );
        expect(Array.from(await readUint32(added.buffer, written))).toEqual(
          [0, 0, 1, 1, 1, 1].slice(0, written)
        );
      } finally {
        context.destroy();
      }
    });
