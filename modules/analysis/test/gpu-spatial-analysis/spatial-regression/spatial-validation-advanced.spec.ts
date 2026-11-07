// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUSpatialCrossValidation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation';
import type {GPUSpatialValidationDesign} from '../../../src/gpu-spatial-analysis/spatial-validation/spatial-validation-design';
import {GPUNestedInterpolationValidation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation';
import {GPUSpatialCrossValidatedOLS} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validated-ols';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import nestedReference from '../../oracles/fixtures/nested-validation-oracle.json';
import reference from './spatial-validation-advanced-reference.json';
async function createHarness() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const inputBuffers = new Map<GraphDataView, Buffer>();
  let nextBuffer = 0;
  return {
    graph,
    writeInput(view: GraphDataView, values: readonly number[]) {
      inputBuffers.get(view)!.write(new Float32Array(values));
    },
    destroy() {
      for (const buffer of buffers) buffer.destroy();
    },
    input<Format extends GPUVectorFormat>(data: readonly number[], format: Format, components = 1) {
      const buffer = createInputBuffer(
        device,
        format === 'uint32' ? new Uint32Array(data) : new Float32Array(data)
      );
      buffers.push(buffer);
      const view = importGraphBuffer(
        graph,
        `input-${nextBuffer++}`,
        buffer,
        format,
        data.length / components
      );
      inputBuffers.set(view, buffer);
      return view;
    },
    output<Format extends GPUVectorFormat>(length: number, format: Format, components = 1) {
      const buffer = createOutputBuffer(device, length * components);
      buffers.push(buffer);
      const view = importGraphBuffer(graph, `output-${nextBuffer++}`, buffer, format, length);
      return {
        view,
        read: () =>
          format === 'uint32'
            ? readUint32(buffer, length * components)
            : readFloat32(buffer, length * components)
      };
    },
    async run() {
      const compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      return () => {
        compiled.destroy();
        for (const buffer of buffers) {
          buffer.destroy();
        }
      };
    }
  };
}

it('cross-validated OLS reports held-out metric arithmetic overflow', async () => {
  const harness = await createHarness();
  const metrics = harness.output(12, 'float32');
  const metricStatus = harness.output(2, 'uint32');
  const fitStatus = harness.output(2, 'uint32');
  harness.graph.add(
    new GPUSpatialCrossValidatedOLS({
      folds: {
        positions: harness.input(
          Array.from({length: 8}, (_, row) => [row, 0]).flat(),
          'float32x2',
          2
        ),
        blocks: harness.input([0, 0, 0, 0, 1, 1, 1, 1], 'uint32'),
        design: {mode: 'groups'},
        foldCount: 2,
        output: {
          folds: harness.output(8, 'uint32').view,
          training: harness.output(16, 'uint32').view,
          testing: harness.output(16, 'uint32').view,
          counts: harness.output(6, 'uint32').view,
          status: harness.output(2, 'uint32').view
        }
      },
      predictors: harness.input([0, 1, 2, 3, 4, 5, 6, 7], 'float32'),
      predictorCount: 1,
      response: harness.input([1e20, 2e20, 1e20, 2e20, 9, 11, 13, 15], 'float32'),
      output: {
        coefficients: harness.output(4, 'float32').view,
        predictions: harness.output(16, 'float32').view,
        fitStatus: fitStatus.view,
        metrics: metrics.view,
        metricStatus: metricStatus.view
      }
    })
  );
  const cleanup = await harness.run();
  try {
    expect((await fitStatus.read())[0]).toBe(0);
    expect((await metricStatus.read())[0] & 8).toBe(8);
    const values = await metrics.read();
    expect(Array.from(values.slice(0, 2))).toEqual([4, 0]);
    expect(Array.from(values.slice(2, 6)).every(Number.isNaN)).toBe(true);
  } finally {
    cleanup();
  }
});

for (const mode of ['groups', 'environmental', 'nndm', 'knndm'] as const) {
  it(`${mode} resident fold design matches direct blockCV4.0.1 reference`, async () => {
    const harness = await createHarness();
    const count = 8,
      folds = mode === 'nndm' ? 8 : mode === 'groups' ? 3 : 2;
    const output = {
      folds: harness.output(count, 'uint32'),
      training: harness.output(count * folds, 'uint32'),
      testing: harness.output(count * folds, 'uint32'),
      counts: harness.output(3 * folds, 'uint32'),
      status: harness.output(folds, 'uint32')
    };
    const diagnostics = harness.output(mode === 'nndm' ? 12 : 4, 'float32');
    let design: GPUSpatialValidationDesign;
    if (mode === 'groups') design = {mode, diagnostics: diagnostics.view};
    else if (mode === 'environmental')
      design = {
        mode,
        features: harness.input(reference.environmental.features, 'float32'),
        featureCount: 1,
        standardize: false,
        diagnostics: diagnostics.view
      };
    else if (mode === 'nndm')
      design = {
        mode,
        predictionPositions: harness.input(reference.predictionPositions, 'float32x2', 2),
        maximumDistance: 4,
        minimumTrainingFraction: 0.2,
        diagnostics: diagnostics.view
      };
    else
      design = {
        mode,
        predictionPositions: harness.input(reference.predictionPositions, 'float32x2', 2),
        candidates: harness.input(reference.knndm.candidates, 'uint32'),
        candidateCount: reference.knndm.candidateCount,
        maximumFoldFraction: 0.8,
        diagnostics: diagnostics.view
      };
    harness.graph.add(
      new GPUSpatialCrossValidation({
        positions: harness.input(reference.positions, 'float32x2', 2),
        blocks: harness.input(reference.groups, 'uint32'),
        foldCount: folds,
        seed: 17,
        design,
        output: {
          folds: output.folds.view,
          training: output.training.view,
          testing: output.testing.view,
          counts: output.counts.view,
          status: output.status.view
        }
      })
    );
    const cleanup = await harness.run();
    try {
      const assigned = Array.from(await output.folds.read());
      const values = await diagnostics.read();
      expect(values[0]).toBe(0);
      if (mode === 'groups') expect(assigned).toEqual(reference.groupFolds);
      if (mode === 'environmental' || mode === 'knndm') {
        const expected =
          mode === 'environmental' ? reference.environmental.folds : reference.knndm.folds;
        for (let first = 0; first < count; first++)
          for (let second = 0; second < count; second++)
            expect(assigned[first] === assigned[second]).toBe(expected[first] === expected[second]);
      }
      if (mode === 'knndm') expect(values[2]).toBeCloseTo(reference.knndm.objective, 4);
      if (mode === 'nndm') {
        for (let row = 0; row < count; row++)
          expect(values[row + 4]).toBeCloseTo(reference.nndm.radii[row], 4);
        expect(Array.from(await output.training.read())).toEqual(reference.nndm.training);
      }
      const training = await output.training.read(),
        testing = await output.testing.read();
      for (let row = 0; row < count; row++) {
        let covered = 0;
        for (let fold = 0; fold < folds; fold++) {
          covered += testing[fold * count + row];
          expect(training[fold * count + row] * testing[fold * count + row]).toBe(0);
        }
        expect(covered).toBe(1);
      }
    } finally {
      cleanup();
    }
  });
}

it('nested variogram and RBF bandwidth search keeps heldout outcomes out of selection and predictions', async () => {
  const harness = await createHarness(),
    rows = 12,
    folds = 2;
  const positions = Array.from({length: rows}, (_, row) => [row % 4, Math.floor(row / 4)]).flat();
  const response = Array.from(
    {length: rows},
    (_, row) => 2 + 0.3 * positions[row * 2] + 0.8 * positions[row * 2 + 1]
  );
  const input = harness.input(response, 'float32');
  const output = {
    selected: harness.output(folds, 'uint32'),
    scores: harness.output(folds * 3 * 3, 'float32'),
    predictions: harness.output(rows * folds, 'float32'),
    metrics: harness.output(folds * 6, 'float32'),
    status: harness.output(folds, 'uint32')
  };
  harness.graph.add(
    new GPUNestedInterpolationValidation({
      positions: harness.input(positions, 'float32x2', 2),
      response: input,
      training: harness.input(
        Array.from({length: rows * folds}, (_, index) =>
          Number((index % rows) % 2 !== Math.floor(index / rows))
        ),
        'uint32'
      ),
      testing: harness.input(
        Array.from({length: rows * folds}, (_, index) =>
          Number((index % rows) % 2 === Math.floor(index / rows))
        ),
        'uint32'
      ),
      foldCount: folds,
      innerBlocks: harness.input(
        Array.from({length: rows}, (_, row) => Math.floor(row / 2)),
        'uint32'
      ),
      innerFoldCount: 2,
      seed: 2,
      neighborCount: 4,
      candidates: [
        {
          mode: 'ordinary-kriging',
          covariance: {model: 'exponential', sill: 1, range: 3, nugget: 0.01}
        },
        {mode: 'rbf', kernel: 'gaussian', epsilon: 0.5, smoothing: 0.1},
        {mode: 'rbf', kernel: 'gaussian', epsilon: 1, smoothing: 0.1}
      ],
      output: {
        selected: output.selected.view,
        scores: output.scores.view,
        predictions: output.predictions.view,
        metrics: output.metrics.view,
        status: output.status.view
      }
    })
  );
  const compiled = harness.graph.compile();
  submitGraph(harness.graph.device, compiled, undefined);
  const before = await output.predictions.read(),
    scores = await output.scores.read(),
    selected = await output.selected.read();
  const metrics = await output.metrics.read();
  for (let fold = 0; fold < folds; fold++) {
    const expected = nestedReference.folds[fold];
    expect(selected[fold]).toBe(expected.selected);
    expected.scores
      .flat()
      .forEach((value, index) => expect(scores[fold * 9 + index]).toBeCloseTo(value, 4));
    expected.metrics.forEach((value, index) =>
      expect(metrics[fold * 6 + index]).toBeCloseTo(value, 4)
    );
    expected.predictions.forEach((value, row) =>
      value === null
        ? expect(Number.isNaN(before[fold * rows + row])).toBe(true)
        : expect(before[fold * rows + row]).toBeCloseTo(value, 4)
    );
  }
  expect(Array.from(await output.status.read())).toEqual([0, 0]);
  // Import exposes a borrowed physical buffer; change only outer fold0 test outcomes.
  const changed = response.map((value, row) => (row % 2 === 0 ? value + 1000 : value));
  harness.writeInput(input, changed);
  submitGraph(harness.graph.device, compiled, undefined);
  const after = await output.predictions.read(),
    nextScores = await output.scores.read();
  for (let row = 0; row < rows; row += 2) expect(after[row]).toBe(before[row]);
  for (let index = 0; index < 9; index++) expect(nextScores[index]).toBe(scores[index]);
  expect((await output.selected.read())[0]).toBe(selected[0]);
  expect((await output.metrics.read())[2]).toBeGreaterThan(100000);
  harness.writeInput(
    input,
    response.map((value, row) => (row % 2 === 0 ? 1e20 : value))
  );
  submitGraph(harness.graph.device, compiled, undefined);
  expect((await output.status.read())[0]).toBe(16);
  expect(Number.isNaN((await output.metrics.read())[2])).toBe(true);
  compiled.destroy();
  harness.destroy();
});

for (const failure of [
  'environment-exhausted',
  'environment-singular',
  'nndm-invalid',
  'nndm-overflow',
  'knndm-overflow',
  'knndm-invalid',
  'knndm-generated'
] as const) {
  it(`advanced validation reports ${failure} without leaking stale folds`, async () => {
    const harness = await createHarness(),
      rows = 8,
      foldCount = failure.startsWith('nndm') ? 8 : 2;
    const folds = harness.output(rows, 'uint32'),
      training = harness.output(rows * foldCount, 'uint32'),
      testing = harness.output(rows * foldCount, 'uint32'),
      counts = harness.output(foldCount * 3, 'uint32'),
      status = harness.output(foldCount, 'uint32'),
      diagnostics = harness.output(failure.startsWith('nndm') ? 12 : 4, 'float32');
    const design: GPUSpatialValidationDesign = failure.startsWith('environment')
      ? {
          mode: 'environmental',
          features: harness.input(
            failure === 'environment-singular'
              ? Array(rows).fill(1)
              : reference.environmental.features,
            'float32'
          ),
          featureCount: 1,
          maximumIterations: failure === 'environment-exhausted' ? 1 : 100,
          diagnostics: diagnostics.view
        }
      : failure.startsWith('nndm')
        ? {
            mode: 'nndm',
            predictionPositions: harness.input(
              failure === 'nndm-invalid' ? [NaN, 0] : [4e20, 0],
              'float32x2',
              2
            ),
            maximumDistance: 2,
            diagnostics: diagnostics.view
          }
        : {
            mode: 'knndm',
            predictionPositions: harness.input(
              failure === 'knndm-overflow' ? [4e20, 0] : reference.predictionPositions,
              'float32x2',
              2
            ),
            ...(failure === 'knndm-invalid'
              ? {candidates: harness.input(Array(rows).fill(0), 'uint32'), candidateCount: 1}
              : {candidateCount: 4}),
            diagnostics: diagnostics.view
          };
    harness.graph.add(
      new GPUSpatialCrossValidation({
        positions: harness.input(reference.positions, 'float32x2', 2),
        blocks: harness.input(reference.groups, 'uint32'),
        foldCount,
        design,
        output: {
          folds: folds.view,
          training: training.view,
          testing: testing.view,
          counts: counts.view,
          status: status.view
        }
      })
    );
    const cleanup = await harness.run();
    try {
      const state = await diagnostics.read();
      if (failure === 'knndm-generated') {
        expect(state[0]).toBe(0);
        expect(state[2]).toBeGreaterThanOrEqual(0);
        expect(Array.from(await folds.read()).every(value => value < 2)).toBe(true);
      } else {
        expect(state[0]).toBe(
          failure === 'environment-exhausted'
            ? 3
            : failure.startsWith('nndm') || failure.endsWith('overflow')
              ? 1
              : 2
        );
        expect(Array.from(await folds.read())).toEqual(Array(rows).fill(0xffffffff));
        expect(Array.from(await training.read()).every(value => value === 0)).toBe(true);
      }
    } finally {
      cleanup();
    }
  });
}

it('nested validation publishes explicit empty-fold and overlapping-mask failures', async () => {
  const harness = await createHarness(),
    rows = 8;
  const output = {
    selected: harness.output(2, 'uint32'),
    scores: harness.output(6, 'float32'),
    predictions: harness.output(16, 'float32'),
    metrics: harness.output(12, 'float32'),
    status: harness.output(2, 'uint32')
  };
  harness.graph.add(
    new GPUNestedInterpolationValidation({
      positions: harness.input(reference.positions, 'float32x2', 2),
      response: harness.input(reference.environmental.features, 'float32'),
      training: harness.input([...Array(rows).fill(1), ...Array(rows).fill(0)], 'uint32'),
      testing: harness.input([...Array(rows).fill(1), ...Array(rows).fill(0)], 'uint32'),
      foldCount: 2,
      innerBlocks: harness.input(reference.groups, 'uint32'),
      innerFoldCount: 2,
      candidates: [{mode: 'rbf', epsilon: 1, kernel: 'gaussian', smoothing: 0.1}],
      output: {
        selected: output.selected.view,
        scores: output.scores.view,
        predictions: output.predictions.view,
        metrics: output.metrics.view,
        status: output.status.view
      }
    })
  );
  const cleanup = await harness.run();
  try {
    expect(Array.from(await output.selected.read())).toEqual([0xffffffff, 0xffffffff]);
    expect(Array.from(await output.status.read())).toEqual([7, 10]);
    expect(Array.from(await output.predictions.read()).every(Number.isNaN)).toBe(true);
  } finally {
    cleanup();
  }
});
