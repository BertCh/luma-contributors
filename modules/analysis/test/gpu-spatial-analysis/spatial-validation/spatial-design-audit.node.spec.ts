// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeSpatialCrossValidationOnCPU,
  CPUSpatialCrossValidationOperation
} from '../../../src/gpu-spatial-analysis/spatial-validation/cpu-spatial-cross-validation-operation';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUSpatialCrossValidation,
  type GPUSpatialCrossValidationProps
} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation';
import {
  GPUSpatialCrossValidationOperation,
  type SpatialCrossValidationOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation-operation';
import type {GPUSpatialValidationDesign} from '../../../src/gpu-spatial-analysis/spatial-validation/spatial-validation-design';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {createSamplingCPUInputs} from '../point-process/validation-sampling-operation-test-utils';
import {watchConstructionFields} from './construction-capture-audit.test-utils';
import {
  BLOCK_AUDIT_POSITIONS,
  BLOCK_AUDIT_BLOCKS,
  BLOCK_AUDIT_MASK,
  BLOCK_AUDIT_EXPECTED,
  executeSpatialAuditKernel
} from './spatial-design-audit.test-utils';

it.each([
  false,
  true
])('W5-25 preserves standardized coordinate differences (reciprocal division=%s)', reciprocalDivision => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `standardized-differences-${serial++}`, format, length);
  try {
    for (const standardize of [true, false]) {
      for (const featureCount of [1, 2]) {
        const low = Math.fround(0.1),
          high = low + 2 ** -25;
        const features = Float32Array.from(
          [3, low, high, 3, 3, high, high, high].flatMap(value =>
            featureCount === 1 ? [value] : [value, 7]
          )
        );
        const positions = Float32Array.from({length: 16}, (_, index) =>
          index % 2 === 0 ? index / 2 : 0
        );
        const contributor = new GPUSpatialCrossValidation({
          id: `standardized-${standardize}-${featureCount}`,
          positions: view('float32x2', 8),
          blocks: view('uint32', 8),
          foldCount: 3,
          seed: 0,
          design: {
            mode: 'environmental',
            features: view('float32', 8 * featureCount),
            featureCount,
            standardize
          },
          output: {
            folds: view('uint32', 8),
            training: view('uint32', 24),
            testing: view('uint32', 24),
            counts: view('uint32', 9),
            status: view('uint32', 3)
          }
        });
        const values = {
          positions,
          features,
          centeredFeatures: new Float32Array(features.length),
          folds: new Uint32Array(8),
          state: new Float32Array(4),
          roundingZero: new Uint32Array(1)
        };
        for (const kernel of getGPUCommandNodeKernelSources(
          contributor
            .getCommandNodes(graph)
            .filter(node => node.id.endsWith('-center-features') || node.id.endsWith('-design'))
        ))
          executeSpatialAuditKernel(kernel, values, {
            expandArrayIncrement: true,
            reciprocalDivision
          });
        // Exactly three distinct values. Farthest-first seeds 3, low, high; each
        // cluster is constant, hence its independent objective is exactly zero.
        expect(Array.from(values.state)).toEqual([0, 2, 0, NaN]);
        expect(Array.from(values.folds)).toEqual([0, 1, 2, 0, 0, 2, 2, 2]);
        const cpu = computeSpatialCrossValidationOnCPU(
          {positions, blocks: new Uint32Array(8), features},
          {foldCount: 3, seed: 0, design: {mode: 'environmental', featureCount, standardize}}
        );
        expect(Array.from(cpu.folds)).toEqual([0, 1, 2, 0, 0, 2, 2, 2]);
        expect(Array.from(cpu.status)).toEqual([0, 0, 0]);
        expect(Array.from(cpu.counts)).toEqual([5, 3, 0, 7, 1, 0, 4, 4, 0]);
      }
    }
  } finally {
    device.destroy();
  }
});

it.each([
  false,
  true
])('review 2 R1 centering preserves three distinct packed environmental groups (reciprocal division=%s)', reciprocalDivision => {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  let nextView = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `distinct-centering-${nextView++}`, format, length);
  try {
    const contributor = new GPUSpatialCrossValidation({
      positions: view('float32x2', 8),
      blocks: view('uint32', 8),
      foldCount: 3,
      seed: 0,
      design: {
        mode: 'environmental',
        features: view('float32', 8),
        featureCount: 1,
        standardize: false
      },
      output: {
        folds: view('uint32', 8),
        training: view('uint32', 24),
        testing: view('uint32', 24),
        counts: view('uint32', 9),
        status: view('uint32', 3)
      }
    });
    const kernels = getGPUCommandNodeKernelSources(
      contributor
        .getCommandNodes(graph)
        .filter(node => node.id.endsWith('-center-features') || node.id.endsWith('-design'))
    );
    for (const [low, high] of [
      [1, 1 + 2 ** -22],
      [1 + 2 ** -21, 10 + 2 ** -20]
    ]) {
      const features = Float32Array.of(10, low, high, low, high, 10, low, high);
      const positions = Float32Array.from({length: 16}, (_, index) =>
        index % 2 === 0 ? index / 2 : 0
      );
      const cpu = computeSpatialCrossValidationOnCPU(
        {positions, blocks: new Uint32Array(8), features},
        {
          foldCount: 3,
          seed: 0,
          design: {mode: 'environmental', featureCount: 1, standardize: false}
        }
      );
      // Three identical-value groups, seeded 10 then the low value then the third value.
      // Each fitted mean equals its group value, hence the independent SSE is exactly zero.
      expect(Array.from(cpu.folds)).toEqual([0, 1, 2, 1, 2, 0, 1, 2]);
      expect(cpu.diagnostics![2]).toBe(0);
      const values = {
        positions,
        features,
        centeredFeatures: new Float32Array(8),
        folds: new Uint32Array(8),
        state: new Float32Array(4),
        roundingZero: new Uint32Array(1)
      };
      executeSpatialAuditKernel(kernels[0], values);
      if (low > 1) expect(Array.from(values.centeredFeatures)).toEqual(Array.from(features));
      else
        expect(Array.from(values.centeredFeatures)).toEqual([
          9,
          0,
          2 ** -22,
          0,
          2 ** -22,
          9,
          0,
          2 ** -22
        ]);
      executeSpatialAuditKernel(kernels[1], values, {
        expandArrayIncrement: true,
        reciprocalDivision
      });
      expect(
        values.state[0],
        `low ${low}; high ${high}; diagnostics ${Array.from(values.state)}`
      ).toBe(0);
      expect(values.state[2]).toBe(0);
      expect(Array.from(values.state)).toEqual([0, 2, 0, NaN]);
      expect(Array.from(values.folds)).toEqual([0, 1, 2, 1, 2, 0, 1, 2]);
    }
  } finally {
    device.destroy();
  }
});

it('R8 environmental Lloyd labels and objective retain representable feature translations', () => {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  let nextView = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `translated-lloyd-${nextView++}`, format, length);
  try {
    const contributor = new GPUSpatialCrossValidation({
      positions: view('float32x2', 8),
      blocks: view('uint32', 8),
      foldCount: 2,
      seed: 0,
      design: {
        mode: 'environmental',
        features: view('float32', 8),
        featureCount: 1,
        standardize: false
      },
      output: {
        folds: view('uint32', 8),
        training: view('uint32', 16),
        testing: view('uint32', 16),
        counts: view('uint32', 6),
        status: view('uint32', 2)
      }
    });
    const kernels = getGPUCommandNodeKernelSources(
      contributor
        .getCommandNodes(graph)
        .filter(node => node.id.endsWith('-center-features') || node.id.endsWith('-design'))
    );
    expect(kernels).toHaveLength(2);
    for (const shift of [0, 500000, -500000]) {
      const values = {
        positions: Float32Array.from({length: 16}, (_, index) => (index % 2 === 0 ? index / 2 : 0)),
        features: Float32Array.from([0, 1, 2, 5, 6, 7, 9, 11], value => shift + value / 32),
        centeredFeatures: new Float32Array(8),
        folds: new Uint32Array(8),
        state: new Float32Array(4),
        roundingZero: new Uint32Array(1)
      };
      executeSpatialAuditKernel(kernels[0], values);
      expect(Array.from(values.centeredFeatures)).toEqual([
        0, 0.03125, 0.0625, 0.15625, 0.1875, 0.21875, 0.28125, 0.34375
      ]);
      // Once centered values are materialized, the fit must not depend on the raw
      // translated buffer. This also prevents a compiler from rebuilding raw centroids.
      values.features.fill(NaN);
      executeSpatialAuditKernel(kernels[1], values, {expandArrayIncrement: true});
      // Stable means 2/32 and 8.25/32: SSE=(14+14.75)/1024 exactly.
      expect(Array.from(values.folds)).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
      expect(values.state[0]).toBe(0);
      expect(values.state[2]).toBe(0.028076171875);
    }
    const overflow = {
      positions: Float32Array.from({length: 16}, (_, index) => (index % 2 === 0 ? index / 2 : 0)),
      features: Float32Array.of(-3e38, -3e38, -3e38, -3e38, 3e38, 3e38, 3e38, 3e38),
      centeredFeatures: new Float32Array(8),
      folds: new Uint32Array(8),
      state: new Float32Array(4),
      roundingZero: new Uint32Array(1)
    };
    executeSpatialAuditKernel(kernels[0], overflow);
    // No safe nonzero origin exists; the fallback copies every packed raw value.
    expect(Array.from(overflow.centeredFeatures)).toEqual(
      Array.from(Float32Array.of(-3e38, -3e38, -3e38, -3e38, 3e38, 3e38, 3e38, 3e38))
    );
    executeSpatialAuditKernel(kernels[1], overflow, {expandArrayIncrement: true});
    // All raw rows are usable; subtraction overflow is status 1, not an omitted
    // four-row constant cluster (which would incorrectly yield infeasible status 2).
    expect(overflow.state[0]).toBe(1);
    expect(Array.from(overflow.folds)).toEqual(new Array(8).fill(0xffffffff));
  } finally {
    device.destroy();
  }
});

it('R2 NNDM masks honor the larger explicit buffer on CPU and in generated WGSL', () => {
  const positions = Float32Array.from({length: 16}, (_, index) =>
    index % 2 === 0 ? index / 2 : 0
  );
  const actual = computeSpatialCrossValidationOnCPU(
    {positions, blocks: new Uint32Array(8), predictionPositions: Float32Array.of(0, 0)},
    {
      foldCount: 8,
      bufferDistance: 3,
      design: {mode: 'nndm', maximumDistance: 0}
    }
  );
  expect(actual.diagnostics![4]).toBe(1);
  expect.soft(Array.from(actual.training.slice(0, 8))).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
  expect.soft(Array.from(actual.counts.slice(0, 3))).toEqual([4, 1, 3]);
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  let nextView = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `nndm-buffer-${nextView++}`, format, length);
  try {
    const contributor = new GPUSpatialCrossValidation({
      positions: view('float32x2', 8),
      blocks: view('uint32', 8),
      foldCount: 8,
      bufferDistance: 3,
      design: {mode: 'nndm', maximumDistance: 0, predictionPositions: view('float32x2', 1)},
      output: {
        folds: view('uint32', 8),
        training: view('uint32', 64),
        testing: view('uint32', 64),
        counts: view('uint32', 24),
        status: view('uint32', 8)
      }
    });
    const values = {
      positions,
      folds: Uint32Array.from({length: 8}, (_, row) => row),
      designState: Float32Array.of(0, 0, NaN, NaN, 1, 1, 1, 1, 1, 1, 1, 1),
      adaptiveNeighbors: Uint32Array.of(1, 0, 1, 2, 3, 4, 5, 6),
      training: new Uint32Array(64),
      testing: new Uint32Array(64),
      counts: new Uint32Array(24),
      status: new Uint32Array(8)
    };
    for (const kernel of getGPUCommandNodeKernelSources(
      contributor
        .getCommandNodes(graph)
        .filter(node => node.id.endsWith('-masks') || node.id.endsWith('-counts'))
    ))
      executeSpatialAuditKernel(kernel, values);
    expect(Array.from(values.training.slice(0, 8))).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
    expect(Array.from(values.counts.slice(0, 3))).toEqual([4, 1, 3]);
  } finally {
    device.destroy();
  }
});

it('R1 preserves CPU convergence by default while an explicit 100-step budget binds', async () => {
  const rows = 2048;
  const input = {
    positions: Float32Array.from({length: rows * 2}, (_, index) =>
      index % 2 === 0 ? index / 2 : 0
    ),
    blocks: new Uint32Array(rows),
    features: Float32Array.from({length: rows}, (_, row) => row / 2047)
  };
  for (const maximumIterations of [undefined, 100, 150]) {
    const actual = computeSpatialCrossValidationOnCPU(input, {
      foldCount: 13,
      seed: 0,
      design: {mode: 'environmental', featureCount: 1, standardize: false, maximumIterations}
    });
    // Distinct ordered sites have a nonempty stable 13-cluster partition. The 100th
    // Lloyd assignment still changes; omission must retain convergence as the CPU default.
    const exhausted = maximumIterations === 100;
    expect(actual.diagnostics![0]).toBe(exhausted ? 3 : 0);
    expect(actual.diagnostics![1]).toBe(exhausted ? 100 : 102);
    expect(actual.folds.filter(fold => fold < 13)).toHaveLength(exhausted ? 0 : rows);
    expect(Array.from(actual.status)).toEqual(new Array(13).fill(exhausted ? 3 : 0));
    // spatial-design-audit.oracle.py: SciPy vq census at convergence, in cluster-ID order.
    const foldSizes = [149, 167, 159, 164, 153, 156, 162, 165, 150, 151, 154, 158, 160];
    expect(Array.from(actual.counts)).toEqual(
      exhausted ? new Array(39).fill(0) : foldSizes.flatMap(count => [2048 - count, count, 0])
    );
    if (!exhausted) expect(actual.diagnostics![2]).toBe(Math.fround(1.0147895258459558));
    const operation = new CPUSpatialCrossValidationOperation({
      rowCount: rows,
      positions: 'positions',
      blocks: 'blocks',
      foldCount: 13,
      seed: 0,
      design: {
        mode: 'environmental',
        features: 'features',
        featureCount: 1,
        standardize: false,
        maximumIterations,
        diagnostics: 'diagnostics'
      },
      output: {
        folds: 'folds',
        training: 'training',
        testing: 'testing',
        counts: 'counts',
        status: 'status'
      }
    });
    const callback = await operation.execute({
      inputs: createSamplingCPUInputs({
        positions: {format: 'float32x2', values: input.positions},
        blocks: {format: 'uint32', values: input.blocks},
        features: {format: 'float32', values: input.features}
      })
    });
    expect(Array.from(callback['status'] as Uint32Array)).toEqual(
      new Array(13).fill(exhausted ? 3 : 0)
    );
    expect(Array.from(callback['folds'] as Uint32Array).filter(fold => fold < 13)).toHaveLength(
      exhausted ? 0 : rows
    );
    expect((callback['diagnostics'] as Float32Array)[1]).toBe(exhausted ? 100 : 102);
    expect(Array.from(callback['counts'] as Uint32Array)).toEqual(
      exhausted ? new Array(39).fill(0) : foldSizes.flatMap(count => [2048 - count, count, 0])
    );
  }
});

it('R1 generated CPU kNNDM restarts also retain convergence by default', () => {
  const actual = computeSpatialCrossValidationOnCPU(
    {
      positions: Float32Array.from({length: 4096}, (_, index) =>
        index % 2 === 0 ? index / 2 / 2047 : 0
      ),
      blocks: new Uint32Array(2048),
      predictionPositions: Float32Array.of(0, 0)
    },
    {foldCount: 13, seed: 0, design: {mode: 'knndm', candidateCount: 1}}
  );
  // One valid farthest-first ladder candidate must be selected after its 102nd Lloyd step.
  expect(actual.diagnostics![0]).toBe(0);
  expect(actual.diagnostics![3]).toBe(0);
  expect(actual.folds.filter(fold => fold < 13)).toHaveLength(2048);
});

it('audit-validation binds an explicit Lloyd budget and suppresses unconverged folds', () => {
  const input = {
    positions: new Float32Array([0, 0, 1, 0, 2, 0, 3, 0, 10, 0, 11, 0, 12, 0, 13, 0, 99, 0]),
    blocks: new Uint32Array(9),
    mask: new Uint32Array([1, 1, 1, 1, 1, 1, 1, 1, 0]),
    features: new Float32Array([0, 1, 1, 2, 10, 11, 11, 12, 99])
  };
  // Farthest-first centers 0 and 12: first-step squared-error = 12. The next
  // centers 1 and 11 give error 4 and unchanged labels. See spatial-design-audit.oracle.py.
  for (const maximumIterations of [1, 2]) {
    const result = computeSpatialCrossValidationOnCPU(input, {
      foldCount: 2,
      design: {mode: 'environmental', featureCount: 1, standardize: false, maximumIterations}
    });
    const converged = maximumIterations === 2;
    expect(Array.from(result.diagnostics!)).toEqual([
      converged ? 0 : 3,
      maximumIterations,
      converged ? 4 : 12,
      NaN
    ]);
    expect(Array.from(result.folds)).toEqual(
      converged ? [0, 0, 0, 0, 1, 1, 1, 1, 0xffffffff] : new Array(9).fill(0xffffffff)
    );
    expect(Array.from(result.training)).toEqual(
      converged ? [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0] : new Array(18).fill(0)
    );
    expect(Array.from(result.testing)).toEqual(
      converged ? [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0] : new Array(18).fill(0)
    );
    expect(Array.from(result.counts)).toEqual(converged ? [4, 4, 0, 4, 4, 0] : [0, 0, 0, 0, 0, 0]);
    expect(Array.from(result.status)).toEqual(converged ? [0, 0] : [3, 3]);
  }
});

it('rejects nonpositive, fractional and nonfinite environmental iteration budgets', () => {
  for (const maximumIterations of [0, -1, 1.5, NaN, Infinity]) {
    expect(() =>
      computeSpatialCrossValidationOnCPU(
        {
          positions: new Float32Array([0, 0, 1, 0]),
          blocks: new Uint32Array(2),
          features: new Float32Array([0, 1])
        },
        {foldCount: 2, design: {mode: 'environmental', featureCount: 1, maximumIterations}}
      )
    ).toThrow('maximumIterations');
  }
});

it('audit-validation seeded block masks match exact Philox and SciPy buffer exclusion', () => {
  for (const shift of [0, 32]) {
    const actual = computeSpatialCrossValidationOnCPU(
      {
        positions: Float32Array.from(BLOCK_AUDIT_POSITIONS, value => value + shift),
        blocks: Uint32Array.from(BLOCK_AUDIT_BLOCKS),
        mask: Uint32Array.from(BLOCK_AUDIT_MASK)
      },
      {foldCount: 4, seed: 1234, bufferDistance: 1}
    );
    for (const [port, expected] of Object.entries(BLOCK_AUDIT_EXPECTED))
      expect(Array.from(actual[port as keyof typeof BLOCK_AUDIT_EXPECTED]), port).toEqual(expected);
  }
});

it.each([
  'blocks',
  'groups',
  'environmental',
  'nndm',
  'knndm'
] as const)('captures the %s fold design and all graph/context maps once', mode => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let nextView = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `fold-capture-${nextView++}`, format, length);
  try {
    const foldCount = mode === 'nndm' ? 8 : 2;
    const designValues: GPUSpatialValidationDesign | undefined =
      mode === 'blocks'
        ? undefined
        : mode === 'groups'
          ? {mode, diagnostics: view('float32', 4)}
          : mode === 'environmental'
            ? {
                mode,
                features: view('float32', 8),
                featureCount: 1,
                standardize: false,
                maximumIterations: 2,
                diagnostics: view('float32', 4)
              }
            : mode === 'nndm'
              ? {
                  mode,
                  predictionPositions: view('float32x2', 2),
                  maximumDistance: 4,
                  minimumTrainingFraction: 0.25,
                  diagnostics: view('float32', 12)
                }
              : {
                  mode,
                  predictionPositions: view('float32x2', 2),
                  candidates: view('uint32', 16),
                  candidateCount: 2,
                  maximumFoldFraction: 0.9,
                  diagnostics: view('float32', 4)
                };
    const design = designValues ? watchConstructionFields(designValues, 'design') : undefined;
    const origin: [number, number] = [-73, 40];
    const frameValues = {id: 'capture-frame', kind: 'local-azimuthal-equidistant' as const, origin};
    const frame = watchConstructionFields(frameValues, 'frame');
    const contextValues = {
      coordinateSpace: 'planar' as const,
      metric: 'native' as const,
      frame: frame.props
    };
    const context = watchConstructionFields(contextValues, 'spatialContext');
    const outputValues = {
      folds: view('uint32', 8),
      training: view('uint32', foldCount * 8),
      testing: view('uint32', foldCount * 8),
      counts: view('uint32', foldCount * 3),
      status: view('uint32', foldCount)
    };
    const output = watchConstructionFields(outputValues, 'output');
    const values: GPUSpatialCrossValidationProps = {
      id: 'capture-folds',
      spatialContext: context.props,
      positions: view('float32x2', 8),
      blocks: view('uint32', 8),
      mask: view('uint32', 8),
      design: design?.props,
      foldCount,
      seed: 1234,
      bufferDistance: 1,
      output: output.props
    };
    const watched = watchConstructionFields(values, 'props');
    const contributor = new GPUSpatialCrossValidation(watched.props);
    const watchers = [watched, context, frame, output, ...(design ? [design] : [])];
    for (const watcher of watchers) watcher.expectReadOnce();
    values.foldCount = 64;
    values.bufferDistance = -1;
    outputValues.folds = view('uint32', 1);
    origin[0] = 100;
    if (designValues) designValues.diagnostics = view('float32', 1);
    expect(contributor.props.foldCount).toBe(foldCount);
    expect(contributor.spatialContext.frame!.origin).toEqual([-73, 40]);
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
    for (const watcher of watchers) watcher.expectReadOnce();
  } finally {
    device.destroy();
  }
});

it.each([
  'cpu',
  'gpu'
] as const)('captures %s fold operation fields before lowering/execution and retains the SciPy partition', async form => {
  const designValues = {
    mode: 'environmental' as const,
    features: 'features',
    featureCount: 1,
    standardize: false,
    maximumIterations: 2,
    diagnostics: 'diagnostics'
  };
  const design = watchConstructionFields(designValues, 'design');
  const outputValues = {
    folds: 'folds',
    training: 'training',
    testing: 'testing',
    counts: 'counts',
    status: 'status'
  };
  const output = watchConstructionFields(outputValues, 'output');
  const values: SpatialCrossValidationOperationProps = {
    id: 'capture-operation',
    rowCount: 8,
    positions: 'positions',
    blocks: 'blocks',
    mask: 'mask',
    foldCount: 2,
    seed: 0,
    bufferDistance: 0,
    design: design.props,
    output: output.props
  };
  const watched = watchConstructionFields(values, 'props');
  const operation =
    form === 'cpu'
      ? new CPUSpatialCrossValidationOperation(watched.props)
      : new GPUSpatialCrossValidationOperation(watched.props);
  for (const watcher of [watched, design, output]) watcher.expectReadOnce();
  values.rowCount = 1;
  values.bufferDistance = -1;
  values.mask = undefined;
  designValues.featureCount = 2;
  designValues.maximumIterations = 1;
  outputValues.folds = 'changed';
  try {
    if (operation instanceof CPUSpatialCrossValidationOperation) {
      const actual = await operation.execute({
        inputs: createSamplingCPUInputs({
          positions: {
            format: 'float32x2',
            values: Float32Array.from([0, 0, 1, 0, 2, 0, 3, 0, 10, 0, 11, 0, 12, 0, 13, 0])
          },
          blocks: {format: 'uint32', values: new Uint32Array(8)},
          mask: {format: 'uint32', values: new Uint32Array(8).fill(1)},
          features: {format: 'float32', values: Float32Array.from([0, 1, 1, 2, 10, 11, 11, 12])}
        })
      });
      // SciPy vq oracle in spatial-design-audit.oracle.py: labels 00001111, SSE=4.
      expect(actual).toEqual({
        folds: Uint32Array.from([0, 0, 0, 0, 1, 1, 1, 1]),
        training: Uint32Array.from([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]),
        testing: Uint32Array.from([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]),
        counts: Uint32Array.from([4, 4, 0, 4, 4, 0]),
        status: Uint32Array.from([0, 0]),
        diagnostics: Float32Array.from([0, 2, 4, NaN])
      });
    } else {
      const device = createNullWebGPUDevice();
      try {
        const graph = new GPUCommandGraph(device);
        const views = {
          positions: createTransientView(graph, 'positions', 'float32x2', 8),
          blocks: createTransientView(graph, 'blocks', 'uint32', 8),
          mask: createTransientView(graph, 'mask', 'uint32', 8),
          features: createTransientView(graph, 'features', 'float32', 8),
          ...Object.fromEntries(
            Object.entries(operation.outputs).map(([port, shape]) => [
              port,
              createTransientView(graph, port, shape.format, shape.length)
            ])
          )
        };
        const contributor = operation.createContributor(
          port => views[port as keyof typeof views] as never
        ) as GPUSpatialCrossValidation;
        expect(contributor.props.bufferDistance).toBe(0);
        expect(contributor.props.mask).toBe(views.mask);
        expect(contributor.props.design).toMatchObject({
          mode: 'environmental',
          featureCount: 1,
          maximumIterations: 2
        });
        expect(() => contributor.getCommandNodes(graph)).not.toThrow();
      } finally {
        device.destroy();
      }
    }
    for (const watcher of [watched, design, output]) watcher.expectReadOnce();
  } finally {
    if (operation instanceof GPUSpatialCrossValidationOperation) operation.destroy();
  }
});

it('reads pure CPU fold options and environmental getters once before computing the SciPy oracle', () => {
  const design = watchConstructionFields(
    {mode: 'environmental' as const, featureCount: 1, standardize: false, maximumIterations: 2},
    'design'
  );
  const options = watchConstructionFields(
    {id: 'pure-fold-capture', foldCount: 2, seed: 0, bufferDistance: 0, design: design.props},
    'options'
  );
  const actual = computeSpatialCrossValidationOnCPU(
    {
      positions: Float32Array.from([0, 0, 1, 0, 2, 0, 3, 0, 10, 0, 11, 0, 12, 0, 13, 0]),
      blocks: new Uint32Array(8),
      features: Float32Array.from([0, 1, 1, 2, 10, 11, 11, 12])
    },
    options.props
  );
  options.expectReadOnce();
  design.expectReadOnce();
  expect(Array.from(actual.folds)).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
  expect(Array.from(actual.diagnostics!)).toEqual([0, 2, 4, NaN]);
});
