// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  getWGSLKernelNodeRecord,
  createWGSLKernelNode,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUSeeminglyUnrelatedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression';
import {GPUSpatialPanelRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-regression';
import {computeSeeminglyUnrelatedRegressionOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-seemingly-unrelated-regression-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

function createScene(rows: number, iteration?: {maximumIterations: number; tolerance: number}) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `review-${serial++}`, format, length);
  const predictors = view('float32', rows * 2),
    responses = view('float32', rows * 2);
  const output = {
    coefficients: view('float32', 4),
    covariance: view('float32', 16),
    residualCovariance: view('float32', 4),
    residuals: view('float32', rows * 2),
    status: view('uint32', 1),
    iteration: {status: {iterationCount: view('uint32', 1)}}
  };
  const parameters = view('float32', 2);
  const model = new GPUSeeminglyUnrelatedRegression({
    id: 'review',
    rowCount: rows,
    predictorCount: 1,
    equationCount: 2,
    iteration,
    predictors,
    responses,
    spatial: {
      weights: {
        offsets: view('uint32', rows + 1),
        neighbors: view('uint32', rows),
        weights: view('float32', rows)
      },
      output: {parameters, status: view('uint32', 2)}
    },
    output
  });
  return {
    device,
    graph,
    view,
    model,
    predictors,
    responses,
    parameters,
    output,
    nodes: model.getCommandNodes(graph)
  };
}

it('B1 restores a representable structural intercept before adding its small centered part', () => {
  const scene = createScene(8);
  try {
    const node = scene.nodes.find(candidate => candidate.id === 'review-unfilter-coefficients')!;
    const record = getWGSLKernelNodeRecord(scene.graph, node.id)!;
    const memory = createCPUGraphMemory();
    const binding = (name: string) => record.bindings.find(value => value.name === name)!.view;
    memory.write(
      scene.predictors,
      Float32Array.from([
        2 ** 24,
        2 ** 24,
        2 ** 24,
        2 ** 24,
        2 ** 24 + 4,
        2 ** 24 + 4,
        2 ** 24 + 4,
        2 ** 24 + 4,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7
      ])
    );
    memory.write(
      scene.responses,
      Float32Array.from([2 ** 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    );
    memory.write(scene.parameters, Float32Array.of(0, 0));
    memory.write(binding('sourceCoefficients'), Float32Array.of(1, 1, 0, 1));
    executeCommandNodesOnCPU([node], memory);
    expect(Array.from(memory.readFloat32(scene.output.coefficients))).toEqual([1, 1, 0, 1]);
  } finally {
    scene.device.destroy();
  }
});

it('forms the spatial GLS correction after removing exact equation-wise OLS slopes', () => {
  const scene = createScene(8);
  try {
    const node = scene.nodes.find(candidate => candidate.id === 'review-filtered-system-tile-gls')!;
    const record = getWGSLKernelNodeRecord(scene.graph, node.id)!;
    const binding = (name: string) => record.bindings.find(value => value.name === name)!.view;
    const memory = createCPUGraphMemory();
    const design = [0, 0, 0, 0, 4, 4, 4, 4];
    const errors = [-1, 1, 1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1];
    memory.write(binding('predictors'), Float32Array.from([...design, ...design]));
    memory.write(
      binding('responses'),
      Float32Array.from(
        errors,
        (error, index) => error + 1 + design[index % 8] * (index < 8 ? 1 : 2)
      )
    );
    memory.write(binding('initialResiduals'), Float32Array.from(errors));
    const constants = (name: string) =>
      Number(record.declarations!.match(new RegExp(`const ${name}: u32 = (\\d+)u`))![1]);
    const scratch = new Float32Array(binding('scratch').length);
    scratch.set([2, 2], constants('MEANS_START'));
    scratch.set([3, 5], constants('RESPONSES_START'));
    scratch.set([1, 2], constants('INITIAL_START'));
    scratch.set([1, -0.5, -0.5, 1], constants('PRECISION_START'));
    memory.write(binding('scratch'), scratch);
    executeCommandNodesOnCPU([node], memory);
    // Both error vectors are orthogonal to both equations' columns. The right-hand
    // side for a change in any coefficient is exactly zero for this precision matrix.
    const output = memory.readFloat32(binding('scratch'));
    const partial = constants('PARTIAL_START');
    const right = Array.from({length: 4}, (_, row) => output[partial + 5 * row + 4]);
    expect(right).toEqual([0, 0, 0, 0]);
  } finally {
    scene.device.destroy();
  }
});

it('preserves the small projected variance moment of spatial SUR', () => {
  const scene = createScene(5);
  try {
    const memory = createCPUGraphMemory();
    const find = (suffix: string) => scene.nodes.find(node => node.id === `review-gm-0-${suffix}`)!;
    const findBinding = (suffix: string, name: string) =>
      getWGSLKernelNodeRecord(scene.graph, find(suffix).id)!.bindings.find(
        value => value.name === name
      )!.view;
    const lag = find('residual-lag');
    const tile = find('moments-tiles');
    // One successor per row. Perturb one unit weight by 2^-23, the adjacent f32,
    // so the low word survives in W'W and (Wu)'Wu even after their high words agree.
    const weights = [1, 1, 1, 1, 1 + 2 ** -23];
    const errors = [0.25, -0.25, 0.25, -0.25, 0];
    memory.write(findBinding('residual-lag', 'weights'), Float32Array.from(weights));
    memory.write(findBinding('residual-lag', 'offsets'), Uint32Array.of(0, 1, 2, 3, 4, 5));
    memory.write(findBinding('residual-lag', 'neighbors'), Uint32Array.of(1, 2, 3, 4, 0));
    memory.write(findBinding('residual-lag', 'residual'), Float32Array.from(errors));
    executeCommandNodesOnCPU([lag, tile], memory);
    const words = memory.readFloat32(findBinding('moments-tiles', 'partial'));
    const moment = (slot: number) => words[2 * slot] + words[2 * slot + 1];
    const lagSquares = 0.1875 + (weights[4] / 4) ** 2;
    const weightSquares = 4 + weights[4] ** 2;
    expect(Math.abs(moment(2) - lagSquares)).toBeLessThan(1e-14);
    expect(Math.abs(moment(6) - weightSquares)).toBeLessThan(1e-13);
    const expected = lagSquares - (0.25 * weightSquares) / 5;
    expect(Math.abs(moment(2) - (0.25 * moment(6)) / 5 - expected)).toBeLessThan(1e-14);

    const solve = find('lambda-solve');
    const declaration = getWGSLKernelNodeRecord(scene.graph, solve.id)!.declarations!;
    const slot = (name: string) =>
      Number(declaration.match(new RegExp(`const ${name}: u32 = (\\d+)u`))![1]);
    const workspaceView = findBinding('lambda-solve', 'workspace');
    const workspace = new Float32Array(workspaceView.length);
    workspace[slot('W_BOUND')] = 0.95;
    memory.write(workspaceView, workspace);
    // Synthesize two nearly proportional variance moments. After rotating away their
    // nuisance direction, the residual is delta/sqrt(2) with derivative -0.5/sqrt(2).
    // The remaining score has derivative -1, so lambda is 2*delta/9 + O(delta^2).
    const delta = 2 ** -25;
    memory.write(
      findBinding('lambda-solve', 'moments'),
      Float32Array.of(1, 0, 0, 0, 1, delta, 0, 0, 0.25, 0, 1, 0, 5, 0)
    );
    executeCommandNodesOnCPU([solve], memory);
    expect(
      Math.abs(memory.readFloat32(workspaceView)[slot('W_LAMBDA')] - (2 * delta) / 9)
    ).toBeLessThan(1e-12);
  } finally {
    scene.device.destroy();
  }
});

it('B2 compares two-round spatial SUR in original filtered predictor coordinates', () => {
  const rows = 12;
  const predictors = Float32Array.from(
    Array.from({length: 24}, (_, index) => {
      const row = index % rows;
      return index < rows ? 100 + ((7 * row) % 11) : 300 + ((row * row + 3 * row) % 13);
    })
  );
  const responses = Float32Array.from(
    Array.from({length: 24}, (_, index) => {
      const row = index % rows;
      return index < rows
        ? 2 + ((5 * row + 1) % 9)
        : -2 + ((5 * row + 3) % 9) + ((3 * row + 1) % 4);
    })
  );
  const weights = {
    offsets: Uint32Array.from({length: 13}, (_, row) => row),
    neighbors: Uint32Array.from({length: 12}, (_, row) => (row + 1) % rows),
    weights: new Float32Array(12).fill(1)
  };
  const run = (maximumIterations: number, centered = false) =>
    computeSeeminglyUnrelatedRegressionOnCPU(
      {
        predictors: centered
          ? predictors.map((value, index) => value - predictors[Math.floor(index / rows) * rows])
          : predictors,
        responses,
        spatial: {weights}
      },
      {
        rowCount: rows,
        equationCount: 2,
        predictorCount: 1,
        iteration: {maximumIterations, tolerance: 0.02}
      }
    );
  const first = run(1),
    second = run(2),
    shifted = run(2, true);
  expect(Array.from(second.status)).toEqual([4]);
  expect(Array.from(second.iterationCount)).toEqual([2]);
  expect(Array.from(shifted.status)).toEqual([0]);
  const scene = createScene(rows, {maximumIterations: 2, tolerance: 0.02});
  try {
    const capture = scene.nodes.find(node => node.id.endsWith('iteration-2-capture-convergence'));
    const finish = scene.nodes.find(node => node.id.endsWith('iteration-2-finish-convergence'));
    expect(capture).toBeDefined();
    expect(finish).toBeDefined();
    const record = getWGSLKernelNodeRecord(scene.graph, finish!.id)!;
    const binding = (name: string) => record.bindings.find(value => value.name === name)!.view;
    const memory = createCPUGraphMemory();
    memory.write(scene.predictors, predictors);
    memory.write(scene.parameters, first.spatial!.parameters);
    memory.write(binding('parameters'), Float32Array.of(0.02));
    // Inject the independently computed CPU iterates into the centered GPU fit. The GPU
    // comparison must undo this coordinate shift before applying its stopping threshold.
    const filteredCentered = (result: typeof first) =>
      Float32Array.from(result.coefficients, (value, index) =>
        index % 2
          ? value
          : (value -
              responses[(index / 2) * rows] +
              predictors[(index / 2) * rows] * result.coefficients[index + 1]) *
            (1 - result.spatial!.parameters[index / 2])
      );
    const scratch = new Float32Array(binding('scratch').length);
    const iterationSlot = Number(
      record.declarations!.match(/const ITERATION_COUNT: u32 = (\d+)u/)![1]
    );
    scratch[iterationSlot] = 2;
    memory.write(binding('scratch'), scratch);
    const executeUngated = (node: NonNullable<typeof finish>) => {
      const original = getWGSLKernelNodeRecord(scene.graph, node.id)!;
      executeCommandNodesOnCPU(
        [
          createWGSLKernelNode(scene.graph, {
            id: `${node.id}-isolated`,
            operation: original.operation,
            bindings: [...original.bindings],
            invocationCount: original.invocationCount,
            workgroupSize: original.workgroupSize[0],
            declarations: original.declarations,
            body: original.body
          })
        ],
        memory
      );
    };
    memory.write(binding('coefficients'), filteredCentered(first));
    executeUngated(capture!);
    memory.write(binding('coefficients'), filteredCentered(second));
    memory.write(binding('status'), Uint32Array.of(4));
    executeUngated(finish!);
    expect(Array.from(memory.readUint32(binding('status')))).toEqual([4]);
    expect(memory.readFloat32(binding('scratch'))[iterationSlot]).toBe(2);
  } finally {
    scene.device.destroy();
  }
});

it('B4 folds required-count-only loss against entity capacity and clears it on recovery', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `count-${serial++}`, format, length);
    const required = view('uint32', 1),
      offsets = view('uint32', 5);
    const nodes = new GPUSpatialPanelRegression({
      id: 'count',
      periodCount: 3,
      predictorCount: 1,
      predictors: view('float32', 12),
      response: view('float32', 12),
      weights: {
        offsets,
        neighbors: view('uint32', 4),
        weights: view('float32', 4),
        status: {requiredCount: required}
      },
      output: {
        table: view('float32', 12),
        summary: view('float32', 16),
        status: {code: view('uint32', 1)}
      }
    }).getCommandNodes(graph);
    const folds = nodes.filter(node => node.id.startsWith('count-entity-weights-loss'));
    expect(folds.length).toBeGreaterThan(0);
    const loss = getWGSLKernelNodeRecord(graph, folds[folds.length - 1].id)!.bindings.find(
      binding => binding.name === 'loss'
    )!.view;
    const memory = createCPUGraphMemory();
    memory.write(offsets, Uint32Array.of(0, 1, 2, 3, 4));
    for (const [requiredCount, expectedLoss] of [
      [4, 0],
      [5, 1],
      [4, 0]
    ]) {
      memory.write(required, Uint32Array.of(requiredCount));
      executeCommandNodesOnCPU(folds, memory);
      expect(Array.from(memory.readUint32(loss))).toEqual([expectedLoss]);
    }
  } finally {
    device.destroy();
  }
});

it('S3 transforms nonzero cross-equation covariance blocks on both sides', () => {
  const scene = createScene(8);
  try {
    const node = scene.nodes.find(candidate => candidate.id === 'review-unfilter-coefficients')!;
    const record = getWGSLKernelNodeRecord(scene.graph, node.id)!;
    const binding = (name: string) => record.bindings.find(value => value.name === name)!.view;
    const memory = createCPUGraphMemory();
    const predictors = new Float32Array(16);
    predictors[0] = 2;
    predictors[8] = 3;
    memory.write(scene.predictors, predictors);
    memory.write(scene.parameters, Float32Array.of(0.5, 0.25));
    // V=I+v*v', v=[1,2,3,4]. With A1=[[2,-2],[0,1]] and
    // A2=[[4/3,-3],[0,1]], A*v=[-2,2,-8,4]. Hence A V A'=A A'+(Av)(Av)'.
    const vector = [1, 2, 3, 4];
    memory.write(
      binding('sourceCovariance'),
      Float32Array.from(
        {length: 16},
        (_, index) =>
          vector[Math.floor(index / 4)] * vector[index % 4] +
          (Math.floor(index / 4) === index % 4 ? 1 : 0)
      )
    );
    executeCommandNodesOnCPU([node], memory);
    const matrix = memory.readFloat32(scene.output.covariance);
    expect(matrix).toHaveLength(16);
    const expected = [12, -6, 16, -8, -6, 5, -16, 8, 16, -16, 64 + 97 / 9, -35, -8, 8, -35, 17];
    expected.forEach((value, index) => {
      expect(Number.isFinite(matrix[index])).toBe(true);
      expect(Math.abs(matrix[index] - value)).toBeLessThan(1e-4);
    });
  } finally {
    scene.device.destroy();
  }
});

import {GPUSpatialPanelErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood';

for (const model of ['panel', 'error'] as const)
  it(`W5-35 captured certificate guards dependent values and recovers ${model}`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device),
        memory = createCPUGraphMemory();
      recordWGSLKernelNodes(graph);
      let serial = 0;
      const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
        createTransientView(graph, `complete-${serial++}`, format, length);
      const complete = view('uint32', 1),
        tests = view('float32', 6),
        diagnosticStatus = view('uint32', 1);
      const props = {
        id: 'complete',
        periodCount: 4,
        predictorCount: 1,
        predictors: view('float32', 16),
        response: view('float32', 16),
        weights: {
          offsets: view('uint32', 5),
          neighbors: view('uint32', 8),
          weights: view('float32', 8),
          status: {complete}
        },
        output: {
          table: view('float32', 12),
          summary: view('float32', 22),
          residuals: view('float32', 16),
          status: {code: view('uint32', 1)}
        }
      };
      const contributor =
        model === 'panel'
          ? new GPUSpatialPanelRegression(props)
          : new GPUSpatialPanelErrorMaximumLikelihood({
              ...props,
              maximumPower: 3,
              diagnostics: {tests, status: diagnosticStatus}
            });
      expect(contributor.props.weights.status!.complete).toBe(complete);
      props.weights.status.complete = view('uint32', 1);
      const nodes = contributor
        .getCommandNodes(graph)
        .filter(
          node =>
            getWGSLKernelNodeRecord(graph, node.id)?.operation ===
              'RegressionWeightsCompleteness' || node.id.endsWith('-diagnostic-loss-final')
        );
      expect(nodes.some(node => node.id.includes('-weights-complete-'))).toBe(true);
      for (const certificate of [1, 0, 1]) {
        memory.write(complete, Uint32Array.of(certificate));
        for (const output of [
          props.output.table,
          props.output.summary,
          props.output.residuals,
          tests
        ])
          memory.write(output, new Float32Array(output.length).fill(7));
        memory.write(props.output.status.code, Uint32Array.of(0));
        memory.write(diagnosticStatus, Uint32Array.of(0));
        executeCommandNodesOnCPU(nodes, memory);
        expect(Array.from(memory.readUint32(props.output.status.code))).toEqual([
          certificate ? 0 : 3
        ]);
        for (const output of [props.output.table, props.output.summary, props.output.residuals]) {
          const values = Array.from(memory.readFloat32(output)).slice(
            0,
            model === 'panel' && output === props.output.summary ? 16 : output.length
          );
          expect(values.every(certificate ? value => value === 7 : Number.isNaN)).toBe(true);
        }
        if (model === 'error') {
          expect(Array.from(memory.readUint32(diagnosticStatus))).toEqual([certificate ? 0 : 1]);
          expect(
            Array.from(memory.readFloat32(tests)).every(
              certificate ? value => value === 7 : Number.isNaN
            )
          ).toBe(true);
        }
      }
    } finally {
      device.destroy();
    }
  });

it('F13 direct SUR captures nested iteration status with the existing count writer', () => {
  const scene = createScene(8);
  try {
    const count = scene.output.iteration.status.iterationCount;
    const iteration = {status: {iterationCount: count}};
    const props = {
      ...scene.model.props,
      id: 'nested-count',
      output: {...scene.output, iteration}
    };
    const model = new GPUSeeminglyUnrelatedRegression(props);
    expect(model.props.output.iteration?.status.iterationCount).toBe(count);
    iteration.status.iterationCount = scene.view('uint32', 1);
    expect(model.props.output.iteration?.status.iterationCount).toBe(count);
    expect(
      model
        .getCommandNodes(scene.graph)
        .some(node =>
          node.resources?.some(
            resource =>
              'buffer' in resource &&
              resource.buffer === count &&
              resource.usage === 'storage-read-write'
          )
        )
    ).toBe(true);
    expect(
      () =>
        new GPUSeeminglyUnrelatedRegression({
          ...props,
          output: {...props.output, iteration: {status: {iterationCount: scene.view('uint32', 2)}}}
        })
    ).toThrow(/shape mismatch/);
  } finally {
    scene.device.destroy();
  }
});
