// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSeeminglyUnrelatedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression';
import {getOrdinarySURScratchLength} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-sur-nodes';
import {getSpatialSURAllocationLengths} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-sur-nodes';
import {
  auditWGSLModule,
  getWGSLModuleUsage
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import fixtures from './iterated-sur-fixtures.json';

it('R2 covariance preserves resolved noise small relative to the fitted signal', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `small-noise-${serial++}`, format, length);
    const predictors = view('float32', 16),
      responses = view('float32', 16);
    const output = {
      coefficients: view('float32', 4),
      covariance: view('float32', 16),
      residualCovariance: view('float32', 4),
      residuals: view('float32', 16),
      status: view('uint32', 1)
    };
    const memory = createCPUGraphMemory();
    const amplitude = 2 ** -16;
    memory.write(
      predictors,
      Float32Array.from(
        {length: 16},
        (_, index) => 2 * ((index < 8 ? index : Math.floor(index / 2)) % 2) - 1
      )
    );
    memory.write(
      responses,
      Float32Array.from({length: 16}, (_, index) => {
        const row = index % 8,
          first = 2 * (row % 2) - 1,
          second = 2 * (Math.floor(row / 2) % 2) - 1,
          third = 2 * (Math.floor(row / 4) % 2) - 1;
        return index < 8
          ? 1 + first + amplitude * (second + first * second)
          : 2 + second + amplitude * (0.5 * first + first * second + third);
      })
    );
    const nodes = new GPUSeeminglyUnrelatedRegression({
      equationCount: 2,
      predictorCount: 1,
      rowCount: 8,
      predictors,
      responses,
      output
    }).getCommandNodes(graph);
    const covarianceNode = nodes.find(node => node.id.endsWith('-tile-error-covariance'))!;
    executeCommandNodesOnCPU([nodes[0], nodes[1]], memory);
    const scratchUse = covarianceNode.resources!.find(
      resource =>
        'buffer' in resource &&
        'format' in resource.buffer &&
        resource.buffer.buffer.id.endsWith('-statistics')
    )!;
    if (!('buffer' in scratchUse) || !('format' in scratchUse.buffer))
      throw new Error('missing scratch');
    const scratch = memory.readFloat32(scratchUse.buffer);
    // Inject analytically exact centered means and OLS slopes from the orthogonal binary
    // contrasts, isolating the floor rule from tile-mean rounding and unsupported dense solves.
    scratch[1] = 0;
    scratch[2] = 0;
    scratch[3] = 1;
    scratch[4] = 2;
    scratch[5] = 1;
    scratch[6] = 1;
    memory.write(scratchUse.buffer, scratch);
    executeCommandNodesOnCPU([covarianceNode], memory);
    expect(Array.from(memory.readFloat32(scratchUse.buffer).slice(17, 21))).toEqual(
      [2, 1, 1, 2.25].map(value => value * amplitude ** 2)
    );
  } finally {
    device.destroy();
  }
});

for (const iterated of [false, true]) {
  it(`preserves finite covariance when a residual product overflows: ${iterated ? 'iterated' : 'default'}`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
        createTransientView(graph, `overflow-${serial++}`, format, length);
      const predictors = view('float32', 16);
      const responses = view('float32', 16);
      const initialResiduals = view('float32', 16);
      const output = {
        coefficients: view('float32', 4),
        covariance: view('float32', 16),
        residualCovariance: view('float32', 4),
        residuals: view('float32', 16),
        status: view('uint32', 1)
      };
      const memory = createCPUGraphMemory();
      memory.write(
        predictors,
        Float32Array.from({length: 16}, (_, index) => (index % 2 ? 1 : -1))
      );
      memory.write(
        responses,
        Float32Array.from({length: 16}, (_, index) => (index % 2 ? 3 : 1))
      );
      const errors = new Float32Array(16);
      errors[0] = 2 ** 64;
      errors[9] = 2 ** 64;
      memory.write(initialResiduals, errors);
      const nodes = new GPUSeeminglyUnrelatedRegression({
        equationCount: 2,
        predictorCount: 1,
        rowCount: 8,
        predictors,
        responses,
        initialResiduals,
        iteration: iterated ? {maximumIterations: 1} : undefined,
        output
      }).getCommandNodes(graph);
      // Execute the actual means and covariance tile kernels, bypassing the interpreter's
      // unsupported Cholesky pointer-array assignments. Inspect their public resource view.
      const covarianceNode = nodes.find(node => node.id.endsWith('-tile-error-covariance'))!;
      executeCommandNodesOnCPU(
        [
          ...nodes.filter(node => node.type === 'copy'),
          nodes.find(node => node.id.endsWith('-tile-means'))!,
          nodes.find(node => node.id.endsWith('-merge-means'))!,
          covarianceNode
        ],
        memory
      );
      const scratchUse = covarianceNode.resources?.find(
        resource =>
          'buffer' in resource &&
          'format' in resource.buffer &&
          resource.buffer.buffer.id.endsWith('-statistics')
      );
      if (!scratchUse || !('buffer' in scratchUse) || !('format' in scratchUse.buffer))
        throw new Error('missing SUR statistics view');
      const partial = memory.readFloat32(scratchUse.buffer);
      // Scratch prefix = count 1 + means 2+2 + slopes 2 + precision 4 + scale 1
      // + centered beta 4 + zero 1 = 17. Distinct impulses give EE'/8=diag(2^125).
      expect(Array.from(partial.slice(17, 21))).toEqual([2 ** 125, 0, 0, 2 ** 125]);
    } finally {
      device.destroy();
    }
  });
}

it('SUR allocation helpers match independently counted three-equation storage', () => {
  // e=3, p=2, width=3, system=9, one row tile. Scratch: count 1 + predictor
  // means 6 + response means 3 + OLS slopes 6 + precision 9 + scale 1 +
  // centered beta 9 + zero 1 + partial normal/right 9*(9+1) + controls 2 = 128.
  expect(getOrdinarySURScratchLength(3, 2, 40)).toBe(128);
  const lengths = getSpatialSURAllocationLengths(40, 3, 2);
  expect(
    Object.fromEntries(Object.entries(lengths).filter(([name]) => name.startsWith('SUR ')))
  ).toEqual({
    // 40 rows × 3 equations × 2 predictors, 3 × (2+1) prior coefficients, one tolerance.
    'SUR centered predictors': 240,
    'SUR convergence previous coefficients': 9,
    'SUR convergence disabled tolerance': 1,
    'SUR filtered predictors': 240,
    'SUR filtered responses/residuals': 120,
    'SUR centered responses': 120,
    'SUR filtered coefficients': 9,
    'SUR filtered covariance': 81,
    'SUR unfiltered equation': 160,
    'SUR statistics scratch': 128,
    'SUR weight partials': 1,
    'SUR GM table': 16,
    'SUR GM summary': 13,
    'SUR OLS summary': 22,
    // Workspace: two predictor means, four scalar controls, three coefficients, and
    // 18 remaining controls and aggregates, including the two candidate rejection counts.
    'SUR GM workspace': 27,
    'SUR GM mean partials': 3,
    // Per equation, seven Kelejian–Prucha moments with two words each, one tile,
    // and a two-word lagged residual for each of the 40 rows.
    'SUR GM moment partials': 14,
    'SUR GM moments': 14,
    'SUR GM residual lag': 80,
    'SUR GM filtered partials': 13,
    'SUR GM filtered moments': 13,
    'SUR GM residual partials': 5,
    'SUR GM residual totals': 5
  });
});

it('iterated SUR kernels fit shader budgets and the declared command count at maximum shapes', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const [equationCount, predictorCount, spatial] of [
      [2, 7, false],
      [3, 4, false],
      [4, 3, false],
      [2, 2, true],
      [4, 3, true]
    ] as const) {
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
        createTransientView(graph, `iterated-sur-${serial++}`, format, length);
      const size = equationCount * (predictorCount + 1);
      const contributor = new GPUSeeminglyUnrelatedRegression({
        equationCount,
        predictorCount,
        rowCount: 40,
        predictors: view('float32', equationCount * predictorCount * 40),
        responses: view('float32', equationCount * 40),
        initialResiduals: view('float32', equationCount * 40),
        mask: view('uint32', 40),
        ...(spatial
          ? {
              mask: undefined,
              initialResiduals: undefined,
              spatial: {
                weights: {
                  offsets: view('uint32', 41),
                  neighbors: view('uint32', 80),
                  weights: view('float32', 80)
                },
                output: {
                  parameters: view('float32', equationCount),
                  status: view('uint32', equationCount)
                }
              }
            }
          : {}),
        iteration: {maximumIterations: 9},
        output: {
          coefficients: view('float32', size),
          covariance: view('float32', size * size),
          residualCovariance: view('float32', equationCount * equationCount),
          residuals: view('float32', equationCount * 40),
          status: view('uint32', 1),
          iteration: {status: {iterationCount: view('uint32', 1)}}
        }
      });
      const nodes = contributor.getCommandNodes(graph);
      if (spatial) expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      else expect(nodes.length).toBe(contributor.estimatedCommandNodeCount);
      for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
        const usage = getWGSLModuleUsage(kernel.source, kernel.constants);
        expect(usage.storageBuffers, kernel.id).toBeLessThanOrEqual(8);
        expect(usage.functionBytes, kernel.id).toBeLessThanOrEqual(8192);
        expect(usage.privateBytes, kernel.id).toBeLessThanOrEqual(8192);
      }
    }
  } finally {
    device.destroy();
  }
});

for (const failure of ['excluded', 'singular-design']) {
  it(`interpreted WGSL publishes the defined ${failure} outputs and disarms later rounds`, () => {
    const fixture = fixtures[0];
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
        createTransientView(graph, `interpreted-sur-${serial++}`, format, length);
      const predictors = view('float32', 240);
      const responses = view('float32', 120);
      const mask = view('uint32', 40);
      const output = {
        coefficients: view('float32', 9),
        covariance: view('float32', 81),
        residualCovariance: view('float32', 9),
        residuals: view('float32', 120),
        status: view('uint32', 1),
        iteration: {status: {iterationCount: view('uint32', 1)}}
      };
      const memory = createCPUGraphMemory();
      memory.write(
        predictors,
        failure === 'singular-design'
          ? new Float32Array(240)
          : Float32Array.from(fixture.predictors)
      );
      memory.write(responses, Float32Array.from(fixture.responses));
      memory.write(
        mask,
        failure === 'excluded' ? new Uint32Array(40) : Uint32Array.from(fixture.mask)
      );
      const contributor = new GPUSeeminglyUnrelatedRegression({
        equationCount: 3,
        predictorCount: 2,
        rowCount: 40,
        predictors,
        responses,
        mask,
        iteration: {maximumIterations: 3},
        output
      });
      executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
      expect(Array.from(memory.readUint32(output.status))).toEqual([
        failure === 'excluded' ? 1 : 2
      ]);
      expect(Array.from(memory.readUint32(output.iteration.status.iterationCount))).toEqual([0]);
      for (const name of ['coefficients', 'covariance', 'residualCovariance', 'residuals'] as const)
        expect(memory.readFloat32(output[name]).every(Number.isNaN), name).toBe(true);
      const afterFailure = Array.from(memory.workgroupCounts).filter(([name]) =>
        /iteration-[2-3]-/.test(name)
      );
      expect(afterFailure).toHaveLength(8);
      expect(afterFailure.every(([, count]) => count === 0)).toBe(true);
    } finally {
      device.destroy();
    }
  });
}
