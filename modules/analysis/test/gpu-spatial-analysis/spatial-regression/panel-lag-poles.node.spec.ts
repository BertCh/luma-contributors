// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUSpatialPanelLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  getPanelLagPoleAllocationLengths,
  hasPanelLagPoleOnCPU
} from '../../../src/gpu-spatial-analysis/spatial-regression/panel-lag-pole-nodes';
import {computeSpatialPanelLagMaximumLikelihoodOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-lag-maximum-likelihood-operation';
import {panelPoleScenes} from './panel-lag-poles.test-utils';

for (const predictorScale of [1, 2 ** -25])
  it(`W5-37 scale=${predictorScale} refuses a spatial pole with the rational slope 1/3`, () => {
    const scene = panelPoleScenes.find(value => value.name === 'weightedRationalPole')!;
    const result = computeSpatialPanelLagMaximumLikelihoodOnCPU(
      {
        predictors: Float32Array.from(scene.predictors, value => value * predictorScale),
        response: Float32Array.from(scene.response),
        weights: {
          offsets: Uint32Array.from(scene.weights.offsets),
          neighbors: Uint32Array.from(scene.weights.neighbors),
          weights: Float32Array.from(scene.weights.weights)
        }
      },
      {periodCount: 2, predictorCount: 1, effects: 'unit'}
    );
    expect(
      result.status[0],
      JSON.stringify({variance: result.summary[0], likelihood: result.summary[1]})
    ).toBe(2);
    for (const port of ['coefficients', 'covariance', 'summary', 'fitted', 'residuals'] as const)
      expect(Array.from(result[port]).every(Number.isNaN), port).toBe(true);
  });

it('CPU refuses the exact rational slope 1/3 before publishing a finite maximum', () => {
  const scene = panelPoleScenes.find(value => value.name === 'rationalSlopePole')!;
  const result = computeSpatialPanelLagMaximumLikelihoodOnCPU(
    {
      predictors: Float32Array.from(scene.predictors),
      response: Float32Array.from(scene.response),
      weights: {
        offsets: Uint32Array.from(scene.weights.offsets),
        neighbors: Uint32Array.from(scene.weights.neighbors),
        weights: Float32Array.from(scene.weights.weights)
      }
    },
    {periodCount: 2, predictorCount: 1, effects: 'unit'}
  );
  expect(
    result.status[0],
    JSON.stringify({status: result.status[0], summary: Array.from(result.summary)})
  ).toBe(2);
  for (const port of ['coefficients', 'covariance', 'summary', 'fitted', 'residuals'] as const)
    expect(Array.from(result[port]).every(Number.isNaN)).toBe(true);
});

it('preflights every pole-certificate scratch view', () => {
  expect(getPanelLagPoleAllocationLengths(8, 2)).toEqual({
    'pole candidates': 192,
    'pole partials': 48,
    'pole witnesses': 49
  });
});

for (const scene of panelPoleScenes)
  for (const effects of ['unit', 'random'] as const)
    for (const affine of [false, true])
      for (const perturbed of [false, true])
        it(`integer original-input pole proof ${scene.name}, ${effects}, affine=${affine}, one-ulp perturbation=${perturbed}`, () => {
          const device = createNullWebGPUDevice();
          try {
            const graph = new GPUCommandGraph(device),
              memory = createCPUGraphMemory();
            recordWGSLKernelNodes(graph);
            let serial = 0;
            const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
              createTransientView(graph, `certificate-${serial++}`, format, length);
            const input = {
              predictors: Float32Array.from(scene.predictors),
              response: Float32Array.from(scene.response),
              weights: {
                offsets: Uint32Array.from(scene.weights.offsets),
                neighbors: Uint32Array.from(scene.weights.neighbors),
                weights: Float32Array.from(scene.weights.weights)
              }
            };
            if (affine) {
              input.predictors = Float32Array.from(input.predictors, value => value + 8);
              input.response = Float32Array.from(
                input.response,
                (value, row) => 2 * value + 16 * (row % 4)
              );
            }
            if (perturbed) {
              const storage = new DataView(new ArrayBuffer(4));
              storage.setFloat32(0, input.response[0]);
              storage.setUint32(0, storage.getUint32(0) + (input.response[0] < 0 ? -1 : 1));
              input.response[0] = storage.getFloat32(0);
            }
            const props = {
              ...scene,
              effects,
              predictors: view('float32', input.predictors.length),
              response: view('float32', 8),
              weights: {
                offsets: view('uint32', 5),
                neighbors: view('uint32', input.weights.neighbors.length),
                weights: view('float32', input.weights.weights.length)
              },
              output: {
                coefficients: view('float32', scene.predictorCount + 3),
                covariance: view('float32', (scene.predictorCount + 2) ** 2),
                summary: view('float32', 6),
                status: view('uint32', 1),
                fitted: view('float32', 8),
                residuals: view('float32', 8)
              },
              hausman:
                effects === 'random'
                  ? {tests: view('float32', 3), status: view('uint32', 1)}
                  : undefined
            };
            memory.write(props.predictors, input.predictors);
            memory.write(props.response, input.response);
            memory.write(props.weights.offsets, input.weights.offsets);
            memory.write(props.weights.neighbors, input.weights.neighbors);
            memory.write(props.weights.weights, input.weights.weights);
            const approximate =
              scene.name === 'rationalSlopePole'
                ? [NaN, 1 / 3, 0, 0]
                : scene.name === 'weightedRationalPole'
                  ? [NaN, 1 / 3 + 2 ** -24, 0.5 - 2 ** -24, 0]
                  : scene.name === 'responseCopy'
                    ? [NaN, 1 + 2 ** -22, 2 ** -24, 2 ** -23, 0]
                    : [NaN, 1 + 2 ** -22, 0.5 - 2 ** -24, 0];
            if (affine)
              for (let column = 1; column <= scene.predictorCount; column++)
                approximate[column] *= 2;
            // The shared solve is not emulated: feed a rounded fit in the neighborhood of the
            // exact rational solution, then execute every actual proof/publication kernel.
            for (const [name, output] of Object.entries(props.output))
              memory.write(
                output,
                name === 'status' ? Uint32Array.of(0) : new Float32Array(output.length).fill(987)
              );
            memory.write(props.output.coefficients, Float32Array.from(approximate));
            if (props.hausman) {
              memory.write(props.hausman.tests, Float32Array.of(1, 3, 0.5));
              memory.write(props.hausman.status, Uint32Array.of(0));
            }
            const nodes = new GPUSpatialPanelLagMaximumLikelihood(props)
              .getCommandNodes(graph)
              .filter(node => node.id.includes('-pole-'));
            expect(nodes.length).toBe(
              (effects === 'random' ? 12 : 10) + (scene.predictorCount === 1 ? 2 : 0)
            );
            executeCommandNodesOnCPU(nodes, memory);
            expect(hasPanelLagPoleOnCPU(input, scene.predictorCount, approximate)).toBe(!perturbed);
            expect(Array.from(memory.readUint32(props.output.status))).toEqual([perturbed ? 0 : 2]);
            for (const [name, output] of Object.entries(props.output)) {
              if (name === 'status') continue;
              const values = memory.readFloat32(output);
              if (!perturbed) expect(Array.from(values).every(Number.isNaN), name).toBe(true);
              else if (name !== 'coefficients')
                expect(Array.from(values)).toEqual(new Array(output.length).fill(987));
            }
            if (props.hausman) {
              expect(Array.from(memory.readUint32(props.hausman.status))).toEqual([
                perturbed ? 0 : 1
              ]);
              if (!perturbed)
                expect(
                  Array.from(memory.readFloat32(props.hausman.tests)).every(Number.isNaN)
                ).toBe(true);
            }
            if (!perturbed) {
              const result = computeSpatialPanelLagMaximumLikelihoodOnCPU(input, {
                ...scene,
                effects,
                hausman: effects === 'random'
              });
              expect(Array.from(result.status)).toEqual([2]);
              for (const port of [
                'coefficients',
                'covariance',
                'summary',
                'fitted',
                'residuals'
              ] as const)
                expect(Array.from(result[port]).every(Number.isNaN)).toBe(true);
            }
          } finally {
            device.destroy();
          }
        });

it('tiled profile refuses an admissible exact zero-RSS minimum instead of its nearby finite search point', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device),
      memory = createCPUGraphMemory();
    recordWGSLKernelNodes(graph);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `pole-${serial++}`, format, length);
    const props = {
      effects: 'unit' as const,
      periodCount: 2,
      predictorCount: 1,
      eigenvalues: view('float32', 4),
      predictors: view('float32', 8),
      response: view('float32', 8),
      weights: {
        offsets: view('uint32', 5),
        neighbors: view('uint32', 4),
        weights: view('float32', 4)
      },
      output: {
        coefficients: view('float32', 4),
        covariance: view('float32', 9),
        summary: view('float32', 6),
        status: view('uint32', 1),
        fitted: view('float32', 8),
        residuals: view('float32', 8)
      }
    };
    const nodes = new GPUSpatialPanelLagMaximumLikelihood(props).getCommandNodes(graph);
    const node = nodes.find(node => node.id.endsWith('-profile-rho'))!;
    const record = getWGSLKernelNodeRecord(graph, node.id)!;
    const model = record.bindings.find(binding => binding.name === 'model')!.view;
    const values = new Float32Array(model.length);
    values[1] = 0.5;
    values[2] = 8;
    memory.write(model, values);
    memory.write(props.eigenvalues, Float32Array.of(-1, -1, 1, 1));
    // Independent closed form: RSS(rho)=8*(rho-.5)^2, minimum zero at admissible rho=.5.
    // The determinant is positive there. Gaussian log likelihood tends to +infinity as
    // sigma2 -> 0, so no finite ML estimate or covariance exists. Golden points miss .5.
    executeCommandNodesOnCPU([node], memory);
    const actual = memory.readFloat32(model);
    expect(actual[0]).toBe(2);
  } finally {
    device.destroy();
  }
});

it('CCK-03 retains the first pivot in the last tile through parallel minimum passes', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device),
      memory = createCPUGraphMemory();
    recordWGSLKernelNodes(graph);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `last-pivot-${serial++}`, format, length);
    const rows = 8256;
    const props = {
      periodCount: 2064,
      predictorCount: 1,
      effects: 'unit' as const,
      eigenvalues: view('float32', 4),
      predictors: view('float32', rows),
      response: view('float32', rows),
      weights: {
        offsets: view('uint32', 5),
        neighbors: view('uint32', 4),
        weights: view('float32', 4)
      },
      output: {
        coefficients: view('float32', 4),
        covariance: view('float32', 9),
        summary: view('float32', 6),
        status: view('uint32', 1),
        fitted: view('float32', rows),
        residuals: view('float32', rows)
      }
    };
    const nodes = new GPUSpatialPanelLagMaximumLikelihood(props)
      .getCommandNodes(graph)
      .filter(node => node.id.includes('-pole-proportional-pivot'));
    const predictors = new Float32Array(rows);
    predictors[rows - 1] = 1;
    memory.write(props.predictors, predictors);
    executeCommandNodesOnCPU(nodes, memory);
    const record = getWGSLKernelNodeRecord(graph, nodes[0].id)!;
    const pivots = record.bindings.find(binding => binding.name === 'pivots')!.view;
    expect(memory.readUint32(pivots)[129]).toBe(rows - 1);
    expect(getPanelLagPoleAllocationLengths(rows, 1)['pole pivot reduction']).toBe(65);
    expect(nodes.length).toBe(9);
    // Removing the final pivot restores the row-count sentinel, including empty tiles.
    memory.write(props.predictors, new Float32Array(rows));
    executeCommandNodesOnCPU(nodes, memory);
    expect(memory.readUint32(pivots)[129]).toBe(rows);
  } finally {
    device.destroy();
  }
});

for (const predictorScale of [1, 2 ** -25])
  it(`W5-37 production CPU retains nearby identifiable fit scale=${predictorScale}`, () => {
    const scene = panelPoleScenes.find(value => value.name === 'weightedRationalPole')!;
    const response = Float32Array.from(scene.response);
    new Uint32Array(response.buffer)[0]--;
    const result = computeSpatialPanelLagMaximumLikelihoodOnCPU(
      {
        predictors: Float32Array.from(scene.predictors, value => value * predictorScale),
        response,
        weights: {
          offsets: Uint32Array.from(scene.weights.offsets),
          neighbors: Uint32Array.from(scene.weights.neighbors),
          weights: Float32Array.from(scene.weights.weights)
        }
      },
      {periodCount: 2, predictorCount: 1, effects: 'unit'}
    );
    expect(result.status[0]).toBe(0);
    expect(Number.isFinite(result.summary[1])).toBe(true);
  });
