// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUContributorOperation} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {runOperationGraph, type OperationGraphInput} from '../../utils/mixed-execution-test-utils';
import {GPUSpatialRegressionDiagnosticsOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regression-diagnostics-operation';
import {GPUSpatialErrorGMOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm-operation';
import {GPUInstrumentalVariablesCovariance} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-instrumental-variables-covariance';
import {
  createBipartiteDiagnosticsInput,
  createWeightedConstantLagInput,
  createTranslatedAffineInput
} from './regression-numerics-w5-fixtures';

const weightNames = {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'};
function inputs(
  input:
    | ReturnType<typeof createBipartiteDiagnosticsInput>
    | ReturnType<typeof createTranslatedAffineInput>
) {
  const result: Record<string, OperationGraphInput> = {
    predictors: {format: 'float32' as const, values: input.predictors},
    response: {format: 'float32' as const, values: input.response},
    offsets: {format: 'uint32' as const, values: input.weights.offsets},
    neighbors: {format: 'uint32' as const, values: input.weights.neighbors},
    weights: {format: 'float32' as const, values: input.weights.weights}
  };
  if ('residuals' in input) result['residuals'] = {format: 'float32', values: input.residuals};
  return result;
}

for (const kind of ['bipartite', 'weighted'] as const)
  it(`W5-30/W5-31 GPU ${kind} projected degeneracy publishes unavailable inference`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input =
      kind === 'bipartite' ? createBipartiteDiagnosticsInput() : createWeightedConstantLagInput();
    const result = await runOperationGraph(device, {
      inputs: inputs(input),
      operations: [
        new GPUSpatialRegressionDiagnosticsOperation({
          rowCount: input.response.length,
          predictorCount: 1,
          weights: weightNames,
          predictors: 'predictors',
          response: 'response',
          residuals: 'residuals',
          output: {tests: 'tests', summary: 'summary', status: {code: 'code'}}
        })
      ],
      outputs: ['tests', 'summary', 'code']
    });
    expect(Array.from(result['code'])).toEqual([0]);
    if (kind === 'weighted') {
      expect(result['summary'][3]).toBe(35);
      for (const row of [2, 3, 4]) expect(result['tests'][3 * row]).toBeNaN();
    } else {
      expect(result['summary'][7]).toBe(0);
      for (const slot of [8, 9]) expect(result['summary'][slot]).toBeNaN();
      for (const slot of [15, 17]) expect(result['tests'][slot]).toBeNaN();
    }
  });

for (const estimator of ['homoskedastic', 'kelejian-prucha'] as const)
  for (const departure of [false, true])
    it(`W5-32 GPU ${estimator} translated rational affine departure=${departure}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const result = await runOperationGraph(device, {
        inputs: inputs(createTranslatedAffineInput(departure)),
        operations: [
          new GPUSpatialErrorGMOperation({
            rowCount: 12,
            predictorCount: 1,
            estimator,
            weights: weightNames,
            predictors: 'predictors',
            response: 'response',
            output: {
              table: 'table',
              summary: 'summary',
              residuals: 'residuals',
              status: {code: 'code'}
            }
          })
        ],
        outputs: ['code', 'table']
      });
      expect(Array.from(result['code'])).toEqual([departure ? 0 : 3]);
      if (!departure) expect(result['table'].every(Number.isNaN)).toBe(true);
    });

for (const position of [0, 524287])
  it(`W5-33 GPU rank retains lone outlier variation at row ${position}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rows = 524288;
    const column = new Float32Array(rows).fill(Math.fround(1.9793816));
    column[position] = 0;
    const result = await runOperationGraph(device, {
      inputs: {
        column: {format: 'float32', values: column},
        residuals: {format: 'float32', values: new Float32Array(rows)}
      },
      operations: [
        new GPUContributorOperation({
          id: 'outlier-rank',
          inputs: {column: 'column', residuals: 'residuals'},
          outputs: {
            code: {value: 'code', format: 'uint32', length: 1},
            covariance: {value: 'covariance', format: 'float32', length: 4},
            centeredSS: {value: 'centeredSS', format: 'float32', length: 1}
          },
          createContributor: getView => ({
            getCommandNodes: graph => {
              const contributor = new GPUInstrumentalVariablesCovariance({
                design: getView<'float32'>('column'),
                instruments: getView<'float32'>('column'),
                residuals: getView<'float32'>('residuals'),
                coefficientCount: 1,
                instrumentCount: 1,
                method: 'HC0',
                output: {
                  covariance: getView<'float32'>('covariance'),
                  status: {code: getView<'uint32'>('code')}
                }
              });
              const nodes = contributor.getCommandNodes(graph);
              const workspace = nodes
                .flatMap(node => node.resources ?? [])
                .flatMap(resource =>
                  'buffer' in resource ? [resource.buffer as GraphDataView<'float32'>] : []
                )
                .find(view => view.buffer.id.endsWith('-workspace'))!;
              return [
                ...nodes,
                createWGSLKernelNode(graph, {
                  id: 'read-centered-ss',
                  operation: 'TestReadback',
                  invocationCount: 1,
                  bindings: [
                    {name: 'workspace', view: workspace, type: 'f32', access: 'read'},
                    {
                      name: 'output',
                      view: getView<'float32'>('centeredSS'),
                      type: 'f32',
                      access: 'read_write'
                    }
                  ],
                  body: `let deviation = workspace[workspaceOffset + 7u]; output[outputOffset] = workspace[workspaceOffset + 3u] - deviation * (deviation / 524288.0);`
                })
              ];
            }
          })
        })
      ],
      outputs: ['code', 'covariance', 'centeredSS']
    });
    // Closed form for one zero and n-1 identical a: SS=a²(n-1)/n=3.917943892233287.
    expect(Array.from(result['code'])).toEqual([0]);
    expect(Array.from(result['covariance'])).toEqual([0, 0, 0, 0]);
    expect(Number.isFinite(result['centeredSS'][0])).toBe(true);
    expect(Math.abs(result['centeredSS'][0] - 3.917943892233287)).toBeLessThan(2e-5);
  });
