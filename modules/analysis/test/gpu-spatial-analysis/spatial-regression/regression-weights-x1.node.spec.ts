// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialErrorGM} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm';
import {GPUSpatialTwoStageLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';
import {GPUSpatialWeightsTraces} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces';
import {GPUSpatialLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood';
import {GPUSpatialErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {GPUSpatialErrorGMOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm-operation';
import {GPUSpatialTwoStageLeastSquaresOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares-operation';
import {GPUSpatialWeightsTracesOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces-operation';
import {GPUSpatialLagMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood-operation';
import {GPUSpatialErrorMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood-operation';
import {CPUSpatialErrorGMOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-error-gm-operation';
import {CPUSpatialTwoStageLeastSquaresOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-two-stage-least-squares-operation';
import {CPUSpatialWeightsTracesOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-weights-traces-operation';
import {CPUSpatialLagMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-lag-maximum-likelihood-operation';
import {CPUSpatialErrorMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-error-maximum-likelihood-operation';
import {validateCPURegressionWeights} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-regression-weights';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import type {GPUContributorOperationViewGetter} from '../../../src/analysis-core/operations/gpu-contributor-operation';

for (const Operation of [
  GPUSpatialErrorGMOperation,
  GPUSpatialTwoStageLeastSquaresOperation,
  GPUSpatialWeightsTracesOperation,
  GPUSpatialLagMaximumLikelihoodOperation,
  GPUSpatialErrorMaximumLikelihoodOperation
]) {
  it(`${Operation.name} lowering retains an inherited tile width`, () => {
    const {device, view, props} = setup();
    try {
      const named = Object.assign(
        Object.create({
          get tileRowCount() {
            return 1;
          }
        }),
        {
          id: 'inherited-operation',
          rowCount: 8,
          predictorCount: 1,
          predictors: 'predictors',
          response: 'response',
          maximumPower: 2,
          mode: 'stochastic',
          traceMode: 'stochastic',
          traceProbeCount: 1,
          probeCount: 1,
          weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
          output: {
            table: 'table',
            summary: 'summary',
            powerTraces: 'powerTraces',
            powerSums: 'powerSums',
            infinityNorm: 'infinityNorm',
            status: {code: 'code', converged: 'converged'}
          }
        }
      );
      const operation = new Operation(named);
      const ports = {
        ...props.weights,
        predictors: props.predictors,
        response: props.response,
        ...Object.fromEntries(
          Object.entries(operation.outputs).map(([name, output]) => [
            name,
            view(output.format as 'float32' | 'uint32', output.length)
          ])
        )
      };
      const getView = ((name: keyof typeof ports) =>
        ports[name]) as GPUContributorOperationViewGetter;
      const contributor = operation.createContributor(getView) as unknown as {tileRowCount: number};
      expect(contributor.tileRowCount).toBe(1);
    } finally {
      device.destroy();
    }
  });
}

for (const Operation of [
  GPUSpatialErrorGMOperation,
  GPUSpatialTwoStageLeastSquaresOperation,
  GPUSpatialWeightsTracesOperation,
  GPUSpatialLagMaximumLikelihoodOperation,
  GPUSpatialErrorMaximumLikelihoodOperation,
  CPUSpatialErrorGMOperation,
  CPUSpatialTwoStageLeastSquaresOperation,
  CPUSpatialWeightsTracesOperation,
  CPUSpatialLagMaximumLikelihoodOperation,
  CPUSpatialErrorMaximumLikelihoodOperation
]) {
  it(`${Operation.name} explicitly refuses unsupported named loss ports`, () => {
    const props = {
      id: 'named-loss',
      rowCount: 8,
      predictorCount: 1,
      predictors: 'predictors',
      response: 'response',
      maximumPower: 2,
      weights: {
        offsets: 'offsets',
        neighbors: 'neighbors',
        weights: 'weights',
        status: {candidateOverflow: 'loss'}
      },
      output: {
        table: 'table',
        summary: 'summary',
        powerTraces: 'powerTraces',
        powerSums: 'powerSums',
        infinityNorm: 'infinityNorm',
        status: {code: 'code', converged: 'converged'}
      }
    };
    expect(() => new Operation(props)).toThrow(/named-loss weights.status/);
  });
}
it('CPU array twins refuse status-bearing topology instead of discarding the flag', () => {
  const weights = {
    offsets: Uint32Array.of(0, 1),
    neighbors: Uint32Array.of(0),
    weights: Float32Array.of(1),
    status: {candidateOverflow: Uint32Array.of(1)}
  };
  expect(() => validateCPURegressionWeights('cpu-loss', weights)).toThrow(
    /cpu-loss weights.status/
  );
});

const constructors = [
  GPUSpatialErrorGM,
  GPUSpatialTwoStageLeastSquares,
  GPUSpatialWeightsTraces,
  GPUSpatialLagMaximumLikelihood,
  GPUSpatialErrorMaximumLikelihood
];

function setup() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `x1-${serial++}`, format, length);
  const loss = view('uint32', 1);
  const weights = {
    offsets: view('uint32', 9),
    neighbors: view('uint32', 8),
    weights: view('float32', 8),
    status: {candidateOverflow: loss}
  };
  const output = {
    table: view('float32', 12),
    summary: view('float32', 32),
    powerTraces: view('float32', 2),
    powerSums: view('float32', 2),
    infinityNorm: view('float32', 1),
    status: {code: view('uint32', 1), converged: view('uint32', 1)}
  };
  const props = {
    id: 'x1',
    weights,
    predictorCount: 1,
    predictors: view('float32', 8),
    response: view('float32', 8),
    maximumPower: 2,
    estimator: 'kelejian-prucha' as const,
    mode: 'stochastic' as const,
    traceMode: 'stochastic' as const,
    traceProbeCount: 1,
    probeCount: 1,
    output
  };
  return {device, graph, view, loss, props};
}

for (const Contributor of constructors) {
  for (const chunked of [false, true]) {
    if (chunked && Contributor === GPUSpatialTwoStageLeastSquares) continue;
    it(`${Contributor.name} suppresses outputs for ${chunked ? 'chunked' : 'whole'} candidate loss`, () => {
      const {device, graph, props, loss} = setup();
      try {
        const weights = chunked
          ? {rowCount: 8, chunks: [{rowStart: 0, rowCount: 8, weights: props.weights}]}
          : props.weights;
        // The constructor union's whole-only member is excluded above for chunked inputs.
        const contributor = new Contributor({...props, weights: weights as typeof props.weights});
        const nodes = contributor.getCommandNodes(graph);
        expect(
          nodes.some(node =>
            node.resources?.some(
              resource =>
                'buffer' in resource &&
                (resource.buffer === loss || resource.buffer === loss.buffer)
            )
          )
        ).toBe(true);
        const kernels = getGPUCommandNodeKernelSources(nodes);
        for (const kernel of kernels.filter(kernel =>
          /weights-loss|suppress-incomplete/.test(kernel.id)
        )) {
          expect(auditWGSLModule(kernel.source), kernel.id).toEqual([]);
        }
        expect(
          kernels.some(
            kernel =>
              kernel.id.includes('suppress-incomplete') && kernel.source.includes('0x7fc00000u')
          )
        ).toBe(true);
      } finally {
        device.destroy();
      }
    });
  }

  it(`${Contributor.name} retains inherited tile options and original output maps`, () => {
    const {device, graph, props, view} = setup();
    try {
      const inherited = Object.assign(
        Object.create({
          get tileRowCount() {
            return 1;
          }
        }),
        props
      );
      const contributor = new Contributor(inherited);
      expect(contributor.tileRowCount).toBe(1);
      const original =
        Contributor === GPUSpatialWeightsTraces ? props.output.powerTraces : props.output.table;
      props.output.powerTraces = view('float32', 2);
      props.output.table = view('float32', 12);
      props.output.status.code = view('uint32', 1);
      const nodes = contributor.getCommandNodes(graph);
      expect(
        nodes.some(node =>
          node.resources?.some(
            resource =>
              'buffer' in resource &&
              (resource.buffer === original || resource.buffer === original.buffer)
          )
        )
      ).toBe(true);
    } finally {
      device.destroy();
    }
  });
}
