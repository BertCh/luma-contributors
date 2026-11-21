// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  computeVecchiaLogLikelihoodOnCPU,
  CPUVecchiaLogLikelihoodOperation
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-log-likelihood-operation';
import {
  computeVecchiaSpaceTimeLogLikelihoodOnCPU,
  CPUVecchiaSpaceTimeLogLikelihoodOperation
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-space-time-log-likelihood-operation';
import {
  computeVecchiaFisherScoringOnCPU,
  CPUVecchiaFisherScoringOperation
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-fisher-scoring-operation';
import {GPUVecchiaLogLikelihood} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-log-likelihood';
import {
  GPUVecchiaSpaceTimeLogLikelihood,
  getGPUVecchiaSpaceTimeLogLikelihoodParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-space-time-log-likelihood';
import {GPUVecchiaFisherScoring} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-fisher-scoring';
import {GPUVecchiaLogLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-log-likelihood-operation';
import {GPUVecchiaSpaceTimeLogLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-space-time-log-likelihood-operation';
import {GPUVecchiaFisherScoringOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-fisher-scoring-operation';
import {getGPUVecchiaLogLikelihoodParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-parameters';
import {getCPUVecchiaNeighborhoodStatus} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-neighborhood-status';
import {
  expectVecchiaMarginalTotals,
  expectVecchiaSuppressedTotals
} from './vecchia-marginal-expectations';

type Route = 'spatial' | 'space-time' | 'fisher';
const covariance = {model: 'exponential' as const, sill: 1, range: 3, nugget: 0.1};
const temporalContext = {
  encoding: 'float32-relative' as const,
  unit: 'second' as const,
  epoch: 0n,
  axis: {kind: 'irregular' as const}
};
const routeId = (route: Route) =>
  route === 'fisher'
    ? 'vecchia-fisher-scoring'
    : route === 'space-time'
      ? 'vecchia-space-time-log-likelihood'
      : 'vecchia-log-likelihood';

for (const route of ['spatial', 'space-time', 'fisher'] as const) {
  for (const placement of ['inherited', 'non-enumerable'] as const) {
    for (const certificate of ['overflow', 'complete'] as const) {
      it(`B1 ${route} captures ${placement} ${certificate} before defaulting the evaluator ID`, () => {
        let reads = 0;
        const flag = Uint32Array.of(certificate === 'overflow' ? 23 : 0);
        const input = {
          positions: Float32Array.of(0, 0, 1, 0),
          observations: Float32Array.of(1, 0, 0, 1, 1, 0, 0, 1),
          neighborhoodOffsets: Uint32Array.of(0, 0, 0),
          neighborhoods: new Uint32Array(),
          parameters:
            route === 'space-time'
              ? getGPUVecchiaSpaceTimeLogLikelihoodParameterValues({
                  type: 'metric',
                  joint: covariance,
                  anisotropy: 1
                })
              : getGPUVecchiaLogLikelihoodParameterValues(covariance),
          times: Float32Array.of(0, 0)
        };
        const descriptor = {
          get() {
            reads++;
            return {[certificate]: flag};
          },
          enumerable: false
        };
        if (placement === 'inherited')
          Object.setPrototypeOf(input, Object.defineProperty({}, 'neighborhoodStatus', descriptor));
        else Object.defineProperty(input, 'neighborhoodStatus', descriptor);
        const evaluate = () =>
          route === 'fisher'
            ? computeVecchiaFisherScoringOnCPU(input)
            : route === 'space-time'
              ? computeVecchiaSpaceTimeLogLikelihoodOnCPU(input)
              : computeVecchiaLogLikelihoodOnCPU(input);
        if (certificate === 'complete') {
          expect(evaluate).toThrow(new RegExp(`${routeId(route)}.*complete.*completeness-aware`));
          expect(reads).toBe(1);
        } else {
          // Two independent N(0, 1.1) marginals: the existing literal oracle checks every total.
          const lost = evaluate();
          expect(Array.from(lost.neighborhoodStatus!)).toEqual([0, 1, 0]);
          expect(reads).toBe(1);
          expectVecchiaSuppressedTotals(
            'totals' in lost ? lost.totals : lost.logLikelihood,
            route === 'fisher'
          );
          flag[0] = 0;
          const recovered = evaluate();
          expect(reads).toBe(2);
          expect(Array.from(recovered.neighborhoodStatus!)).toEqual([1, 0, 0]);
          expectVecchiaMarginalTotals(
            'totals' in recovered ? recovered.totals : recovered.logLikelihood,
            route === 'fisher'
          );
        }
      });
    }
  }
}

function createGPUSetup(route: Route) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let next = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `review-${next++}`, format, length);
  const props = {
    positions: view('float32x2', 2),
    observations: view('float32x4', 2),
    neighborhoodOffsets: view('uint32', 3),
    neighborhoods: view('uint32', 1),
    parameters: view('float32', route === 'space-time' ? 28 : 8),
    neighborCount: 1,
    times: view('float32', 2),
    temporalContext,
    output: {
      conditionals: view('float32x4', 2),
      conditionalStatus: view('uint32', 2),
      siteTerms: view('float32', route === 'fisher' ? 20 : 2),
      siteStatus: view('uint32', 2),
      logLikelihood: view('float32', 2),
      totals: view('float32', 20),
      status: view('uint32', 3)
    }
  };
  const Contributor =
    route === 'fisher'
      ? GPUVecchiaFisherScoring
      : route === 'space-time'
        ? GPUVecchiaSpaceTimeLogLikelihood
        : GPUVecchiaLogLikelihood;
  return {device, graph, view, props, Contributor};
}

for (const route of ['spatial', 'space-time', 'fisher'] as const) {
  it(`B2 ${route} validates and writes the first captured evidence destination`, () => {
    const {device, graph, view, props} = createGPUSetup(route);
    let reads = 0;
    const evidence = view('uint32', 3);
    const output = {
      ...props.output,
      get neighborhoodStatus() {
        return ++reads === 1 ? evidence : undefined;
      }
    };
    try {
      const contributor =
        route === 'fisher'
          ? new GPUVecchiaFisherScoring({...props, output})
          : route === 'space-time'
            ? new GPUVecchiaSpaceTimeLogLikelihood({...props, output})
            : new GPUVecchiaLogLikelihood({...props, output});
      expect(reads).toBe(1);
      expect(contributor.props.output.neighborhoodStatus).toBe(evidence);
      const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
      expect(
        sources.some(({source}) => source.includes('evidence[evidenceOffset] = complete'))
      ).toBe(true);
      expect(reads).toBe(1);
    } finally {
      device.destroy();
    }
  });

  for (const invalid of [0, false, '']) {
    it(`S1 ${route} rejects falsy negative port ${JSON.stringify(invalid)}`, () => {
      const {device, props, Contributor} = createGPUSetup(route);
      try {
        for (const name of ['overflow', 'candidateOverflow']) {
          expect(() =>
            Reflect.construct(Contributor, [{...props, neighborhoodStatus: {[name]: invalid}}])
          ).toThrow(new RegExp(`${routeId(route)}.*neighborhoodStatus.${name}.*graph view`));
        }
      } finally {
        device.destroy();
      }
    });
  }

  it(`S2 ${route} uses the resolved default ID in GPU completeness errors`, () => {
    const {device, view, props, Contributor} = createGPUSetup(route);
    try {
      expect(() =>
        Reflect.construct(Contributor, [
          {...props, neighborhoodStatus: {complete: view('uint32', 1)}}
        ])
      ).toThrow(new RegExp(`^${routeId(route)} .*complete`));
    } finally {
      device.destroy();
    }
  });

  for (const cpu of [false, true]) {
    it(`S2 ${route} uses the resolved default ID in ${cpu ? 'CPU' : 'GPU'} operation completeness errors`, () => {
      const output = {
        conditionals: 'conditionals',
        conditionalStatus: 'conditional-status',
        siteTerms: 'terms',
        siteStatus: 'sites',
        logLikelihood: 'likelihood',
        status: 'counts',
        totals: 'totals'
      };
      const props = {
        rowCount: 2,
        neighborCount: 1,
        positions: 'positions',
        observations: 'observations',
        neighborhoodOffsets: 'offsets',
        neighborhoods: 'neighbors',
        parameters: 'parameters',
        neighborhoodStatus: {complete: 'complete'},
        output,
        times: 'times',
        temporalContext
      };
      const Operation =
        route === 'fisher'
          ? cpu
            ? CPUVecchiaFisherScoringOperation
            : GPUVecchiaFisherScoringOperation
          : route === 'space-time'
            ? cpu
              ? CPUVecchiaSpaceTimeLogLikelihoodOperation
              : GPUVecchiaSpaceTimeLogLikelihoodOperation
            : cpu
              ? CPUVecchiaLogLikelihoodOperation
              : GPUVecchiaLogLikelihoodOperation;
      expect(() => Reflect.construct(Operation, [props])).toThrow(
        new RegExp(`^${routeId(route)} .*complete`)
      );
    });
  }
}

it('N1 CPU loss words require Uint32Array rather than byte or numeric array lookalikes', () => {
  const words = Uint32Array.of(256);
  expect(
    Array.from(getCPUVecchiaNeighborhoodStatus('typed-certificate', {candidateOverflow: words}))
  ).toEqual([0, 0, 1]);
  for (const invalid of [new Uint8Array(words.buffer), [256], Float32Array.of(256), 0, false, '']) {
    expect(() =>
      Reflect.apply(getCPUVecchiaNeighborhoodStatus, undefined, [
        'typed-certificate',
        {candidateOverflow: invalid}
      ])
    ).toThrow(/typed-certificate.*candidateOverflow.*Uint32Array/);
  }
});
