// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPULocalKriging,
  type GPULocalKrigingProps
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {
  GPUConditionalGaussianSimulation,
  type GPUConditionalGaussianSimulationProps
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation';
import {GPUConditionalGaussianSimulationOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation-operation';
import {GPUVecchiaLogLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-log-likelihood-operation';
import {GPUVecchiaFisherScoringOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-fisher-scoring-operation';
import {GPUVecchiaSpaceTimeLogLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-space-time-log-likelihood-operation';
import {
  getFisherProps,
  getLikelihoodProps,
  getSpaceTimeLikelihoodProps,
  getLikelihoodShapes
} from './likelihoods-operation-test-utils';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createViews() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  return {
    graph,
    positions: createTransientView(graph, 'positions', 'float32x2', 2),
    observations: createTransientView(graph, 'observations', 'float32x4', 2),
    queries: createTransientView(graph, 'queries', 'float32x2', 1),
    values: createTransientView(graph, 'values', 'float32x4', 1),
    status: createTransientView(graph, 'status', 'uint32', 1)
  };
}

it('PTS-133 local kriging captures nested covariance, loading arrays, means and output routing before declaration', () => {
  const views = createViews();
  const props: GPULocalKrigingProps = {
    id: 'snapshot-local',
    positions: views.positions,
    observations: views.observations,
    supports: views.queries,
    mode: 'co-kriging',
    variableCount: 2,
    neighborCount: 2,
    means: [1, 2],
    covariance: {model: 'exponential', sill: 1, range: 3},
    coregionalizationModels: [
      {covariance: {model: 'exponential', sill: 1, range: 3}, loadings: [1, 0, 0.25, 1]}
    ],
    output: {values: views.values, status: views.status}
  };
  const contributor = new GPULocalKriging(props);
  props.covariance!.range = 99;
  props.coregionalizationModels![0].covariance.range = 98;
  // Caller objects remain writable; captured construction metadata is independent.
  (props.coregionalizationModels![0].loadings as number[])[0] = 4;
  (props.means as number[])[0] = 9;
  props.output.status = createTransientView(views.graph, 'replacement-status', 'uint32', 1);
  expect(contributor.props.covariance!.range).toBe(3);
  expect(contributor.props.coregionalizationModels![0].covariance.range).toBe(3);
  expect(contributor.props.coregionalizationModels![0].loadings[0]).toBe(1);
  expect(contributor.props.means![0]).toBe(1);
  expect(contributor.props.output.status).toBe(views.status);
  expect(contributor.props.observations).toBe(views.observations);
  expect(contributor.props.observations.buffer).toBe(views.observations.buffer);
  expect(contributor.getCommandNodes(views.graph).length).toBeGreaterThan(0);
});

it('PTS-133 simulation captures covariance between construction and declaration and retains live views', () => {
  const views = createViews();
  const props: GPUConditionalGaussianSimulationProps = {
    positions: views.positions,
    observations: views.observations,
    queries: views.queries,
    covariance: {model: 'exponential', sill: 1, range: 3},
    seed: 1,
    replicateCount: 1,
    output: {
      values: createTransientView(views.graph, 'draws', 'float32', 1),
      moments: createTransientView(views.graph, 'moments', 'float32x2', 1),
      status: views.status
    }
  };
  const contributor = new GPUConditionalGaussianSimulation(props);
  props.covariance.range = 99;
  props.seed = 9;
  expect(contributor.props.covariance.range).toBe(3);
  expect(contributor.props.seed).toBe(1);
  expect(contributor.props.observations).toBe(views.observations);
  expect(contributor.getCommandNodes(views.graph).length).toBeGreaterThan(0);
});

it('PTS-133 deferred simulation operation captures nested settings before lowering', () => {
  const props = {
    observationCount: 2,
    queryCount: 1,
    positions: 'positions',
    observations: 'observations',
    queries: 'queries',
    covariance: {model: 'exponential' as const, sill: 1, range: 3},
    seed: 1,
    replicateCount: 1,
    output: {values: 'draws', moments: 'moments', status: 'status'}
  };
  const operation = new GPUConditionalGaussianSimulationOperation(props);
  props.covariance.range = 99;
  props.replicateCount = 99;
  const contributor = constructNetworkOperationContributor(operation, {
    positions: {format: 'float32x2', length: 2},
    observations: {format: 'float32x4', length: 2},
    queries: {format: 'float32x2', length: 1}
  }) as GPUConditionalGaussianSimulation;
  expect(contributor.props.covariance.range).toBe(3);
  expect(contributor.props.replicateCount).toBe(1);
});

it('PTS-133 all three Vecchia operations capture counts before deferred lowering', () => {
  const likelihoodProps = getLikelihoodProps();
  const fisherProps = getFisherProps();
  const spaceTimeProps = getSpaceTimeLikelihoodProps();
  const operations = [
    new GPUVecchiaLogLikelihoodOperation(likelihoodProps),
    new GPUVecchiaFisherScoringOperation(fisherProps),
    new GPUVecchiaSpaceTimeLogLikelihoodOperation(spaceTimeProps)
  ];
  for (const props of [likelihoodProps, fisherProps, spaceTimeProps]) {
    props.neighborCount = 0;
    props.rowCount = 0;
  }
  for (const operation of operations) {
    expect(() =>
      constructNetworkOperationContributor(
        operation,
        getLikelihoodShapes(operation instanceof GPUVecchiaSpaceTimeLogLikelihoodOperation)
      )
    ).not.toThrow();
  }
});
