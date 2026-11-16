// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandNode, GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {GPUCatchmentAccessibility} from '../../../src/gpu-spatial-analysis/catchment-accessibility/gpu-catchment-accessibility';
import {GPUHuffTradeAreas} from '../../../src/gpu-spatial-analysis/catchment-accessibility/gpu-huff-trade-areas';
import {GPUGeographicDistribution} from '../../../src/gpu-spatial-analysis/geographic-distribution/gpu-geographic-distribution';
import {getGPUGeographicDistributionParameterValues} from '../../../src/gpu-spatial-analysis/geographic-distribution/geographic-distribution-parameters';
import type {RecipeTestFixture} from '../recipes/recipe-harness';

/** Use exactly the same caller IDs and graph declarations in node and real-device coverage. */
export function addFiniteScaleAnalysisFixture(fixture: RecipeTestFixture, scale: number) {
  const nodes: GPUCommandNode[] = [];
  const add = (contributor: GPUCommandNodeProducer) => {
    const commands = contributor.getCommandNodes(fixture.graph);
    nodes.push(...commands);
    fixture.graph.add(commands);
  };
  const weights = {
    offsets: fixture.input('offsets', new Uint32Array([0, 1]), 'uint32', 2),
    neighbors: fixture.input('neighbors', new Uint32Array([0]), 'uint32', 1),
    weights: fixture.input('weights', new Float32Array([1]), 'float32', 1)
  };
  const population = fixture.input('population', new Float32Array([scale]), 'float32', 1);
  const scores = (['2sfca', '3sfca'] as const).map(method => {
    const accessibility = fixture.output(`${method}-accessibility`, 'float32', 1);
    const ratios = fixture.output(`${method}-ratios`, 'float32', 1);
    add(
      new GPUCatchmentAccessibility({
        id: method,
        method,
        facilityWeights: weights,
        supply: population,
        demand: population,
        output: {accessibility: accessibility.view, ratios: ratios.view}
      })
    );
    return {accessibility, ratios};
  });
  const tradeArea = fixture.output('trade', 'uint32', 1);
  const probability = fixture.output('probability', 'float32', 1);
  const probabilities = fixture.output('probabilities', 'float32', 1);
  const expectedDemand = fixture.output('expected-demand', 'float32', 1);
  add(
    new GPUHuffTradeAreas({
      attractiveness: population,
      demandWeights: weights,
      facilityWeights: weights,
      demand: fixture.input('demand', new Float32Array([3]), 'float32', 1),
      output: {
        tradeArea: tradeArea.view,
        tradeAreaProbability: probability.view,
        probabilities: probabilities.view,
        expectedDemand: expectedDemand.view
      }
    })
  );
  const positions = fixture.input('positions', new Float32Array([1, 0]), 'float32x2', 1);
  const lineEnds = fixture.input('ends', new Float32Array([2, 0]), 'float32x2', 1);
  const parameters = fixture.input(
    'parameters',
    getGPUGeographicDistributionParameterValues(),
    'float32',
    8
  );
  const geographic = (['chunked', 'sorted'] as const).map(reduction => {
    const mean = fixture.output(`${reduction}-mean-output`, 'float32x2', 1);
    const direction = fixture.output(`${reduction}-direction-output`, 'float32', 3);
    add(
      new GPUGeographicDistribution({
        id: reduction,
        positions,
        lineEnds,
        weights: population,
        parameters,
        reduction,
        output: {meanCenters: mean.view, directionalMeans: direction.view}
      })
    );
    return {mean, direction};
  });
  return {scores, tradeArea, probability, probabilities, expectedDemand, geographic, nodes};
}
