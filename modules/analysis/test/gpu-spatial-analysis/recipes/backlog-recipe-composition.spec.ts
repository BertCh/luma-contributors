// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {addSurfacePreparationRecipe} from '../../../src/gpu-spatial-analysis/recipes/surface-preparation-recipe';
import {
  addExposureByZoneRecipe,
  addRasterZonalDistributionRecipe
} from '../../../src/gpu-spatial-analysis/recipes/raster-summary-recipes';
import {
  addFacilityScenarioRecipe,
  addNetworkDisruptionRecipe
} from '../../../src/gpu-network/recipes/network-scenarios-recipe';
import {getGPUNetworkAccessibilityParameterValues} from '../../../src/gpu-network/network-accessibility/index';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from './recipe-harness';

const GRID: GPURasterMetadata = {
  width: 6,
  height: 1,
  affine: [1, 0, 0, 0, 1, 0],
  pixelInterpretation: 'area'
};

async function createFixture(name: string) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return new RecipeTestFixture(device, name);
}

function publish<Format extends 'float32' | 'uint32'>(
  fixture: RecipeTestFixture,
  name: string,
  source: GraphDataView<Format>
) {
  const output = fixture.output(name, source.format, source.length);
  const type = source.format === 'float32' ? 'f32' : 'u32';
  fixture.graph.add(
    createWGSLKernelNode(fixture.graph, {
      id: name,
      operation: 'backlog-recipe-readback',
      invocationCount: source.length,
      bindings: [
        {name: 'source', view: source, type, access: 'read'},
        {name: 'destination', view: output.view, type, access: 'read_write'}
      ],
      body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
    })
  );
  return output;
}

it('backlog recipe composition retains unresolved support and dense zone identities', async () => {
  const fixture = await createFixture('backlog-surface-distribution');
  try {
    const prepared = addSurfacePreparationRecipe(fixture.graph, {
      grid: GRID,
      input: {
        id: 'observations',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: fixture.input(
            'observations',
            new Float32Array([2, NaN, NaN, NaN, NaN, 10]),
            'float32',
            6
          )
        }
      },
      maximumDistance: 1,
      method: 'nearest',
      filter: {radius: 1, quantile: 0.5, borderMode: 'nodata', noDataPolicy: 'ignore'}
    });
    const zones = fixture.input('zones', new Uint32Array([2, 2, 1, 1, 0, 0]), 'uint32', 6);
    const distribution = addRasterZonalDistributionRecipe(fixture.graph, {
      grid: GRID,
      input: prepared.band,
      zones,
      zoneCount: 4,
      probabilities: fixture.input('probabilities', new Float32Array([0, 0.5, 1]), 'float32', 3)
    });
    const values = publish(fixture, 'prepared-values', prepared.outputs.values);
    const original = publish(fixture, 'original', prepared.outputs.originalValidity);
    const filled = publish(fixture, 'filled', prepared.outputs.filled);
    const filtered = publish(fixture, 'filtered', prepared.outputs.filtered);
    const unresolved = publish(fixture, 'unresolved', prepared.outputs.unresolvedCount);
    const zoneIds = publish(fixture, 'zone-identities', distribution.outputs.zoneIds);
    const counts = publish(fixture, 'counts', distribution.outputs.validCounts);
    const medians = publish(fixture, 'medians', distribution.outputs.medians);
    const quantiles = publish(fixture, 'quantiles', distribution.outputs.quantiles!);
    // The raster wrapper's spread drops the recipe's index signature from the intermediates type.
    expect(distribution.intermediates.zones).toBe(zones);
    fixture.run();
    // Original donors reach only adjacent cells. Filtering cannot turn invalid centers valid.
    expect(await fixture.readFloat32(values, 6)).toEqual([2, 2, NaN, NaN, 10, 10]);
    expect(await fixture.readUint32(original, 6)).toEqual([1, 0, 0, 0, 0, 1]);
    expect(await fixture.readUint32(filled, 6)).toEqual([0, 1, 0, 0, 1, 0]);
    expect(await fixture.readUint32(filtered, 6)).toEqual([0, 0, 0, 0, 0, 0]);
    expect(await fixture.readUint32(unresolved, 1)).toEqual([2]);
    expect(await fixture.readUint32(zoneIds, 4)).toEqual([0, 1, 2, 3]);
    expect(await fixture.readUint32(counts, 4)).toEqual([2, 0, 2, 0]);
    expect(await fixture.readFloat32(medians, 4)).toEqual([10, NaN, 2, NaN]);
    expect(await fixture.readFloat32(quantiles, 12)).toEqual([
      10,
      10,
      10,
      NaN,
      NaN,
      NaN,
      2,
      2,
      2,
      NaN,
      NaN,
      NaN
    ]);
  } finally {
    fixture.destroy();
  }
});

it('backlog density exposure conserves population and recovers from invalid live thresholds', async () => {
  const fixture = await createFixture('backlog-density-exposure');
  let compiled: ReturnType<typeof fixture.graph.compile> | undefined;
  try {
    const hazard = fixture.output('hazard-input', 'float32', 6);
    hazard.buffer.write(new Float32Array([5, NaN, 1, 9, 8, 7]));
    const threshold = fixture.output('threshold-input', 'float32', 1);
    threshold.buffer.write(new Float32Array([5]));
    const result = addExposureByZoneRecipe(fixture.graph, {
      grid: GRID,
      hazard: {id: 'hazard', format: 'float32', storage: {kind: 'buffer', values: hazard.view}},
      population: {
        id: 'density',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: fixture.input('density', new Float32Array([2, 3, 4, -1, 10, 5]), 'float32', 6)
        }
      },
      zones: fixture.input('zones', new Uint32Array([0, 0, 1, 1, 1, 1]), 'uint32', 6),
      zoneCount: 3,
      weightMeaning: 'density',
      cellAreas: fixture.input('areas', new Float32Array([3, 2, 0.5, 4, 0, 2]), 'float32', 6),
      threshold: threshold.view
    });
    const total = publish(fixture, 'total', result.outputs.totalPopulation);
    const observed = publish(fixture, 'observed', result.outputs.observedPopulation);
    const exposed = publish(fixture, 'exposed', result.outputs.exposedPopulation);
    const coverage = publish(fixture, 'coverage', result.outputs.coverage);
    const share = publish(fixture, 'share', result.outputs.exposedShare);
    const invalid = publish(fixture, 'invalid', result.outputs.thresholdInvalidCount);
    compiled = fixture.graph.compile();
    submitGraph(fixture.device, compiled, undefined);
    // Valid cell populations are 6, 6, 2 and 10; negative density and zero area contribute none.
    expect(await fixture.readFloat32(total, 3)).toEqual([12, 12, 0]);
    expect(await fixture.readFloat32(observed, 3)).toEqual([6, 12, 0]);
    expect(await fixture.readFloat32(exposed, 3)).toEqual([6, 10, 0]);
    expect(await fixture.readFloat32(coverage, 3)).toEqual([0.5, 1, NaN]);
    expect((await fixture.readFloat32(share, 3))[1]).toBeCloseTo(5 / 6, 6);
    threshold.buffer.write(new Float32Array([NaN]));
    submitGraph(fixture.device, compiled, undefined);
    expect(await fixture.readUint32(invalid, 1)).toEqual([1]);
    expect(await fixture.readFloat32(exposed, 3)).toEqual([NaN, NaN, NaN]);
    expect(await fixture.readFloat32(share, 3)).toEqual([NaN, NaN, NaN]);
    expect(await fixture.readFloat32(total, 3)).toEqual([12, 12, 0]);
    hazard.buffer.write(new Float32Array([5, 6, 1, 9, 8, 7]));
    threshold.buffer.write(new Float32Array([7]));
    submitGraph(fixture.device, compiled, undefined);
    expect(await fixture.readUint32(invalid, 1)).toEqual([0]);
    expect(await fixture.readFloat32(observed, 3)).toEqual([12, 12, 0]);
    expect(await fixture.readFloat32(exposed, 3)).toEqual([0, 10, 0]);
    expect(await fixture.readFloat32(coverage, 3)).toEqual([1, 1, NaN]);
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
});

it('backlog facility scenarios conserve valid demand across closure and reopening without renumbering', async () => {
  const fixture = await createFixture('backlog-facility-updates');
  let compiled: ReturnType<typeof fixture.graph.compile> | undefined;
  try {
    const mask = fixture.output('facility-mask', 'uint32', 3);
    const demand = fixture.output('demand-input', 'float32', 6);
    mask.buffer.write(new Uint32Array([1, 0, 1]));
    demand.buffer.write(new Float32Array([10, 20, 30, 40, -5, NaN]));
    const result = addFacilityScenarioRecipe(fixture.graph, {
      costs: fixture.input(
        'costs',
        new Float32Array([1, 4, Infinity, 2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 2, 3, Infinity, 1, 1]),
        'float32',
        18
      ),
      demand: demand.view,
      supply: fixture.input('supply', new Float32Array([1, 1, 1]), 'float32', 3),
      facilityMask: mask.view,
      parameters: fixture.parameters(
        'parameters',
        'float32',
        getGPUNetworkAccessibilityParameterValues({threshold: 2})
      )
    });
    const covered = publish(fixture, 'covered', result.outputs.coveredDemand);
    const uncovered = publish(fixture, 'uncovered', result.outputs.uncoveredDemand);
    const facilities = publish(fixture, 'facilities', result.outputs.nearestFacilities);
    const invalid = publish(fixture, 'invalid', result.outputs.invalidDemandCount);
    compiled = fixture.graph.compile();
    submitGraph(fixture.device, compiled, undefined);
    expect(await fixture.readUint32(facilities, 6)).toEqual([0, 2, 2, 0, 0, 0]);
    expect(await fixture.readFloat32(covered, 1)).toEqual([70]);
    expect(await fixture.readFloat32(uncovered, 1)).toEqual([30]);
    expect(await fixture.readUint32(invalid, 1)).toEqual([2]);
    mask.buffer.write(new Uint32Array([0, 0, 0]));
    submitGraph(fixture.device, compiled, undefined);
    expect(await fixture.readUint32(facilities, 6)).toEqual(Array(6).fill(0xffffffff));
    expect(await fixture.readFloat32(covered, 1)).toEqual([0]);
    expect(await fixture.readFloat32(uncovered, 1)).toEqual([100]);
    mask.buffer.write(new Uint32Array([0, 0, 1]));
    demand.buffer.write(new Float32Array([10, 20, 30, 40, 5, 0]));
    submitGraph(fixture.device, compiled, undefined);
    expect(await fixture.readUint32(facilities, 6)).toEqual([2, 2, 2, 0xffffffff, 2, 2]);
    expect(await fixture.readFloat32(covered, 1)).toEqual([35]);
    expect(await fixture.readFloat32(uncovered, 1)).toEqual([70]);
    expect(await fixture.readUint32(invalid, 1)).toEqual([0]);
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
});

it('backlog disruption gates retained-baseline failures and preserves parallel-edge slots', async () => {
  const fixture = await createFixture('backlog-disruption-baseline');
  let compiled: ReturnType<typeof fixture.graph.compile> | undefined;
  try {
    const topology = {
      offsets: fixture.input('offsets', new Uint32Array([0, 2, 3, 3]), 'uint32', 4),
      neighbors: fixture.input('neighbors', new Uint32Array([1, 1, 2]), 'uint32', 3),
      weights: fixture.input('weights', new Float32Array([1, 4, 2]), 'float32', 3)
    };
    const origins = fixture.input('origins', new Uint32Array([0]), 'uint32', 1);
    const targets = fixture.input('targets', new Uint32Array([2, 0, 1]), 'uint32', 3);
    const baseline = fixture.input('retained-costs', new Float32Array([3, 0, 1]), 'float32', 3);
    const converged = fixture.output('retained-converged', 'uint32', 1);
    const mask = fixture.output('edge-mask', 'uint32', 3);
    converged.buffer.write(new Uint32Array([0]));
    mask.buffer.write(new Uint32Array([0, 1, 1]));
    const result = addNetworkDisruptionRecipe(fixture.graph, {
      search: {topology, startNodes: origins, targetNodes: targets, maximumIterations: 8},
      rowCount: 1,
      baseline: {costs: baseline, status: {converged: converged.view}},
      edgeMask: mask.view,
      opportunityWeights: fixture.input('opportunities', new Float32Array([5, 1, 2]), 'float32', 3),
      parameters: fixture.parameters(
        'parameters',
        'float32',
        getGPUNetworkAccessibilityParameterValues({threshold: 3})
      )
    });
    expect(result.outputs.baselineCosts).toBe(baseline);
    expect(result.intermediates['topology']).toEqual(topology);
    expect(result.intermediates['originNodes']).toBe(origins);
    expect(result.intermediates['targetNodes']).toBe(targets);
    const costs = publish(fixture, 'scenario-costs', result.outputs.scenarioCosts);
    const differences = publish(fixture, 'differences', result.outputs.costDifferences);
    const unreachable = publish(fixture, 'unreachable', result.outputs.newlyUnreachable);
    const accessibility = publish(
      fixture,
      'accessibility',
      result.outputs.accessibilityDifferences
    );
    const retained = publish(fixture, 'retained', result.outputs.baselineCosts);
    compiled = fixture.graph.compile();
    submitGraph(fixture.device, compiled, undefined);
    expect(await fixture.readFloat32(costs, 3)).toEqual([6, 0, 4]);
    expect(await fixture.readFloat32(differences, 3)).toEqual([NaN, NaN, NaN]);
    expect(await fixture.readUint32(unreachable, 3)).toEqual([0, 0, 0]);
    expect(await fixture.readFloat32(accessibility, 1)).toEqual([NaN]);
    converged.buffer.write(new Uint32Array([1]));
    submitGraph(fixture.device, compiled, undefined);
    expect(await fixture.readFloat32(differences, 3)).toEqual([3, 0, 3]);
    expect(await fixture.readFloat32(accessibility, 1)).toEqual([-7]);
    mask.buffer.write(new Uint32Array([0, 1, 0]));
    submitGraph(fixture.device, compiled, undefined);
    expect(await fixture.readFloat32(costs, 3)).toEqual([Infinity, 0, 4]);
    expect(await fixture.readUint32(unreachable, 3)).toEqual([1, 0, 0]);
    expect(await fixture.readFloat32(differences, 3)).toEqual([NaN, 0, 3]);
    expect(await fixture.readFloat32(retained, 3)).toEqual([3, 0, 1]);
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
});
