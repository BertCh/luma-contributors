// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {addExposureByZoneRecipe} from '../../../src/gpu-spatial-analysis/recipes/raster-summary-recipes';
import {addSpatialRateRecipe} from '../../../src/gpu-spatial-analysis/recipes/rate-mapping-recipes';
import {addContiguityOrderCorrelogramRecipe} from '../../../src/gpu-spatial-analysis/recipes/contiguity-order-correlogram-recipe';
import {RecipeTestFixture} from './recipe-harness';

function publish<Format extends 'uint32' | 'float32'>(
  fixture: RecipeTestFixture,
  name: string,
  source: GraphDataView<Format>
) {
  const output = fixture.output(name, source.format, source.length);
  fixture.graph.add(
    createWGSLKernelNode(fixture.graph, {
      id: name,
      operation: 'wave3-readback',
      invocationCount: source.length,
      bindings: [
        {
          name: 'source',
          view: source,
          type: source.format === 'uint32' ? 'u32' : 'f32',
          access: 'read'
        },
        {
          name: 'output',
          view: output.view,
          type: source.format === 'uint32' ? 'u32' : 'f32',
          access: 'read_write'
        }
      ],
      body: 'output[outputOffset + index] = source[sourceOffset + index];'
    })
  );
  return output;
}

it('W3-07 raster loss suppresses summaries and complete replay restores valid zero', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'w3-raster');
  try {
    const producer = fixture.output('producer', 'uint32', 1);
    const values = fixture.input('values', Float32Array.of(0, 0), 'float32', 2);
    const band = {
      id: 'band',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values}
    };
    const props = {
      grid: {
        width: 2,
        height: 1,
        affine: [1, 0, 0, 0, 1, 0] as const,
        pixelInterpretation: 'area' as const
      },
      zones: fixture.input('zones', Uint32Array.of(0, 0), 'uint32', 2),
      zoneCount: 1,
      threshold: fixture.input('threshold', Float32Array.of(1), 'float32', 1),
      producerOverflow: producer.view
    };
    const exposure = addExposureByZoneRecipe(fixture.graph, {
      ...props,
      hazard: band,
      population: band,
      weightMeaning: 'count'
    });
    const total = publish(fixture, 'total', exposure.outputs.totalPopulation);
    const loss = publish(fixture, 'loss', exposure.status.stages[0].status.candidateOverflow!);
    for (const flag of [1, 0, 1, 0]) {
      producer.buffer.write(Uint32Array.of(flag));
      fixture.run();
      expect(await fixture.readUint32(loss, 1)).toEqual([flag]);
      const actual = (await fixture.readFloat32(total, 1))[0];
      if (flag) expect(Number.isNaN(actual)).toBe(true);
      else {
        expect(Number.isFinite(actual)).toBe(true);
        expect(actual).toBe(0);
      }
    }
  } finally {
    fixture.destroy();
  }
});

it('CPL-15 chunk-only loss clears on replay and pooled rates recover', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'w3-rate');
  try {
    const candidate = fixture.output('candidate', 'uint32', 1);
    const result = addSpatialRateRecipe(fixture.graph, {
      events: fixture.input('events', Float32Array.of(1, 3), 'float32', 2),
      populations: fixture.input('population', Float32Array.of(10, 10), 'float32', 2),
      weights: {
        rowCount: 2,
        chunks: [
          {
            rowStart: 0,
            rowCount: 2,
            weights: {
              offsets: fixture.input('offsets', Uint32Array.of(0, 1, 2), 'uint32', 3),
              neighbors: fixture.input('neighbors', Uint32Array.of(1, 0), 'uint32', 2),
              weights: fixture.input('weights', Float32Array.of(1, 1), 'float32', 2),
              status: {candidateOverflow: candidate.view}
            }
          }
        ]
      }
    });
    const rates = publish(fixture, 'rates', result.outputs.spatialRates);
    const loss = publish(fixture, 'loss', result.status.stages[0].status.candidateOverflow!);
    const overflow = publish(fixture, 'overflow', result.status.stages[0].status.overflow!);
    for (const flag of [1, 0]) {
      candidate.buffer.write(Uint32Array.of(flag));
      fixture.run();
      expect(await fixture.readUint32(loss, 1)).toEqual([flag]);
      expect(await fixture.readUint32(overflow, 1)).toEqual([0]);
      const rateValues = await fixture.readFloat32(rates, 2);
      expect(rateValues).toHaveLength(2);
      for (const value of rateValues) {
        if (flag) expect(Number.isNaN(value)).toBe(true);
        else {
          expect(Number.isFinite(value)).toBe(true);
          expect(value).toBeCloseTo(0.2, 6);
        } // (1+3)/(10+10).
      }
    }
  } finally {
    fixture.destroy();
  }
});

it('RCP-103 candidate-only weights evidence and ring stages recover on replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'w3-correlogram');
  try {
    const candidate = fixture.output('candidate', 'uint32', 1);
    const result = addContiguityOrderCorrelogramRecipe(fixture.graph, {
      values: fixture.input('values', Float32Array.of(1, 2, 3, 4), 'float32', 4),
      maximumOrder: 1,
      weights: {
        offsets: fixture.input('offsets', Uint32Array.of(0, 2, 4, 6, 8), 'uint32', 5),
        neighbors: fixture.input('neighbors', Uint32Array.of(1, 3, 0, 2, 1, 3, 0, 2), 'uint32', 8),
        weights: fixture.input('weights', new Float32Array(8).fill(1), 'float32', 8),
        status: {candidateOverflow: candidate.view}
      }
    });
    const summary = publish(fixture, 'summary', result.outputs.correlogram);
    expect(result.status.stages.map(stage => stage.stage)).toEqual(['weights', 'order-1']);
    expect(result.status.stages[0].status.candidateOverflow).toBe(candidate.view);
    for (const flag of [1, 0]) {
      candidate.buffer.write(Uint32Array.of(flag));
      fixture.run();
      const value = (await fixture.readFloat32(summary, 10))[0];
      if (flag) expect(Number.isNaN(value)).toBe(true);
      else {
        expect(Number.isFinite(value)).toBe(true);
        expect(value).toBeCloseTo(-0.2, 6);
      } // centered cross-products -2 / squared deviations 5.
    }
  } finally {
    fixture.destroy();
  }
});

import {addStopValueCubeRecipe} from '../../../src/gpu-spatial-analysis/recipes/index';
import {computePointToCellOnCPU} from '../../../src/gpu-spatial-analysis/cell-indexing/cpu-point-to-cell-operation';
it.each([
  'start',
  'finish'
] as const)('CMP-12 fixed dictionary bins live stop %s times and preserves loss on replay', async time => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'w3-stops');
  try {
    const candidate = fixture.output('candidate', 'uint32', 1);
    // Two cells on opposite sides of the prime meridian, north of the equator.
    // Different first occupied keys per period: cell B at t=0 and A at t=1.
    const positions = Float32Array.of(-90, 30, 90, 30);
    const dictionary = computePointToCellOnCPU(positions, {family: 'quadbin', resolution: 1}).cells;
    const result = addStopValueCubeRecipe(fixture.graph, {
      time,
      family: 'quadbin',
      resolution: 1,
      timestamps: fixture.input('timestamps', Uint32Array.of(0, 0, 10, 0, 20, 0), 'uint32x2', 3),
      temporalContext: {
        unit: 'second',
        epoch: 'unix',
        encoding: 'int64-words',
        axis: {kind: 'irregular'}
      },
      periods: {kind: 'fixed', origin: 0n, width: 10, count: 3},
      dictionary: fixture.input('dictionary', dictionary, 'uint32x2', 2),
      weights: {
        offsets: fixture.input('offsets', Uint32Array.of(0, 1, 2), 'uint32', 3),
        neighbors: fixture.input('neighbors', Uint32Array.of(1, 0), 'uint32', 2),
        weights: fixture.input('weights', Float32Array.of(1, 1), 'float32', 2)
      },
      stops: {
        centroids: fixture.input(
          'centroids',
          Float32Array.of(90, 30, -90, 30, -90, 30, -90, 30),
          'float32x2',
          4
        ),
        startRows: fixture.input('starts', Uint32Array.of(0, 1, 1, 0), 'uint32', 4),
        finishRows: fixture.input('finishes', Uint32Array.of(1, 1, 2, 0), 'uint32', 4),
        status: {
          count: fixture.input('count', Uint32Array.of(3), 'uint32', 1),
          candidateOverflow: candidate.view
        }
      }
    });
    const counts = publish(fixture, 'counts', result.outputs.counts);
    const loss = publish(fixture, 'loss', result.status.candidateOverflow);
    for (const flag of [1, 0, 1, 0]) {
      candidate.buffer.write(Uint32Array.of(flag));
      fixture.run();
      expect(await fixture.readUint32(loss, 1)).toEqual([flag]);
      // Hand tally A then B: start A=[0,2,0], B=[1,0,0]; finish A=[0,1,1], B=[0,1,0].
      // The spare lies in A at period zero, so removing the count mask changes either tally.
      expect(await fixture.readUint32(counts, 6)).toEqual(
        flag ? [0, 0, 0, 0, 0, 0] : time === 'start' ? [0, 2, 0, 1, 0, 0] : [0, 1, 1, 0, 1, 0]
      );
    }
  } finally {
    fixture.destroy();
  }
});

import {GPUQuantileLISA} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-quantile-lisa';
import {getGPUPermutationParameterValues} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-parameters';
import {QUANTILE_LISA_EQUAL_CYCLE} from './quantile-lisa-reference';
it('DEF-01 equal-cycle inclusive tails remain one at 9 and 99 draws', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const reference of QUANTILE_LISA_EQUAL_CYCLE.cases) {
    for (const alternative of ['greater', 'two-sided', 'geoda-legacy'] as const) {
      const fixture = new RecipeTestFixture(device, `w3-lisa-${alternative}-${reference.draws}`);
      try {
        const pValues = fixture.output('p', 'float32', 4);
        const inferenceOverflow = fixture.output('inference-overflow', 'uint32', 1);
        const membership = fixture.output('membership', 'uint32', 4);
        const boundaries = fixture.output('boundaries', 'float32', 2);
        const validCount = fixture.output('valid-count', 'uint32', 1);
        const classes = fixture.output('classes', 'uint32', 4);
        const invalidQuantile = fixture.output('invalid-quantile', 'uint32', 1);
        // Spread preserves a compilable fixture until spatial-stats publishes the alternative prop.
        fixture.graph.add(
          new GPUQuantileLISA({
            ...{alternative},
            values: fixture.input(
              'values',
              Float32Array.from(QUANTILE_LISA_EQUAL_CYCLE.values),
              'float32',
              4
            ),
            quantileParameters: fixture.input('quantiles', Uint32Array.of(2, 2), 'uint32', 2),
            parameters: fixture.parameters(
              'parameters',
              'uint32',
              getGPUPermutationParameterValues({
                seed: 1,
                permutationCount: reference.draws,
                significanceLevel: 0.05
              })
            ),
            maximumPermutations: reference.draws,
            maximumNeighbors: 2,
            weights: {
              offsets: fixture.input('offsets', Uint32Array.of(0, 2, 4, 6, 8), 'uint32', 5),
              neighbors: fixture.input(
                'neighbors',
                Uint32Array.of(1, 3, 0, 2, 1, 3, 0, 2),
                'uint32',
                8
              ),
              weights: fixture.input('weights', new Float32Array(8).fill(1), 'float32', 8)
            },
            output: {
              status: {overflow: inferenceOverflow.view},
              exceedances: fixture.output('exceedances', 'uint32', 4).view,
              pseudoPValues: pValues.view,
              membership: membership.view,
              boundaries: boundaries.view,
              validCount: validCount.view,
              classes: classes.view,
              invalidQuantile: invalidQuantile.view
            }
          })
        );
        fixture.run();
        expect(await fixture.readUint32(inferenceOverflow, 1)).toEqual([0]);
        const actual = await fixture.readFloat32(pValues, 4);
        expect(actual).toHaveLength(4);
        const expected =
          alternative === 'greater'
            ? reference.greater
            : alternative === 'two-sided'
              ? reference.twoSided
              : reference.legacy;
        actual.forEach((value, row) => {
          expect(Number.isFinite(value)).toBe(true);
          expect(value).toBeCloseTo(expected[row], 6);
        });
      } finally {
        fixture.destroy();
      }
    }
  }
});

import {addRegionalRasterChangeRecipe} from '../../../src/gpu-spatial-analysis/recipes/raster-summary-recipes';
it('W3-07 regional reductions suppress failed rasterization and recover unchanged area', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'w3-regional');
  try {
    const producer = fixture.output('producer', 'uint32', 1);
    const result = addRegionalRasterChangeRecipe(fixture.graph, {
      grid: {width: 2, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      producerOverflow: producer.view,
      zones: fixture.input('zones', Uint32Array.of(0, 0), 'uint32', 2),
      zoneCount: 1,
      cellAreas: fixture.input('areas', Float32Array.of(1, 1), 'float32', 2),
      threshold: fixture.input('threshold', Float32Array.of(1), 'float32', 1),
      change: {
        sliceCount: 2,
        slices: fixture.input('slices', Float32Array.of(3, 3, 4, 4), 'float32', 4),
        parameters: fixture.parameters(
          'parameters',
          'float32',
          Float32Array.of(0, 1, 0.000001, 0.05, 1)
        )
      }
    });
    const changed = publish(fixture, 'changed', result.outputs.changedArea);
    const mean = publish(fixture, 'mean', result.outputs.meanChanges);
    const loss = publish(
      fixture,
      'loss',
      result.status.stages.find(stage => stage.stage === 'change-summary')!.status
        .candidateOverflow!
    );
    for (const flag of [1, 0]) {
      producer.buffer.write(Uint32Array.of(flag));
      fixture.run();
      expect(await fixture.readUint32(loss, 1)).toEqual([flag]);
      for (const output of [changed, mean]) {
        const value = (await fixture.readFloat32(output, 1))[0];
        if (flag) expect(Number.isNaN(value)).toBe(true);
        else {
          expect(Number.isFinite(value)).toBe(true);
          expect(value).toBe(0);
        }
      }
    }
  } finally {
    fixture.destroy();
  }
});

it('S1 invalid live timestamp row is reported and suppresses the cube', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'review-invalid-stop');
  try {
    const result = addStopValueCubeRecipe(fixture.graph, {
      time: 'start',
      family: 'quadbin',
      resolution: 1,
      dictionary: fixture.input(
        'dictionary',
        computePointToCellOnCPU(Float32Array.of(-90, 30), {family: 'quadbin', resolution: 1}).cells,
        'uint32x2',
        1
      ),
      weights: {
        offsets: fixture.input('offsets', Uint32Array.of(0, 0), 'uint32', 2),
        neighbors: fixture.input('neighbors', Uint32Array.of(0), 'uint32', 1),
        weights: fixture.input('weights', Float32Array.of(0), 'float32', 1)
      },
      timestamps: fixture.input('timestamps', Uint32Array.of(0, 0), 'uint32x2', 1),
      temporalContext: {
        unit: 'second',
        epoch: 'unix',
        encoding: 'int64-words',
        axis: {kind: 'irregular'}
      },
      periods: {kind: 'fixed', origin: 0n, width: 10, count: 2},
      stops: {
        centroids: fixture.input('centroids', Float32Array.of(-90, 30, -90, 30), 'float32x2', 2),
        startRows: fixture.input('starts', Uint32Array.of(0, 99), 'uint32', 2),
        finishRows: fixture.input('finishes', Uint32Array.of(0, 99), 'uint32', 2),
        status: {count: fixture.input('count', Uint32Array.of(2), 'uint32', 1)}
      }
    });
    const invalid = publish(fixture, 'invalid', result.status.invalidCount);
    const counts = publish(fixture, 'counts', result.outputs.counts);
    fixture.run();
    expect(await fixture.readUint32(invalid, 1)).toEqual([1]);
    expect(await fixture.readUint32(counts, 2)).toEqual([0, 0]);
  } finally {
    fixture.destroy();
  }
});
