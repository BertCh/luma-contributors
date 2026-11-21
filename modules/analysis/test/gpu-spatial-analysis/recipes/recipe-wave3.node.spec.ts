import {addZonalDistributionRecipe} from '../../../src/gpu-spatial-analysis/recipes/zonal-distribution-recipe';
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {addExposureByZoneRecipe} from '../../../src/gpu-spatial-analysis/recipes/raster-summary-recipes';
import {
  addExcessRiskRecipe,
  addSpatialRateRecipe
} from '../../../src/gpu-spatial-analysis/recipes/rate-mapping-recipes';
import {addContiguityOrderCorrelogramRecipe} from '../../../src/gpu-spatial-analysis/recipes/contiguity-order-correlogram-recipe';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';

it('W3-07 passes raster producer loss to every dependent zonal reduction', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const values = createTransientView(graph, 'values', 'float32', 2);
    const producerOverflow = createTransientView(graph, 'loss', 'uint32', 1);
    const result = addExposureByZoneRecipe(graph, {
      grid: makeGPURasterMetadata({width: 2, height: 1}),
      hazard: {id: 'hazard', format: 'float32', storage: {kind: 'buffer', values}},
      population: {id: 'population', format: 'float32', storage: {kind: 'buffer', values}},
      weightMeaning: 'count',
      zones: createTransientView(graph, 'zones', 'uint32', 2),
      zoneCount: 1,
      threshold: createTransientView(graph, 'threshold', 'float32', 1),
      producerOverflow
    } as Parameters<typeof addExposureByZoneRecipe>[1]);
    const reductions = result.contributors.filter(
      contributor => contributor.constructor.name === 'GPURasterZonalStatistics'
    );
    expect(reductions).toHaveLength(3);
    for (const contributor of reductions)
      expect(
        (contributor as unknown as {props: {producerOverflow?: unknown}}).props.producerOverflow
      ).toBe(producerOverflow);
    for (const stage of result.status.stages.filter(stage => stage.stage.endsWith('-population')))
      expect(stage.status).toHaveProperty('candidateOverflow');
  } finally {
    device.destroy();
  }
});

it('CPL-15 exposes both causes even with only a chunk candidate loss', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const view = (id: string, length: number) => createTransientView(graph, id, 'uint32', length);
    const candidateOverflow = view('candidate', 1);
    const result = addSpatialRateRecipe(graph, {
      events: view('events', 2),
      populations: view('populations', 2),
      weights: {
        rowCount: 2,
        chunks: [
          {
            rowStart: 0,
            rowCount: 2,
            weights: {
              offsets: view('offsets', 3),
              neighbors: view('neighbors', 2),
              weights: createTransientView(graph, 'weights', 'float32', 2),
              status: {candidateOverflow}
            }
          }
        ]
      }
    });
    expect(result.status.stages[0].status.overflow).toBeDefined();
    expect(result.status.stages[0].status.candidateOverflow).toBeDefined();
  } finally {
    device.destroy();
  }
});

it('RCP-103 preserves order stages after distinct supplied weights evidence', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const candidateOverflow = createTransientView(graph, 'candidate', 'uint32', 1);
    const result = addContiguityOrderCorrelogramRecipe(graph, {
      values: createTransientView(graph, 'values', 'float32', 4),
      maximumOrder: 2,
      weights: {
        offsets: createTransientView(graph, 'offsets', 'uint32', 5),
        neighbors: createTransientView(graph, 'neighbors', 'uint32', 8),
        weights: createTransientView(graph, 'weights', 'float32', 8),
        status: {candidateOverflow}
      }
    });
    expect(result.status.stages.map(stage => stage.stage)).toEqual([
      'weights',
      'order-1',
      'order-2'
    ]);
    expect(result.status.stages[0].status.candidateOverflow).toBe(candidateOverflow);
  } finally {
    device.destroy();
  }
});

it('UCT-10 bare rate destinations preserve identity and reject the removed prop', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const view = (name: string) => createTransientView(graph, name, 'float32', 2);
    const output = view('output');
    const props = {
      events: view('events'),
      populations: view('population'),
      output
    };
    expect(addExcessRiskRecipe(graph, props).outputs.excessRisk).toBe(output);
    const retired = {...props, outputs: output};
    expect(() => addExcessRiskRecipe(graph, retired)).toThrow(/outputs was removed; use output/);
  } finally {
    device.destroy();
  }
});

it('NAM-01 keeps probability units and supplied quantile destination identity', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const probabilities = createTransientView(graph, 'probabilities', 'float32', 1);
    const quantiles = createTransientView(graph, 'quantiles', 'float32', 1);
    const result = addZonalDistributionRecipe(graph, {
      values: createTransientView(graph, 'values', 'float32', 2),
      source: {
        kind: 'zones',
        zones: createTransientView(graph, 'zones', 'uint32', 2),
        zoneCount: 1
      },
      probabilities,
      output: {quantiles}
    });
    expect(result.outputs.quantiles).toBe(quantiles);
    const grouped = result.contributors[0] as unknown as {
      props: {probabilities: unknown; columns: {output: {quantiles: unknown}}[]};
    };
    expect(grouped.props.probabilities).toBe(probabilities);
    expect(grouped.props.columns[0].output.quantiles).toBe(quantiles);
  } finally {
    device.destroy();
  }
});

import {addStopValueCubeRecipe} from '../../../src/gpu-spatial-analysis/recipes/index';
it('CMP-12 declares a dictionary-aligned stop cube with explicit units and loss', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'uint32x2' | 'float32' | 'float32x2'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `cube-${serial++}`, format, length);
    const candidateOverflow = view('uint32', 1);
    const counts = view('uint32', 4);
    const result = addStopValueCubeRecipe(graph, {
      time: 'finish',
      family: 'quadbin',
      resolution: 1,
      dictionary: view('uint32x2', 2),
      weights: {
        offsets: view('uint32', 3),
        neighbors: view('uint32', 2),
        weights: view('float32', 2)
      },
      timestamps: view('uint32x2', 3),
      temporalContext: {
        unit: 'second',
        epoch: 'unix',
        encoding: 'int64-words',
        axis: {kind: 'irregular'}
      },
      periods: {kind: 'fixed', origin: 0n, width: 10, count: 2},
      stops: {
        centroids: view('float32x2', 3),
        startRows: view('uint32', 3),
        finishRows: view('uint32', 3),
        status: {count: view('uint32', 1), candidateOverflow}
      },
      output: {counts}
    });
    expect(result.outputs.counts).toBe(counts);
    expect(result.status.stages[0].status.candidateOverflow).toBe(candidateOverflow);
  } finally {
    device.destroy();
  }
});
