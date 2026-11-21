// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPURasterWatershedSegmentation} from '../../../src/raster-analysis/gpu-raster-watershed-segmentation';
import {GPURasterRegionGraph} from '../../../src/raster-analysis/remote-sensing/gpu-raster-region-graph';
import {addRasterRegionGraphTiles} from '../../../src/raster-analysis/remote-sensing/raster-region-tiles';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid';
import {createRemoteSensingHarness} from './remote-sensing-test-utils';

for (const tiled of [false, true]) {
  for (const maximumSteps of [1, 3]) {
    it(`CPL-05 watershed evidence reaches region adjacency (tiled=${tiled}, steps=${maximumSteps})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = createRemoteSensingHarness(device);
      const {graph} = harness;
      const grid = makeGPURasterMetadata({width: 3, height: 1});
      const values = harness.input([0, 0, 0]);
      const labels = harness.output('labels', 'uint32', 3);
      const converged = harness.output('converged', 'uint32', 1);
      graph.add(
        new GPURasterWatershedSegmentation({
          grid,
          input: {id: 'elevation', format: 'float32', storage: {kind: 'buffer', values}},
          markers: harness.mask([1, 0, 2]),
          maximumSteps,
          output: {labels, status: {converged}}
        })
      );
      const props = {
        grid,
        regionCount: 3,
        mergeThreshold: 0,
        maximumIterations: 3,
        output: {
          adjacency: harness.output('adjacency', 'uint32', 9),
          weights: harness.output('weights', 'float32', 9),
          mapping: harness.output('mapping', 'uint32', 3),
          status: {
            iterationLimitReached: harness.output('iterationLimitReached', 'uint32', 1),
            complete: harness.output('complete', 'uint32', 1)
          }
        }
      };
      if (tiled) {
        addRasterRegionGraphTiles(graph, {
          ...props,
          tiles: [
            {
              grid: makeGPURasterMetadata({width: 2, height: 1}),
              origin: [0, 0],
              labels: graph.createDataView(labels.buffer, {format: 'uint32', length: 2}),
              values: graph.createDataView(values.buffer, {format: 'float32', length: 2}),
              labelsComplete: converged
            },
            {
              grid: makeGPURasterMetadata({width: 1, height: 1}),
              origin: [2, 0],
              labels: graph.createDataView(labels.buffer, {
                format: 'uint32',
                byteOffset: 8,
                length: 1
              }),
              values: graph.createDataView(values.buffer, {
                format: 'float32',
                byteOffset: 8,
                length: 1
              }),
              labelsComplete: harness.mask([1])
            }
          ]
        });
      } else
        graph.add(new GPURasterRegionGraph({...props, labels, values, labelsComplete: converged}));
      // The recipe already declared its nodes; use an empty producer solely to submit/read the harness.
      const result = await harness.run({getCommandNodes: () => []});
      const complete = Number(maximumSteps === 3);
      expect(result['labels']).toEqual(complete ? [1, 1, 2] : [1, 0xfffffffe, 2]);
      expect(result['converged']).toEqual([complete]);
      expect(result['complete']).toEqual([complete]);
      expect(result['iterationLimitReached']).toEqual([0]);
      // Hand-counted: one 1--2 edge once the middle cell belongs to basin 1.
      expect(result['adjacency']).toEqual([0, 0, 0, 0, 0, complete, 0, complete, 0]);
      expect(result['mapping']).toEqual(complete ? [0, 1, 1] : [0, 1, 2]);
    });
  }
}

for (const [name, words] of [
  ['recipe-zero', [0, 1, 1]],
  ['first-tile-zero', [1, 0, 1]],
  ['second-tile-zero', [1, 1, 0]],
  ['nonbinary-positive', [2, 4, 8]]
] as const) {
  it(`review S1 combines every completeness position independently (${name})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRemoteSensingHarness(device);
    const grid = makeGPURasterMetadata({width: 3, height: 1});
    addRasterRegionGraphTiles(harness.graph, {
      grid,
      regionCount: 3,
      mergeThreshold: 0,
      maximumIterations: 3,
      labelsComplete: harness.mask([words[0]]),
      tiles: [
        {
          grid: makeGPURasterMetadata({width: 2, height: 1}),
          origin: [0, 0],
          labels: harness.mask([1, 1]),
          values: harness.input([0, 0]),
          labelsComplete: harness.mask([words[1]])
        },
        {
          grid: makeGPURasterMetadata({width: 1, height: 1}),
          origin: [2, 0],
          labels: harness.mask([2]),
          values: harness.input([0]),
          labelsComplete: harness.mask([words[2]])
        }
      ],
      output: {
        adjacency: harness.output('adjacency', 'uint32', 9),
        weights: harness.output('weights', 'float32', 9),
        mapping: harness.output('mapping', 'uint32', 3),
        status: {
          complete: harness.output('complete', 'uint32', 1),
          iterationLimitReached: harness.output('limit', 'uint32', 1)
        }
      }
    });
    const result = await harness.run({getCommandNodes: () => []});
    expect(result['complete']).toEqual([Number(words.every(word => word !== 0))]);
    expect(result['limit']).toEqual([0]);
    expect(result['adjacency']).toEqual([0, 0, 0, 0, 0, 1, 0, 1, 0]);
    expect(result['mapping']).toEqual([0, 1, 1]);
  });
}

for (const route of ['dense', 'boruvka', 'adaptive'] as const) {
  it(`review earlier fix ignores extreme padding when choosing the accumulator scale (${route})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRemoteSensingHarness(device);
    const result = await harness.run(
      new GPURasterRegionGraph({
        grid: makeGPURasterMetadata({width: 2, height: 1}),
        regionCount: 2,
        maximumIterations: 1,
        mergeThreshold: 0,
        mergeMode: route === 'adaptive' ? 'adaptive' : 'fixed',
        searchStructure: route === 'adaptive' ? 'dense' : route,
        labels: harness.mask([0, 1]),
        values: harness.input([0, 1, 1e34]),
        output: {
          adjacency: harness.output('adjacency', 'uint32', 4),
          weights: harness.output('weights', 'float32', 4),
          mapping: harness.output('mapping', 'uint32', 2),
          status: {iterationLimitReached: harness.output('limit', 'uint32', 1)}
        }
      })
    );
    // Each cell is its own region; singleton means are exactly 0 and 1. Padding is outside the grid.
    expect(result['adjacency']).toEqual([0, 1, 1, 0]);
    expect(Number.isFinite(result['weights'][1])).toBe(true);
    expect(result['weights'][1]).toBe(1);
    expect(Number.isFinite(result['weights'][2])).toBe(true);
    expect(result['weights'][2]).toBe(1);
    expect(result['mapping']).toEqual([0, 1]);
    expect(result['limit']).toEqual([0]);
  });
}
