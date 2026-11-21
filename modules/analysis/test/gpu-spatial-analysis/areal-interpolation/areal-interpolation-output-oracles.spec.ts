// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUArealInterpolation} from '../../../src/gpu-spatial-analysis/areal-interpolation/gpu-areal-interpolation';
import {GPUSpatialLag} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-lag';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {addChangeOfSupportRecipe} from '../../../src/gpu-spatial-analysis/recipes/change-of-support-recipe';
import {getGPUPolygonRasterizationParameterValues} from '../../../src/raster-analysis/polygon-rasterization/index';
import {createArealRecipeRectangleFixture} from './areal-interpolation-output-oracle-fixtures';

for (const capacity of [1, 2]) {
  it(`writes the recipe caller nested overflow and suppresses incomplete lag at capacity ${capacity}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOutputOracleFixture(device);
    const {graph, input, output} = fixture;
    const rectangle = createArealRecipeRectangleFixture();
    const {grid} = rectangle;
    const overflow = output('caller-overflow', 'uint32', 1),
      lag = output('lag', 'float32', 1);
    const sourceRaster = output('source-raster', 'uint32', 8);
    const targetRaster = output('target-raster', 'uint32', 8);
    const sourceRasterOverflow = output('source-raster-overflow', 'uint32', 1);
    const targetRasterOverflow = output('target-raster-overflow', 'uint32', 1);
    const sourceCrossingCount = output('source-crossings', 'uint32', 1);
    const targetCrossingCount = output('target-crossings', 'uint32', 1);
    const requiredPairCount = output('required-pairs', 'uint32', 1);
    const zones = (name: string, system: typeof rectangle.source) => ({
      kind: 'polygons' as const,
      positions: input(`${name}-positions`, 'float32x2', system.positions),
      featureOffsets: input(`${name}-features`, 'uint32', system.featureOffsets),
      polygonOffsets: input(`${name}-polygons`, 'uint32', system.polygonOffsets),
      ringOffsets: input(`${name}-rings`, 'uint32', system.ringOffsets),
      crossingCapacity: system.crossingCapacity
    });
    addChangeOfSupportRecipe(graph, {
      grid,
      pairCapacity: capacity,
      parameters: input('parameters', 'float32', getGPUPolygonRasterizationParameterValues(grid)),
      source: zones('source', rectangle.source),
      target: zones('target', rectangle.target),
      sourceValues: input('values', 'float32', new Float32Array([10, 20])),
      scratch: {sourceZones: sourceRaster, targetZones: targetRaster},
      output: {
        status: {
          requiredCount: requiredPairCount,
          rasterOverflow: [sourceRasterOverflow, targetRasterOverflow],
          rasterRequiredCount: [sourceCrossingCount, targetCrossingCount]
        },
        extensiveWeights: {
          offsets: output('offsets', 'uint32', 2),
          neighbors: output('neighbors', 'uint32', capacity),
          weights: output('weights', 'float32', capacity),
          status: {overflow}
        },
        extensiveValues: lag
      }
    });
    const compiled = graph.compile();
    try {
      const actual = await fixture.submitTwice(compiled, async () => ({
        sourceRaster: await fixture.readRows([sourceRaster]),
        targetRaster: await fixture.readRows([targetRaster]),
        rasterOverflow: await fixture.readRows([sourceRasterOverflow, targetRasterOverflow]),
        crossings: await fixture.readRows([sourceCrossingCount, targetCrossingCount]),
        requiredPairs: await fixture.readRows([requiredPairCount]),
        overflow: await fixture.readRows([overflow]),
        lag: await fixture.readRows([lag])
      }));
      // Hand-enumerated unit-cell centres on [0,4] x [0,2]. Each rectangle
      // has two vertical edges crossing two rows: source 8 records, target 4.
      expect(actual.sourceRaster).toEqual([0, 0, 1, 1, 0, 0, 1, 1]);
      expect(actual.targetRaster).toEqual([
        0xffffffff, 0, 0, 0xffffffff, 0xffffffff, 0, 0, 0xffffffff
      ]);
      expect(actual.rasterOverflow).toEqual([0, 0]);
      expect(actual.crossings).toEqual([8, 4]);
      expect(actual.requiredPairs).toEqual([2]);
      expect(actual.overflow).toEqual([capacity === 1 ? 1 : 0]);
      // Independent Shapely rectangle shares are 1/2 per source: complete sum 15.
      if (capacity === 1) expect(Number.isNaN(actual.lag[0])).toBe(true);
      else expect(actual.lag).toEqual([15]);
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}

const PRODUCED_STATUS_FIELDS = ['count', 'requiredCount', 'candidateOverflow'] as const;

for (const optional of [undefined, 'distances', ...PRODUCED_STATUS_FIELDS]) {
  it(`writes supported nested ${optional ?? 'no optional metadata'} and Shapely rectangle weights`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOutputOracleFixture(device);
    const {input, output, graph} = fixture;
    const offsets = output('offsets', 'uint32', 2);
    const neighbors = output('neighbors', 'uint32', 2);
    const weights = output('weights', 'float32', 2);
    const overflow = output('overflow', 'uint32', 1);
    const metadata = optional
      ? output('metadata', 'uint32', optional === 'distances' ? 2 : 1, {
          preserveOnReplay: optional === 'distances'
        })
      : undefined;
    // Distances remain caller metadata. Supported nested status words have a writer on every
    // submission; unsupported diagnostic destinations are rejected in the Node binding fixture.
    const distances =
      optional === 'distances'
        ? graph.createDataView(metadata!.buffer, {
            format: 'float32',
            length: 2,
            byteOffset: metadata!.byteOffset
          })
        : undefined;
    graph.add(
      new GPUArealInterpolation({
        grid: makeGPURasterMetadata({width: 4, height: 2}),
        sourceZones: input('source', 'uint32', new Uint32Array([0, 0, 1, 1, 0, 0, 1, 1])),
        targetZones: input(
          'target',
          'uint32',
          new Uint32Array([0xffffffff, 0, 0, 0xffffffff, 0xffffffff, 0, 0, 0xffffffff])
        ),
        sourceZoneCount: 2,
        targetZoneCount: 1,
        output: {
          weights: {
            offsets,
            neighbors,
            weights,
            distances,
            status: optional && optional !== 'distances' ? {[optional]: metadata} : undefined
          },
          status: {overflow}
        }
      })
    );
    const compiled = graph.compile();
    try {
      const actual = await fixture.submitTwice(compiled, async () => ({
        offsets: await fixture.readRows([offsets]),
        neighbors: await fixture.readRows([neighbors]),
        weights: await fixture.readRows([weights]),
        overflow: await fixture.readRows([overflow]),
        metadata: metadata ? await fixture.readRows([metadata]) : []
      }));
      // Shapely oracle beside this spec: each 4-unit source overlaps the target by 2.
      expect(actual.offsets).toEqual([0, 2]);
      expect(actual.neighbors).toEqual([0, 1]);
      expect(actual.weights).toEqual([0.5, 0.5]);
      expect(actual.overflow).toEqual([0]);
      expect(actual.metadata).toEqual(
        optional === 'distances'
          ? [0xa5a5a5a5, 0xa5a5a5a5]
          : optional === 'candidateOverflow'
            ? [0]
            : optional
              ? [2]
              : []
      );
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}

for (const capacity of [1, 2]) {
  it(`propagates canonical areal overflow to spatial lag at capacity ${capacity}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOutputOracleFixture(device);
    const {graph, input, output} = fixture;
    const overflow = output('overflow', 'uint32', 1);
    const weights = {
      offsets: output('offsets', 'uint32', 2),
      neighbors: output('neighbors', 'uint32', capacity),
      weights: output('weights', 'float32', capacity),
      status: {overflow}
    };
    const lag = output('lag', 'float32', 1);
    graph.add(
      new GPUArealInterpolation({
        grid: makeGPURasterMetadata({width: 4, height: 2}),
        sourceZones: input('source', 'uint32', new Uint32Array([0, 0, 1, 1, 0, 0, 1, 1])),
        targetZones: input(
          'target',
          'uint32',
          new Uint32Array([0xffffffff, 0, 0, 0xffffffff, 0xffffffff, 0, 0, 0xffffffff])
        ),
        sourceZoneCount: 2,
        targetZoneCount: 1,
        output: {weights, status: {overflow}}
      })
    );
    graph.add(
      new GPUSpatialLag({
        weights,
        values: input('values', 'float32', new Float32Array([10, 20])),
        output: lag
      })
    );
    const compiled = graph.compile();
    try {
      const actual = await fixture.submitTwice(compiled, async () => ({
        overflow: await fixture.readRows([overflow]),
        lag: await fixture.readRows([lag])
      }));
      expect(actual.overflow).toEqual([capacity === 1 ? 1 : 0]);
      // Shapely rectangle shares are 1/2 for each source. Missing the second
      // source must suppress 5, rather than passing it off as the complete sum 15.
      if (capacity === 1) expect(Number.isNaN(actual.lag[0])).toBe(true);
      else expect(actual.lag).toEqual([15]);
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}
