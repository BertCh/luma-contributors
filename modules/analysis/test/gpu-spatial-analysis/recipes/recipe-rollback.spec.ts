// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {addSurfacePreparationRecipe} from '../../../src/gpu-spatial-analysis/recipes/surface-preparation-recipe';
import {addRasterZonalDistributionRecipe} from '../../../src/gpu-spatial-analysis/recipes/raster-summary-recipes';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from './recipe-harness';
import {GPUSpatialJoinPrepared} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared';
import {addZonalDistributionRecipe} from '../../../src/gpu-spatial-analysis/recipes/zonal-distribution-recipe';
import {addRateClusterMapRecipe} from '../../../src/gpu-spatial-analysis/recipes/rate-cluster-map-recipe';
import {addTrackQualityRecipe} from '../../../src/gpu-network/recipes/track-quality-recipe';
import {addExcessRiskRecipe} from '../../../src/gpu-spatial-analysis/recipes/rate-mapping-recipes';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

const GRID = {
  width: 2,
  height: 2,
  affine: [1, 0, 0, 0, 1, 0] as [number, number, number, number, number, number],
  pixelInterpretation: 'area' as const
};

it('R1 executes from the single captured Proxy population value', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'proxy-populations-review');
  try {
    const populations = fixture.input('populations', new Float32Array([2, 2, 2, 2]), 'float32', 4);
    const short = fixture.input('short', new Float32Array([2]), 'float32', 1);
    const output = fixture.output('risks', 'float32', 4);
    let reads = 0;
    const props = new Proxy(
      {
        id: 'retry',
        events: fixture.input('events', new Float32Array([1, 3, 1, 3]), 'float32', 4),
        populations,
        output: output.view
      },
      {
        get(target, key, receiver) {
          if (key === 'populations') return ++reads <= 2 ? populations : short;
          return Reflect.get(target, key, receiver);
        }
      }
    );
    addExcessRiskRecipe(fixture.graph, props);
    fixture.run();
    expect(reads).toBe(1);
    expect(await fixture.readFloat32(output, 4)).toEqual([0.5, 1.5, 0.5, 1.5]);
  } finally {
    fixture.destroy();
  }
});

it('surface preparation retries a failed second contributor with the same IDs and executes the corrected filter', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'surface-rollback');
  try {
    const values = fixture.input('input', new Float32Array([2, 4, 6, 8]), 'float32', 4);
    const output = fixture.output('output', 'float32', 4);
    const props = {
      id: 'retry',
      grid: GRID,
      input: {id: 'band', format: 'float32' as const, storage: {kind: 'buffer' as const, values}},
      maximumDistance: 1,
      outputs: {values: output.view}
    };
    expect(() =>
      addSurfacePreparationRecipe(fixture.graph, {...props, filter: {radius: 9}})
    ).toThrow(/radius/);
    addSurfacePreparationRecipe(fixture.graph, {...props, filter: {radius: 0}});
    fixture.run();
    expect(await fixture.readFloat32(output, 4)).toEqual([2, 4, 6, 8]);
  } finally {
    fixture.destroy();
  }
});

it('nested raster calibration and zonal statistics retry a late output failure without duplicate resources', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'nested-rollback');
  try {
    const values = fixture.input('input', new Float32Array([2, 4, 6, 8]), 'float32', 4);
    const zones = fixture.input('zones', new Uint32Array([0, 0, 1, 1]), 'uint32', 4);
    const medians = fixture.output('medians', 'float32', 2);
    const shortMedians = fixture.output('short-medians', 'float32', 1);
    const counts = fixture.output('counts', 'uint32', 2);
    const props = {
      id: 'retry',
      grid: GRID,
      input: {id: 'band', format: 'float32' as const, storage: {kind: 'buffer' as const, values}},
      zones,
      zoneCount: 2
    };
    expect(() =>
      addRasterZonalDistributionRecipe(fixture.graph, {
        ...props,
        outputs: {medians: shortMedians.view}
      })
    ).toThrow();
    const alias = fixture.graph.createDataView(values.buffer, {
      format: 'float32',
      length: 2,
      byteOffset: 4
    });
    expect(() =>
      addRasterZonalDistributionRecipe(fixture.graph, {...props, outputs: {medians: alias}})
    ).toThrow(/share|alias|overlap/);
    addRasterZonalDistributionRecipe(fixture.graph, {
      ...props,
      outputs: {medians: medians.view, validCounts: counts.view}
    });
    fixture.run();
    expect(await fixture.readFloat32(medians, 2)).toEqual([3, 7]);
    expect(await fixture.readUint32(counts, 2)).toEqual([2, 2]);
  } finally {
    fixture.destroy();
  }
});

it('R2 reuses a declared prepared polygon index when executing zonal distributions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'prepared-distribution-review');
  const polygons = {
    kind: 'polygons' as const,
    positions: fixture.input('polygon', new Float32Array([0, 0, 2, 0, 2, 2, 0, 2]), 'float32x2', 4),
    featureOffsets: fixture.input('features', new Uint32Array([0, 1]), 'uint32', 2),
    polygonOffsets: fixture.input('polygons', new Uint32Array([0, 1]), 'uint32', 2),
    ringOffsets: fixture.input('rings', new Uint32Array([0, 4]), 'uint32', 2)
  };
  const prepared = new GPUSpatialJoinPrepared({
    id: 'prepared',
    geometry: polygons,
    edgeSlabs: true
  });
  try {
    const median = fixture.output('median', 'float32', 1);
    const count = fixture.output('count', 'uint32', 1);
    const props = {
      id: 'retry',
      values: fixture.input('values', new Float32Array([2, 6]), 'float32', 2),
      source: {
        kind: 'points' as const,
        join: {
          points: {
            kind: 'points' as const,
            positions: fixture.input(
              'points',
              new Float32Array([0.25, 0.25, 0.5, 0.5]),
              'float32x2',
              2
            )
          },
          polygons,
          prepared,
          candidateCapacity: 8
        }
      },
      outputs: {medians: median.view, validCounts: count.view}
    };
    expect(() => addZonalDistributionRecipe(fixture.graph, props)).toThrow(/prepared.*declared/);
    fixture.graph.add(prepared);
    addZonalDistributionRecipe(fixture.graph, props);
    fixture.run();
    expect(await fixture.readFloat32(median, 1)).toEqual([4]);
    expect(await fixture.readUint32(count, 1)).toEqual([2]);
  } finally {
    fixture.destroy();
    prepared.destroy();
  }
});

it('R3 R4 R5 validate color ports and independent rates, then execute with a frozen parameter wrapper', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'color-ports-review');
  try {
    const positions = [
      0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 2, 0, 2, 1, 1, 1, 0, 1, 1, 1, 1, 2, 0, 2, 1, 1, 2, 1, 2, 2, 1, 2
    ];
    const rates = fixture.output('rates', 'float32', 4);
    const scores = fixture.output('scores', 'float32', 4);
    const colors = fixture.output('colors', 'uint32', 8);
    const quadrants = fixture.output('quadrants', 'uint32', 4);
    const colorView = fixture.graph.createDataView(colors.view.buffer, {
      format: 'uint32',
      length: 4,
      byteOffset: 4
    });
    const short = fixture.graph.createDataView(colors.view.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 4
    });
    const props = {
      id: 'retry',
      events: fixture.input('events', new Float32Array([2, 2, 2, 2]), 'float32', 4),
      populations: fixture.input('populations', new Float32Array([10, 10, 10, 10]), 'float32', 4),
      polygons: {
        kind: 'polygons' as const,
        positions: fixture.input('positions', new Float32Array(positions), 'float32x2', 16),
        featureOffsets: fixture.input('features', new Uint32Array([0, 1, 2, 3, 4]), 'uint32', 5),
        polygonOffsets: fixture.input('polygons', new Uint32Array([0, 1, 2, 3, 4]), 'uint32', 5),
        ringOffsets: fixture.input('rings', new Uint32Array([0, 4, 8, 12, 16]), 'uint32', 5)
      },
      parameters: Object.freeze(fixture.parameterBuffer('parameters', 'float32', 8)),
      neighborCapacity: 16,
      pairCapacity: 16,
      palette: fixture.input('palette', new Uint32Array([11, 22, 33, 44, 55]), 'uint32', 5),
      outputs: {
        smoothedRates: rates.view,
        zScores: scores.view,
        colors: colorView,
        quadrants: quadrants.view
      }
    };
    expect(() =>
      addRateClusterMapRecipe(fixture.graph, {...props, outputs: {...props.outputs, colors: short}})
    ).toThrow(/colors/);
    const shortQuadrants = fixture.output('short-quadrants', 'uint32', 1);
    expect(() =>
      addRateClusterMapRecipe(fixture.graph, {
        ...props,
        outputs: {...props.outputs, quadrants: shortQuadrants.view}
      })
    ).toThrow(/quadrants/);
    const shortPalette = fixture.input('short-palette', new Uint32Array([11]), 'uint32', 1);
    expect(() => addRateClusterMapRecipe(fixture.graph, {...props, palette: shortPalette})).toThrow(
      /palette/
    );
    expect(() =>
      addRateClusterMapRecipe(fixture.graph, {
        ...props,
        outputs: {...props.outputs, zScores: rates.view}
      })
    ).toThrow(/share|alias|overlap/);
    addRateClusterMapRecipe(fixture.graph, props);
    fixture.run();
    for (const rate of await fixture.readFloat32(rates, 4)) {
      expect(Number.isFinite(rate)).toBe(true);
      expect(rate).toBeCloseTo(0.2, 6);
    }
    for (const score of await fixture.readFloat32(scores, 4)) expect(score).toBeNaN();
    expect(await fixture.readUint32(colors, 8)).toEqual([0, 11, 11, 11, 11, 0, 0, 0]);
    expect(await fixture.readUint32(quadrants, 4)).toEqual([0, 0, 0, 0]);
  } finally {
    fixture.destroy();
  }
});

it('R6 retries unequal track pairs and keeps metrics before flags during GPU execution', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'track-review');
  try {
    const pairB = fixture.input('pair-b', new Uint32Array([0, 0]), 'uint32', 2);
    const short = fixture.input('short', new Uint32Array([0]), 'uint32', 1);
    const props = {
      id: 'retry',
      trajectory: {
        positions: fixture.input(
          'positions',
          new Float32Array([0, 0, 1, 0, 2, 0, 20, 0]),
          'float32x2',
          4
        ),
        timestamps: fixture.input('times', new Float32Array([0, 0, 1, 10]), 'float32', 4),
        trackOffsets: fixture.input('offsets', new Uint32Array([0, 4]), 'uint32', 2)
      },
      thresholds: fixture.input('thresholds', new Float32Array([1.5, 5]), 'float32', 2),
      similarity: {
        pairA: fixture.input('pair-a', new Uint32Array([0, 0]), 'uint32', 2),
        pairB,
        maximumFrechetVertexCount: 8
      }
    };
    expect(() =>
      addTrackQualityRecipe(fixture.graph, {
        ...props,
        similarity: {...props.similarity, pairB: short}
      })
    ).toThrow(/pairA and pairB/);
    const result = addTrackQualityRecipe(fixture.graph, props);
    const flags = fixture.output('flags', 'uint32', 4);
    fixture.graph.add(
      createWGSLKernelNode(fixture.graph, {
        id: 'publish',
        operation: 'review',
        invocationCount: 4,
        bindings: [
          {name: 'source', view: result.outputs.flags, type: 'u32', access: 'read'},
          {name: 'output', view: flags.view, type: 'u32', access: 'read_write'}
        ],
        body: 'output[outputOffset + index] = source[sourceOffset + index];'
      })
    );
    fixture.run();
    expect(await fixture.readUint32(flags, 4)).toEqual([0, 2, 0, 12]);
  } finally {
    fixture.destroy();
  }
});
