// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {getRasterPipelines} from '../../gpu-raster/raster-pipeline-count';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {
  GPUTerrainViewshed,
  GPUTerrainLineOfSight,
  GPUTerrainCumulativeViewshed,
  GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH,
  getGPUTerrainViewshedParameterValues,
  getGPUTerrainLineOfSightParameterValues
} from '../../../src/gpu-terrain/terrain-analysis/index';
import {
  getRasterGridGeoreference,
  getRasterGroundScale
} from '../../../src/analysis-core/contracts/raster-grid';
import {
  computeTerrainViewshed,
  computeTerrainSightLine,
  computeTerrainViewshedWithTolerance
} from './terrain-analysis-oracle';
import * as kernelNodes from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createBand} from '../terrain-test-utils';

const KERNEL_CASES = [
  ['viewshed', 'march', false],
  ['viewshed', 'march', true],
  ['viewshed', 'pyramid', false],
  ['viewshed', 'pyramid', true],
  ['line-of-sight', 'march', false],
  ['line-of-sight', 'pyramid', false],
  ['cumulative', 'march', false],
  ['cumulative', 'pyramid', false]
] as const;

function captureKernel(
  [kind, traversal, tolerance]: (typeof KERNEL_CASES)[number],
  overrides: Record<string, unknown> = {}
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const capture = vi.spyOn(kernelNodes, 'createWGSLKernelNode');
  try {
    const common = {
      grid: makeGPURasterMetadata({width: 9, height: 7, cellSize: 30}),
      elevation: createBand(graph, 'elevation', 63),
      parameters: createTransientView(graph, 'parameters', 'float32', 12),
      traversal,
      ...overrides
    };
    const contributor =
      kind === 'viewshed'
        ? new GPUTerrainViewshed({
            ...common,
            toleranceParameters: tolerance
              ? createTransientView(graph, 'tolerance', 'float32', 4)
              : undefined,
            output: {visibility: createTransientView(graph, 'visibility', 'uint32', 63)}
          })
        : kind === 'line-of-sight'
          ? new GPUTerrainLineOfSight({
              ...common,
              pairs: createTransientView(graph, 'pairs', 'float32x4', 3),
              output: {
                visibility: createTransientView(graph, 'visibility', 'uint32', 3),
                clearance: createTransientView(graph, 'clearance', 'float32', 3)
              }
            })
          : new GPUTerrainCumulativeViewshed({
              ...common,
              observers: createTransientView(graph, 'observers', 'float32x2', 3),
              output: {
                visibleCount: createTransientView(graph, 'visible', 'uint32', 63),
                marginalCount: createTransientView(graph, 'marginal', 'uint32', 63)
              }
            });
    const nodes = contributor.getCommandNodes(graph);
    const {declarations, body, variant} = capture.mock.calls.at(-1)![1];
    const source = getRasterPipelines(nodes).at(-1)!.source;
    return {declarations, body, variant, source};
  } finally {
    capture.mockRestore();
    device.destroy();
  }
}

it('sight-line uniform declarations, bodies and variants retain the reviewed validity and sampling snapshot', () => {
  const hash = (source: string | undefined) =>
    createHash('sha256')
      .update(source ?? '')
      .digest('hex');
  const snapshots = KERNEL_CASES.map(kernelCase => {
    const kernel = captureKernel(kernelCase);
    expect(kernel).toEqual(captureKernel(kernelCase, {cellSizeMode: 'uniform'}));
    expect(kernel.declarations).not.toContain('getRayGroundCell');
    expect(kernel.body).not.toContain('getRayGroundCell');
    // The old fingerprints predated the approved finite-height and positive-weight fixes.
    // Keep full declaration/body fingerprints, and explicitly pin their sampling contracts:
    // zero-weight corners cannot affect interpolation, and integer lattice hits avoid reciprocal
    // rounding. These are policy assertions, not values inferred from a GPU readback.
    expect(kernel.declarations).toContain('if (weights.x > 0.0)');
    expect(kernel.declarations).toContain('if (weights.y > 0.0)');
    expect(kernel.declarations).toContain('if (weights.z > 0.0)');
    expect(kernel.declarations).toContain('if (weights.w > 0.0)');
    expect(kernel.declarations).toContain(
      'if (fraction == 0.0 || first == second) { return first; }'
    );
    expect(kernel.declarations).toContain('if (integerDelta % divisor == 0u)');
    if (kernelCase[0] === 'viewshed' || kernelCase[0] === 'cumulative') {
      expect(kernel.body).toContain('!isFiniteValue(observerHeight)');
      expect(kernel.body).toContain('!isFiniteValue(targetHeight)');
    }
    if (kernelCase[0] !== 'viewshed' || kernelCase[1] === 'pyramid' || kernelCase[2]) {
      expect(kernel.declarations).toContain(
        'if (!isFiniteValue(eye) || !isFiniteValue(targetTop))'
      );
    }
    return {
      variant: kernel.variant,
      declarations: hash(kernel.declarations),
      body: hash(kernel.body)
    };
  });
  expect(snapshots).toMatchInlineSnapshot(`
    [
      {
        "body": "d538b04ef8326092911ab7933a51084f05afcc4c820e43d336f95167664f036c",
        "declarations": "bef2d61ccb28243d99fae80f034561b011309221ee1750a7eae0c0fdf4cd54e6",
        "variant": "line-of-sight",
      },
      {
        "body": "7aa5ed301c35ced7447e480e1966ee71b2f3cee85eec4a523ec4ffc107907010",
        "declarations": "8dd309be100c0624b2f68b114014709c506f44a16d3fa135a050fd5419adef04",
        "variant": "tolerance",
      },
      {
        "body": "1f807151993f7835a8aafc9795e183f5846865b406269c5b6c7102b29cb150c8",
        "declarations": "d67ab8863cda2a547872d66d26e69de4d30516754240d63e7f3fab6f645050bb",
        "variant": "pyramid",
      },
      {
        "body": "7aa5ed301c35ced7447e480e1966ee71b2f3cee85eec4a523ec4ffc107907010",
        "declarations": "d67ab8863cda2a547872d66d26e69de4d30516754240d63e7f3fab6f645050bb",
        "variant": "pyramid",
      },
      {
        "body": "79e8d4029a7f0ff1b1f0619d46413b04cfc5a3a94b539ad222710a2cc79cbadc",
        "declarations": "bcd5f4ca2b85a0fe4e9e6f2b6696f2b1c9278b1e8e57c836bd8f6074c5dcad28",
        "variant": "march",
      },
      {
        "body": "79e8d4029a7f0ff1b1f0619d46413b04cfc5a3a94b539ad222710a2cc79cbadc",
        "declarations": "f3160a287ff86fb419f54fb4b59527f963a55f17fcee1bea56d05b82da319354",
        "variant": "pyramid",
      },
      {
        "body": "043a29e482c192ac5c1135c8c31879e2055d4bb2999b03dfd1c7a3d7f571ce74",
        "declarations": "1379cb5be0154fdd50f9c5ca60359a4801af145b55000c8361907657a82506e4",
        "variant": "march",
      },
      {
        "body": "043a29e482c192ac5c1135c8c31879e2055d4bb2999b03dfd1c7a3d7f571ce74",
        "declarations": "2111f055f7291883d91bca636968edf5bb769929125dca1eaa17f72b7c273424",
        "variant": "pyramid",
      },
    ]
  `);
});

it('uniform sight-line validity follows hand-worked endpoint and ridge fixtures', () => {
  const grid = makeGPURasterMetadata({width: 3, height: 1, cellSize: 1});
  const settings = getGPUTerrainViewshedParameterValues(
    {observer: [0, 0], observerHeight: 0},
    grid
  );
  // Integer centers have one nonzero bilinear weight. The unused NaN neighbor cannot invalidate
  // the first two cells: the eye sees itself and the adjacent 10 m cell; the NaN target is noData.
  const terrain = [0, 10, NaN];
  expect(computeTerrainViewshed(terrain, undefined, 3, 1, settings)).toEqual([1, 1, 3]);
  expect(
    computeTerrainViewshedWithTolerance(terrain, undefined, 3, 1, settings, [0, 0, 0, 0]).map(
      result => result.code
    )
  ).toEqual([1, 1, 3]);
  const pairSettings = getGPUTerrainLineOfSightParameterValues({observerHeight: 0}, grid);
  expect(computeTerrainSightLine(terrain, undefined, 3, 1, [0, 0], [1, 0], pairSettings).code).toBe(
    1
  );
  // At column 1.5, both 10 m and NaN have positive weights, so this target is noData.
  expect(
    computeTerrainSightLine(terrain, undefined, 3, 1, [0, 0], [1.5, 0], pairSettings).code
  ).toBe(3);
  // A 2 m ridge at x=1 lies above the zero-height line joining x=0 and x=2.
  expect(computeTerrainViewshed([0, 2, 0], undefined, 3, 1, settings)).toEqual([1, 1, 0]);
  settings[2] = NaN;
  expect(computeTerrainViewshed(terrain, undefined, 3, 1, settings)).toEqual([3, 3, 3]);
  expect(
    computeTerrainViewshedWithTolerance(terrain, undefined, 3, 1, settings, [0, 0, 0, 0]).map(
      result => result.code
    )
  ).toEqual([3, 3, 3]);
});

it.each([
  'viewshed',
  'line-of-sight',
  'cumulative'
] as const)('%s validates cellSizeMode and rejects rotated non-uniform grids', kind => {
  const kernelCase = [kind, 'march', false] as const;
  expect(() => captureKernel(kernelCase, {cellSizeMode: 'polar'})).toThrow(
    'cellSizeMode must be uniform, web-mercator, or geographic'
  );
  const grid = makeGPURasterMetadata({width: 9, height: 7});
  grid.affine = [1, 0.1, 0, 0.1, -1, 60];
  for (const cellSizeMode of ['geographic', 'web-mercator'] as const) {
    expect(() => captureKernel(kernelCase, {grid, cellSizeMode})).toThrow(
      `${cellSizeMode} cell sizes need an axis-aligned grid`
    );
  }
  expect(() => captureKernel(kernelCase, {grid})).not.toThrow();
});

it('sight-line packers always write area and point grid row edges in world units', () => {
  expect(GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH).toBe(10);
  for (const pixelInterpretation of ['area', 'point'] as const) {
    for (const rowStep of [-0.25, 0.25, -30]) {
      const grid = makeGPURasterMetadata({width: 9, height: 7});
      grid.pixelInterpretation = pixelInterpretation;
      grid.affine = [0.25, 0, 10, 0, rowStep, 60];
      const northEdge = 60 + (pixelInterpretation === 'point' ? -0.5 * rowStep : 0);
      const edges = [northEdge, northEdge + 7 * rowStep];
      const viewshed = getGPUTerrainViewshedParameterValues({observer: [1, 2]}, grid);
      const sightLine = getGPUTerrainLineOfSightParameterValues({}, grid);
      expect(viewshed).toHaveLength(10);
      expect(sightLine).toHaveLength(12);
      expect(Array.from(viewshed.slice(8))).toEqual(edges);
      expect(Array.from(sightLine.slice(10))).toEqual(edges);
      expect(() =>
        getGPUTerrainViewshedParameterValues({observer: [1, 2]}, grid, new Float32Array(8))
      ).toThrow(/must hold 10/);
    }
  }
});

it('every non-uniform sight-line kernel uses the middle row and mode-specific variant', () => {
  for (const kernelCase of KERNEL_CASES) {
    const uniform = captureKernel(kernelCase);
    for (const cellSizeMode of ['geographic', 'web-mercator'] as const) {
      const kernel = captureKernel(kernelCase, {cellSizeMode});
      const edgeSlot = kernelCase[0] === 'viewshed' ? 8 : 10;
      const targetRow = kernelCase[0] === 'line-of-sight' ? 'targetPosition.y' : 'f32(row)';
      expect(kernel.variant).toBe(`${uniform.variant}-${cellSizeMode}`);
      expect(() => new WgslReflect(kernel.source)).not.toThrow();
      expect(kernel.declarations).toContain('fn getRayGroundCell(');
      expect(kernel.declarations).toContain(
        'mix(northEdge, southEdge, rowCoordinate / f32(HEIGHT))'
      );
      expect(kernel.declarations).toContain(
        cellSizeMode === 'geographic' ? 'sin(latitude)' : 'tanh(mercatorY)'
      );
      expect(kernel.body).toContain(
        `settings[settingsOffset + ${edgeSlot}u], settings[settingsOffset + ${edgeSlot + 1}u], 0.5 * (observer.y + ${targetRow}) + 0.5)`
      );
      expect(kernel.declarations).not.toContain('fn getPixelDirection');
      expect(kernel.declarations).not.toContain('fn getGroundCellSize');
    }
  }
});

it('the geographic sight-line oracle matches WGS84 meters per degree at the equator', () => {
  const grid = makeGPURasterMetadata({width: 7, height: 3});
  grid.affine = [1 / 3600, 0, 0, 0, -1 / 3600, 1.5 / 3600];
  const georeference = getRasterGridGeoreference(grid);
  const terrain = new Float64Array(21);
  const settings = getGPUTerrainLineOfSightParameterValues({observerHeight: 1}, grid);
  const result = computeTerrainSightLine(
    terrain,
    undefined,
    7,
    3,
    [0, 1],
    [4, 1],
    settings,
    undefined,
    {cellSizeMode: 'geographic', georeference}
  );
  // Independent WGS84 equatorial meters/degree: 2*pi*a/360, and that times (1-e^2).
  const metersPerDegree = [111319.49079327358, 110574.27582159436];
  expect(getRasterGroundScale('geographic', 0)[0]).toBeCloseTo(metersPerDegree[0], 8);
  expect(getRasterGroundScale('geographic', 0)[1]).toBeCloseTo(metersPerDegree[1], 8);
  expect(-1 / result.targetSlope).toBeCloseTo((4 * metersPerDegree[0]) / 3600, 10);
  const uniformSettings = Array.from(settings);
  uniformSettings[3] = metersPerDegree[0] / 3600;
  uniformSettings[4] = metersPerDegree[1] / 3600;
  const uniform = computeTerrainSightLine(
    terrain,
    undefined,
    7,
    3,
    [0, 1],
    [4, 1],
    uniformSettings
  );
  expect(result).toEqual(uniform);
});

it('the geographic oracle range ring at 60 degrees is elliptical in pixels', () => {
  const width = 257;
  const height = 257;
  const center = 128;
  const grid = makeGPURasterMetadata({width, height});
  grid.affine = [1 / 3600, 0, 0, 0, -1 / 3600, 60 + (center + 0.5) / 3600];
  const terrain = new Float64Array(width * height);
  const settings = getGPUTerrainViewshedParameterValues(
    {observer: [center, center], observerHeight: 1, maximumDistance: 1500},
    grid
  );
  const options = {
    cellSizeMode: 'geographic' as const,
    georeference: getRasterGridGeoreference(grid)
  };
  const codes = computeTerrainViewshed(terrain, undefined, width, height, settings, options);
  let columnRadius = 0;
  let rowRadius = 0;
  for (let offset = 1; offset <= center; offset++) {
    if (codes[center * width + center + offset] === 1) columnRadius = offset;
    if (codes[(center + offset) * width + center] === 1) rowRadius = offset;
  }
  const expectedRatio = 1 / Math.cos(Math.PI / 3);
  expect(Math.abs(columnRadius / rowRadius / expectedRatio - 1)).toBeLessThan(0.01);
  expect(codes[center * width + center + columnRadius + 1]).toBe(2);
  expect(codes[(center + rowRadius + 1) * width + center]).toBe(2);
  const tolerant = computeTerrainViewshedWithTolerance(
    terrain,
    undefined,
    width,
    height,
    settings,
    [0, 0, 0, 0],
    options
  );
  expect(tolerant.map(result => result.code)).toEqual(codes);
});
