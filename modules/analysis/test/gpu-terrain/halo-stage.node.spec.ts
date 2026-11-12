// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createSourceFile, isClassDeclaration, ScriptTarget} from 'typescript';
import {makeGPURasterMetadata} from '../../src/raster-analysis/raster-grid/index';
import {GPUIsobands} from '../../src/raster-analysis/isolines/gpu-isobands';
import {
  getGPUTextureShadingCascadeSigmas,
  GPUTextureShading
} from '../../src/gpu-terrain/terrain-illumination/gpu-texture-shading';
import {getGPUGeomorphonsParameterValues} from '../../src/gpu-terrain/geomorphons/index';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';
import {createBand} from './terrain-test-utils';
import {computeGeomorphons} from './geomorphons/geomorphons-oracle';
import {computeWeissLandforms} from './topographic-position/weiss-landforms-oracle';

it('GPUIsobands bounds every constructed output: classes 0, triangles 1, boundary edges 2', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `isobands-halo-${instance++}`, format, length);
  const base = {
    grid: makeGPURasterMetadata({width: 5, height: 5}),
    values: view('float32', 25),
    breaks: view('float32', 1),
    parameters: view('float32', 12)
  };
  const classes = {bandClasses: view('uint32', 25)};
  const triangles = {
    triangles: view('float32x2', 30),
    triangleBands: view('uint32', 10),
    status: {count: view('uint32', 1), overflow: view('uint32', 1)}
  };
  const edges = {
    edges: view('float32x4', 20),
    edgeBands: view('uint32', 20),
    status: {edgeCount: view('uint32', 1), edgeOverflow: view('uint32', 1)}
  };
  try {
    expect(new GPUIsobands({...base, output: classes}).requiredHalo).toBe(0);
    expect(new GPUIsobands({...base, output: triangles}).requiredHalo).toBe(1);
    expect(new GPUIsobands({...base, output: edges}).requiredHalo).toBe(2);
    expect(
      new GPUIsobands({
        ...base,
        output: {
          ...classes,
          ...triangles,
          ...edges,
          status: {...triangles.status, ...edges.status}
        }
      }).requiredHalo
    ).toBe(2);
  } finally {
    device.destroy();
  }
});

/**
 * Enumerates source reads along one axis, away from raster borders. Separable
 * blur uses every integer tap; injection doubles indices when the grid scale changes; bilinear
 * reconstruction fetches floor and floor + 1 even at zero interpolation weight. The optional
 * nonzero-only enumeration also checks the review's lower bounds on influential samples.
 * Both axes use these same reads. No support-radius formula is used here.
 */
function enumerateTextureSourceIndices(
  coordinate: number,
  lastLevel: number,
  scales: readonly number[],
  radii: readonly number[],
  nonzeroOnly = false
): Set<number> {
  const position = coordinate / scales[lastLevel];
  let indices = new Set(
    scales[lastLevel] === 1 || nonzeroOnly
      ? [Math.floor(position), Math.ceil(position)]
      : [Math.floor(position), Math.floor(position) + 1]
  );
  for (let level = lastLevel; level >= 0; level--) {
    const expanded = new Set<number>();
    const injection = scales[level] / (level === 0 ? 1 : scales[level - 1]);
    for (const index of indices) {
      for (let offset = -radii[level]; offset <= radii[level]; offset++) {
        expanded.add((index + offset) * injection);
      }
    }
    indices = expanded;
  }
  return indices;
}

it.each([
  {levelCount: 3, baseSigma: 2, downsampleLevels: false, expected: 38, nonzero: 38},
  {levelCount: 3, baseSigma: 2, downsampleLevels: true, expected: 41, nonzero: 40},
  {levelCount: 6, baseSigma: 1, downsampleLevels: true, expected: 182, nonzero: 181},
  {levelCount: 8, baseSigma: 1, downsampleLevels: true, expected: 734, nonzero: 733},
  {levelCount: 4, baseSigma: 0.1, downsampleLevels: true, expected: 7, nonzero: 7}
])('GPUTextureShading bounds enumerated blur and reconstruction reads: $levelCount/$baseSigma/$downsampleLevels', configuration => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const {levelCount, baseSigma, downsampleLevels, expected, nonzero} = configuration;
  const contributor = new GPUTextureShading({
    grid: makeGPURasterMetadata({width: 5, height: 5}),
    elevation: createBand(graph, 'texture-elevation', 25),
    parameters: createTransientView(graph, 'texture-settings', 'float32', 12),
    output: {textureShade: createTransientView(graph, 'texture-shade', 'float32', 25)},
    levelCount,
    baseSigma,
    downsampleLevels
  });
  try {
    // Derive the kernel schedule independently from the shader's sigma threshold (4 pixels).
    let scale = 1;
    const sigmas = getGPUTextureShadingCascadeSigmas(levelCount, baseSigma);
    const scales = sigmas.map((sigma, level) => {
      if (downsampleLevels && level > 0 && sigma / scale >= 4) scale *= 2;
      return scale;
    });
    const radii = sigmas.map((sigma, level) => Math.max(1, Math.ceil((3 * sigma) / scales[level])));
    expect(contributor.levelScaleLog2.map(value => 2 ** value)).toEqual(scales);
    let maximumDistance = 0;
    let maximumNonzeroDistance = 0;
    for (let coordinate = 0; coordinate < scales.at(-1)!; coordinate++) {
      // Every level can contribute because band weights can change per frame.
      for (let level = 0; level < levelCount; level++) {
        for (const source of enumerateTextureSourceIndices(coordinate, level, scales, radii)) {
          maximumDistance = Math.max(maximumDistance, Math.abs(source - coordinate));
        }
        for (const source of enumerateTextureSourceIndices(
          coordinate,
          level,
          scales,
          radii,
          true
        )) {
          maximumNonzeroDistance = Math.max(maximumNonzeroDistance, Math.abs(source - coordinate));
        }
      }
    }
    expect(maximumDistance).toBe(expected);
    expect(maximumNonzeroDistance).toBe(nonzero);
    expect(contributor.requiredHalo).toBe(maximumDistance);
    if (levelCount === 3 && downsampleLevels) {
      // The reviewed impulse 39 pixels east affects even pixel 64; odd phases reach 40.
      expect(enumerateTextureSourceIndices(64, 2, scales, radii).has(103)).toBe(true);
    }
  } finally {
    device.destroy();
  }
});

// Removing `implements` is a declaration change only: structural TypeScript assignment would
// still accept any object with requiredHalo. Inspect the declaration to pin the withdrawn claim.
it.each([
  ['GPUGeomorphons', 'geomorphons/gpu-geomorphons.ts'],
  ['GPUTerrainWeissLandforms', 'topographic-position/gpu-terrain-weiss-landforms.ts']
])('%s does not claim a complete GPURasterHaloStage bound', (className, relativePath) => {
  const source = readFileSync(
    new URL(`../../src/gpu-terrain/${relativePath}`, import.meta.url),
    'utf8'
  );
  const syntax = createSourceFile(relativePath, source, ScriptTarget.Latest, true);
  const declaration = syntax.statements.find(
    statement => isClassDeclaration(statement) && statement.name?.text === className
  );
  expect(declaration).toBeDefined();
  if (!declaration || !isClassDeclaration(declaration)) throw new Error(`Missing ${className}`);
  const claims = declaration.heritageClauses?.flatMap(clause =>
    clause.types.map(type => type.expression.getText(syntax))
  );
  expect(claims).toContain('GPUCommandNodeProducer');
  expect(claims).not.toContain('GPURasterHaloStage');
});

it('geomorphon nominal radius 4 misses an east sample six columns away with per-frame cells [1, 2]', () => {
  const width = 25;
  const height = 17;
  const elevation = new Float32Array(width * height);
  elevation[8 * width + 14] = 10;
  const grid = makeGPURasterMetadata({width, height, cellSize: [1, 2]});
  const full = computeGeomorphons(
    elevation,
    undefined,
    width,
    height,
    getGPUGeomorphonsParameterValues({}, grid),
    {searchRadius: 4}
  );
  const windowGrid = makeGPURasterMetadata({width: 9, height: 9, cellSize: [1, 2]});
  const window = computeGeomorphons(
    new Float32Array(81),
    undefined,
    9,
    9,
    getGPUGeomorphonsParameterValues({}, windowGrid),
    {searchRadius: 4}
  );
  const eastPower = 3 ** 7;
  expect(Math.floor(full.pattern[8 * width + 8] / eastPower) % 3).toBe(2);
  expect(Math.floor(window.pattern[4 * 9 + 4] / eastPower) % 3).toBe(1);
});

it('Weiss radius-one local terms cannot bound global standardization, while local classes agree', () => {
  const width = 33;
  const elevation = new Float32Array(width * width);
  elevation[16 * width + 16] = 1;
  elevation[28 * width + 28] = 100;
  const window = new Float32Array(9);
  window[4] = 1;
  for (const standardization of ['global', 'local'] as const) {
    const options = {
      scales: [{radius: 1}, {radius: 1}] as const,
      cellSize: 1,
      standardization
    };
    const full = computeWeissLandforms(elevation, undefined, width, width, options);
    const cropped = computeWeissLandforms(window, undefined, 3, 3, options);
    if (standardization === 'global') {
      expect(full.standardizedPosition[16 * width + 16]).toBeCloseTo(0.3111, 4);
      expect(cropped.standardizedPosition[4]).toBeCloseTo(2.7938, 4);
      expect(full.landforms[16 * width + 16]).toBe(5);
      expect(cropped.landforms[4]).toBe(10);
    } else {
      expect(cropped.landforms[4]).toBe(full.landforms[16 * width + 16]);
      expect(cropped.standardizedPosition[4]).toBe(full.standardizedPosition[16 * width + 16]);
    }
  }
});
