// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  GPUReliefShading,
  getGPUReliefShadingParameterValues
} from '../../../src/gpu-terrain/terrain-illumination/gpu-relief-shading';
import {
  GPUTerrainDerivatives,
  getGPUTerrainDerivativesParameterValues
} from '../../../src/gpu-terrain/terrain-analysis/gpu-terrain-derivatives';
import {
  GPUTerrainViewshed,
  getGPUTerrainViewshedParameterValues
} from '../../../src/gpu-terrain/terrain-analysis/gpu-terrain-viewshed';
import {
  GPUTerrainLineOfSight,
  getGPUTerrainLineOfSightParameterValues
} from '../../../src/gpu-terrain/terrain-analysis/gpu-terrain-line-of-sight';
import {GPUTerrainCumulativeViewshed} from '../../../src/gpu-terrain/terrain-analysis/gpu-terrain-cumulative-viewshed';
import {computeReliefShading} from './relief-shading-oracle';
import {GPUPointHorizonProfile} from '../../../src/gpu-terrain/point-horizon/gpu-point-horizon-profile';
import {getGPUPointHorizonProfileParameterValues} from '../../../src/gpu-terrain/point-horizon/point-horizon-parameters';
import {POINT_PROFILE_GRID, POINT_PROFILE_ELEVATION} from './illumination-sight-w3-fixtures';

it('UNI-19 firstAzimuthIndex is clockwise from grid north and azimuthBinCount selects the span', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = POINT_PROFILE_GRID;
  // Eye at (1,1): the east neighbor rises one meter; column 3 completes its bilinear footprint.
  const terrain = createInputBuffer(device, POINT_PROFILE_ELEVATION);
  const observers = createInputBuffer(device, new Float32Array([1, 1, 0, 0]));
  const skyline = createOutputBuffer(device, 1);
  const parameters = new GPUParameterBuffer(device, {
    id: 'profile-parameters',
    format: 'float32',
    length: 8,
    values: getGPUPointHorizonProfileParameterValues({}, grid)
  });
  const graph = new GPUCommandGraph(device);
  graph.add(
    new GPUPointHorizonProfile({
      grid,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(
            graph,
            'elevation',
            terrain,
            'float32',
            POINT_PROFILE_ELEVATION.length
          )
        }
      },
      observers: importGraphBuffer(graph, 'observers', observers, 'float32x4', 1),
      parameters: parameters.importToGraph(graph),
      cellSize: 1,
      minimumDistance: 1,
      maximumDistance: 1,
      azimuthCount: 4,
      firstAzimuthIndex: 1,
      azimuthBinCount: 1,
      traversal: 'march',
      output: {skylineAngle: importGraphBuffer(graph, 'skyline', skyline, 'float32', 1)}
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const angle = (await readFloat32(skyline, 1))[0];
    expect(Number.isFinite(angle)).toBe(true);
    expect(angle).toBeCloseTo(45, 4); // atan(1 meter / 1 meter) = 45 degrees.
  } finally {
    compiled.destroy();
    parameters.destroy();
    terrain.destroy();
    observers.destroy();
    skyline.destroy();
  }
});

it('DEF-03 resident relief codes honor Imhof', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeGPURasterMetadata({width: 3, height: 3, cellSize: 1});
  // Plane z = east + north / 2: Horn's center derivatives are exactly (1, 0.5).
  const elevation = new Float32Array([0, 1, 2, -0.5, 0.5, 1.5, -1, 0, 1]);
  const input = createInputBuffer(device, elevation);
  const shade = createOutputBuffer(device, 9);
  const validity = createOutputBuffer(device, 9);
  const style = {
    lights: [{azimuthDegrees: 225, altitudeDegrees: 45}],
    lightWeighting: 'imhof-swing' as const,
    imhofSwingDegrees: 65
  };
  const words = getGPUReliefShadingParameterValues(style, grid);
  const parameters = new GPUParameterBuffer(device, {
    id: 'relief-parameters',
    format: 'float32',
    length: 80,
    values: words
  });
  const graph = new GPUCommandGraph(device);
  graph.add(
    new GPUReliefShading({
      grid,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', input, 'float32', 9)
        }
      },
      parameters: parameters.importToGraph(graph),
      output: {
        hillshade: importGraphBuffer(graph, 'shade', shade, 'float32', 9),
        outputValidity: importGraphBuffer(graph, 'validity', validity, 'uint32', 9)
      }
    })
  );
  const compiled = graph.compile();
  try {
    // Independent Python math: normalize((-1,-.5,1)), aspect=atan2(-1,-.5),
    // azimuth'=225+65*sin(aspect-225); max(dot(normal,light(azimuth',45)),0).
    for (const [code, expected] of [
      [0, 0.9714045207910316],
      [1, 0.9714045207910316],
      [2, 0.9980901053651041]
    ]) {
      words[6] = code;
      parameters.write(words);
      submitGraph(device, compiled, undefined);
      const actual = (await readFloat32(shade, 9))[4];
      expect(Number.isFinite(actual)).toBe(true);
      expect(actual).toBeCloseTo(expected, 5);
      expect((await readUint32(validity, 9))[4]).toBe(1);
    }
    const oracle = computeReliefShading({
      width: 3,
      height: 3,
      elevation,
      cellSize: [1, 1],
      settings: style
    });
    expect(oracle.hillshade[4]).toBeCloseTo(0.9980901053651041, 10);
    // Independent Python dot products for normal (-2/3,-1/3,2/3). Opposite lights have
    // equal sin^2 aspect weights; their explicit fixed weights are 3:1, so codes 0 and 1 differ.
    words.set(
      getGPUReliefShadingParameterValues(
        {
          lights: [
            {azimuthDegrees: 225, altitudeDegrees: 45, weight: 3},
            {azimuthDegrees: 45, altitudeDegrees: 45, weight: 1}
          ]
        },
        grid
      )
    );
    for (const [code, expected] of [
      [0, 0.7285533905932737],
      [1, 0.4857022603955158]
    ]) {
      words[6] = code;
      parameters.write(words);
      submitGraph(device, compiled, undefined);
      const actual = (await readFloat32(shade, 9))[4];
      expect(Number.isFinite(actual)).toBe(true);
      expect(actual).toBeCloseTo(expected, 5);
      expect((await readUint32(validity, 9))[4]).toBe(1);
    }
    // Raw code 99 and a positive subnormal must both be invalid, even with flush-to-zero math.
    for (const rawWord of [0x42c60000, 0x00000001]) {
      new Uint32Array(words.buffer, words.byteOffset, words.length)[6] = rawWord;
      parameters.write(words);
      submitGraph(device, compiled, undefined);
      expect(Number.isNaN((await readFloat32(shade, 9))[4])).toBe(true);
      expect((await readUint32(validity, 9))[4]).toBe(0);
    }
  } finally {
    compiled.destroy();
    parameters.destroy();
    input.destroy();
    shade.destroy();
    validity.destroy();
  }
});

for (const route of ['direct', 'tolerance', 'pyramid'] as const) {
  it(`URT-05 invalid resident viewshed heights are noData on ${route}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const grid = makeGPURasterMetadata({width: 3, height: 1, cellSize: 1});
    const input = createInputBuffer(device, new Float32Array([0, 10, 0]));
    const output = createOutputBuffer(device, 3);
    const words = getGPUTerrainViewshedParameterValues({observer: [0, 0], observerHeight: 0}, grid);
    const parameters = new GPUParameterBuffer(device, {
      id: 'viewshed-parameters',
      format: 'float32',
      length: 10,
      values: words
    });
    const tolerance = new GPUParameterBuffer(device, {
      id: 'tolerance',
      format: 'float32',
      length: 4,
      values: new Float32Array(4)
    });
    const graph = new GPUCommandGraph(device);
    graph.add(
      new GPUTerrainViewshed({
        grid,
        elevation: {
          id: 'elevation',
          format: 'float32',
          storage: {
            kind: 'buffer',
            values: importGraphBuffer(graph, 'elevation', input, 'float32', 3)
          }
        },
        parameters: parameters.importToGraph(graph),
        traversal: route === 'pyramid' ? 'pyramid' : 'march',
        toleranceParameters: route === 'tolerance' ? tolerance.importToGraph(graph) : undefined,
        output: {visibility: importGraphBuffer(graph, 'visibility', output, 'uint32', 3)}
      })
    );
    const compiled = graph.compile();
    try {
      for (const slot of [2, 3]) {
        const invalid = words.slice();
        invalid[slot] = NaN;
        parameters.write(invalid);
        submitGraph(device, compiled, undefined);
        expect(Array.from(await readUint32(output, 3))).toEqual([3, 3, 3]);
      }
    } finally {
      compiled.destroy();
      parameters.destroy();
      tolerance.destroy();
      input.destroy();
      output.destroy();
    }
  });
}

it('URT-05 invalid pair heights are noData and invalid cumulative observers contribute zero', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeGPURasterMetadata({width: 3, height: 1, cellSize: 1});
  const terrain = createInputBuffer(device, new Float32Array([0, 10, 0]));
  const pairs = createInputBuffer(device, new Float32Array([0, 0, 2, 0, 0, 0, 2, 0]));
  const heights = createInputBuffer(device, new Float32Array([NaN, 0, 0, NaN]));
  const observers = createInputBuffer(device, new Float32Array([0, 0, 0, 0]));
  const observerHeights = createInputBuffer(device, new Float32Array([NaN, 0]));
  const visibility = createOutputBuffer(device, 2);
  const counts = createOutputBuffer(device, 3);
  const parameters = new GPUParameterBuffer(device, {
    id: 'sight-parameters',
    format: 'float32',
    length: 12,
    values: getGPUTerrainLineOfSightParameterValues({observerHeight: 0}, grid)
  });
  const graph = new GPUCommandGraph(device);
  const elevation = {
    id: 'elevation',
    format: 'float32' as const,
    storage: {
      kind: 'buffer' as const,
      values: importGraphBuffer(graph, 'elevation', terrain, 'float32', 3)
    }
  };
  const settings = parameters.importToGraph(graph);
  graph.add(
    new GPUTerrainLineOfSight({
      grid,
      elevation,
      parameters: settings,
      pairs: importGraphBuffer(graph, 'pairs', pairs, 'float32x4', 2),
      pairHeights: importGraphBuffer(graph, 'heights', heights, 'float32x2', 2),
      output: {visibility: importGraphBuffer(graph, 'visibility', visibility, 'uint32', 2)}
    })
  );
  graph.add(
    new GPUTerrainCumulativeViewshed({
      grid,
      elevation,
      parameters: settings,
      observers: importGraphBuffer(graph, 'observers', observers, 'float32x2', 2),
      observerHeights: importGraphBuffer(graph, 'observer-heights', observerHeights, 'float32', 2),
      output: {visibleCount: importGraphBuffer(graph, 'counts', counts, 'uint32', 3)}
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(Array.from(await readUint32(visibility, 2))).toEqual([3, 3]);
    // Valid eye at column 0 sees itself and the ridge at 1; ridge blocks target 2.
    expect(Array.from(await readUint32(counts, 3))).toEqual([1, 1, 0]);
  } finally {
    compiled.destroy();
    parameters.destroy();
    for (const buffer of [terrain, pairs, heights, observers, observerHeights, visibility, counts])
      buffer.destroy();
  }
});

it('URT-05 flat hillshade clears validity for either nonfinite resident light angle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeGPURasterMetadata({width: 3, height: 3, cellSize: 1});
  const terrain = createInputBuffer(device, new Float32Array(9));
  const shade = createOutputBuffer(device, 9);
  const validity = createOutputBuffer(device, 9);
  const words = getGPUTerrainDerivativesParameterValues({}, grid);
  const parameters = new GPUParameterBuffer(device, {
    id: 'light-parameters',
    format: 'float32',
    length: 8,
    values: words
  });
  const graph = new GPUCommandGraph(device);
  graph.add(
    new GPUTerrainDerivatives({
      grid,
      borderMode: 'clamp',
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', terrain, 'float32', 9)
        }
      },
      parameters: parameters.importToGraph(graph),
      output: {
        hillshade: importGraphBuffer(graph, 'shade', shade, 'float32', 9),
        outputValidity: importGraphBuffer(graph, 'validity', validity, 'uint32', 9)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(Array.from(await readUint32(validity, 9))).toEqual(new Array(9).fill(1));
    const validShade = await readFloat32(shade, 9);
    expect(Array.from(validShade).every(Number.isFinite)).toBe(true);
    expect(validShade[4]).toBeCloseTo(0.7071067811865476, 5);
    for (const slot of [3, 4]) {
      const invalid = words.slice();
      invalid[slot] = NaN;
      parameters.write(invalid);
      submitGraph(device, compiled, undefined);
      expect(Array.from(await readUint32(validity, 9))).toEqual(new Array(9).fill(0));
      expect(Array.from(await readFloat32(shade, 9)).every(Number.isNaN)).toBe(true);
    }
  } finally {
    compiled.destroy();
    parameters.destroy();
    terrain.destroy();
    shade.destroy();
    validity.destroy();
  }
});
