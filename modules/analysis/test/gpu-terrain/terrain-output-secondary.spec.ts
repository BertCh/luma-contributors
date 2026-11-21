// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPURasterTextureToBuffer} from '@luma.gl/graph/gpu-raster';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {makeGPURasterMetadata} from '../../src/raster-analysis/index';
import {
  GPUTerrainCurvature,
  getGPUTerrainCurvatureParameterValues
} from '../../src/gpu-terrain/terrain-curvature/index';
import {
  GPUTerrainLineOfSight,
  GPUTerrainCumulativeViewshed,
  GPU_TERRAIN_VISIBILITY,
  getGPUTerrainLineOfSightParameterValues
} from '../../src/gpu-terrain/terrain-analysis/index';
import {
  GPUReliefShading,
  getGPUReliefShadingParameterValues,
  GPUTextureShading,
  getGPUTextureShadingParameterValues
} from '../../src/gpu-terrain/terrain-illumination/index';
import {createTerrainReviewFixture} from './terrain-processing/terrain-review-fixture';
import {TEXTURE_FIXTURES} from './parity-wp5-round8-composite-fixtures';
import {expectFlatRelief, expectReliefColor} from './terrain-features/terrain-terminal-assertions';

it('ring curvature is nine on a hand-derived quadratic with gain two and z factor three', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = createTerrainReviewFixture(device);
  const grid = makeGPURasterMetadata({width: 3, height: 3});
  const ring = rig.output('float32', 9);
  // z=-(x²+y²): four axis differences 1 and four diagonal differences 2.
  // (4+8)/(8*1*1) * gain(2) * zFactor(3) = 9; rings crossing the border are invalid.
  rig.graph.add(
    new GPUTerrainCurvature({
      grid,
      ringRadii: [1],
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: rig.input('float32', [-2, -1, -2, -1, 0, -1, -2, -1, -2]).view
        }
      },
      parameters: rig.input(
        'float32',
        getGPUTerrainCurvatureParameterValues({ringGains: [2], zFactor: 3}, grid)
      ).view,
      output: {ringCurvature: ring.view}
    })
  );
  const compiled = rig.run();
  try {
    const values = await ring.read();
    for (const [index, value] of values.entries()) {
      if (index === 4) expect(value).toBe(9);
      else expect(value).toBeNaN();
    }
  } finally {
    compiled.destroy();
    rig.destroy();
  }
});

for (const traversal of ['march', 'pyramid'] as const) {
  for (const includeClearance of [true, false]) {
    it(`line-of-sight clearance is minus three metres, ${traversal}, output ${includeClearance}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = createTerrainReviewFixture(device);
      const grid = makeGPURasterMetadata({width: 11, height: 1});
      const clearance = rig.output('float32', 2);
      const visibility = rig.output('uint32', 2);
      // Eye (0,1), wall (5,2), target (10,0): slopes .2 and -.1;
      // clearance=(targetSlope-maxTerrainSlope)*10 = -3 metres.
      const values = new Float32Array(11);
      values[5] = 2;
      rig.graph.add(
        new GPUTerrainLineOfSight({
          grid,
          traversal,
          elevation: {
            id: 'elevation',
            format: 'float32',
            storage: {kind: 'buffer', values: rig.input('float32', values).view}
          },
          pairs: rig.graph.createDataView(
            rig.input('float32', [0, 0, 10, 0, 0, 0, 2, 0]).view.buffer,
            {
              format: 'float32x4',
              length: 2
            }
          ),
          parameters: rig.input(
            'float32',
            getGPUTerrainLineOfSightParameterValues(
              {observerHeight: 1, targetHeight: 0, curvatureCoefficient: 0},
              grid
            )
          ).view,
          output: {
            visibility: visibility.view,
            clearance: includeClearance ? clearance.view : undefined
          }
        })
      );
      const compiled = rig.run();
      try {
        // Column 2 is unobstructed: its nonzero result detects a skipped dispatch.
        expect(await visibility.read()).toEqual([
          GPU_TERRAIN_VISIBILITY.hidden,
          GPU_TERRAIN_VISIBILITY.visible
        ]);
        if (includeClearance) {
          const [value] = await clearance.read();
          expect(Number.isFinite(value)).toBe(true);
          expect(Math.abs(value + 3)).toBeLessThan(1e-5);
        }
      } finally {
        compiled.destroy();
        rig.destroy();
      }
    });
  }
  for (const includeMarginal of [true, false]) {
    it(`cumulative viewshed counts two marginal observers, ${traversal}, output ${includeMarginal}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = createTerrainReviewFixture(device);
      const grid = makeGPURasterMetadata({width: 12, height: 1});
      const values = new Float32Array(12);
      values[6] = 0.5;
      const visible = rig.output('uint32', 12),
        marginal = rig.output('uint32', 12);
      // The wall at column 6 is 1/10 metre above the line to column 10, projecting
      // to 1/6 metre at the target: marginal twice under the 0.5m tolerance.
      // Column 2 has no intervening wall and must be visible to both observers.
      rig.graph.add(
        new GPUTerrainCumulativeViewshed({
          grid,
          traversal,
          observersPerDispatch: 1,
          elevation: {
            id: 'elevation',
            format: 'float32',
            storage: {kind: 'buffer', values: rig.input('float32', values).view}
          },
          observers: rig.graph.createDataView(rig.input('float32', [0, 0, 0, 0]).view.buffer, {
            format: 'float32x2',
            length: 2
          }),
          parameters: rig.input(
            'float32',
            getGPUTerrainLineOfSightParameterValues(
              {observerHeight: 1, targetHeight: 0, curvatureCoefficient: 0, toleranceMeters: 0.5},
              grid
            )
          ).view,
          output: {
            visibleCount: visible.view,
            marginalCount: includeMarginal ? marginal.view : undefined
          }
        })
      );
      const compiled = rig.run();
      try {
        const visibleValues = await visible.read();
        expect(visibleValues[10]).toBe(0);
        expect(visibleValues[2]).toBe(2);
        if (includeMarginal) expect((await marginal.read())[10]).toBe(2);
      } finally {
        compiled.destroy();
        rig.destroy();
      }
    });
  }
}

for (const selection of ['all', 'hillshade', 'relief', 'color'] as const) {
  it(`relief scalar oracle on flat terrain: ${selection}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = createTerrainReviewFixture(device);
    const grid = makeGPURasterMetadata({width: 3, height: 3});
    const hillshade = rig.output('float32', 9),
      relief = rig.output('float32', 9),
      color = rig.output('uint32', 9);
    const all = selection === 'all';
    // One light at 30 degrees: shade=.5, exposure=.5 -> relief=.25. Tint is disabled,
    // so a constant [0.8,0.4,0.2] base becomes RGB [0.2,0.1,0.05] with opaque alpha.
    rig.graph.add(
      new GPUReliefShading({
        grid,
        elevation: {
          id: 'elevation',
          format: 'float32',
          storage: {kind: 'buffer', values: rig.input('float32', new Float32Array(9).fill(10)).view}
        },
        parameters: rig.input(
          'float32',
          getGPUReliefShadingParameterValues(
            {
              lights: [{azimuthDegrees: 90, altitudeDegrees: 30}],
              exposure: 0.5,
              tintStrength: 0,
              elevationStops: [{elevation: 0, color: [0.8, 0.4, 0.2]}]
            },
            grid
          )
        ).view,
        output: all
          ? {hillshade: hillshade.view, relief: relief.view, color: color.view}
          : selection === 'hillshade'
            ? {hillshade: hillshade.view}
            : selection === 'relief'
              ? {relief: relief.view}
              : {color: color.view}
      })
    );
    const compiled = rig.run();
    try {
      for (const [output, expected] of [
        [hillshade, 0.5],
        [relief, 0.25]
      ] as const) {
        if (
          !all &&
          output !==
            (selection === 'hillshade' ? hillshade : selection === 'relief' ? relief : undefined)
        )
          continue;
        expectFlatRelief(await output.read(), expected);
      }
      if (all || selection === 'color') expectReliefColor(await color.read());
    } finally {
      compiled.destroy();
      rig.destroy();
    }
  });
}

it('texture-shade texture-only publication matches the committed SciPy convolution reference', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = TEXTURE_FIXTURES[0];
  const settings = fixture.cases[0];
  const {width, height, cellSize, north} = fixture.input;
  const count = width * height;
  const rig = createTerrainReviewFixture(device);
  const grid = makeGPURasterMetadata({width, height, cellSize, origin: [0, north]});
  const texture = device.createTexture({
    width,
    height,
    format: 'rgba32float',
    usage: Texture.STORAGE | Texture.SAMPLE | Texture.COPY_DST
  });
  texture.writeData(new Float32Array(count * 4).fill(12345));
  const view = rig.graph.createTextureView(
    rig.graph.importTexture(
      {id: 'texture-output', width, height, format: 'rgba32float', usage: texture.props.usage},
      texture
    ),
    {mipLevelCount: 1}
  );
  const result = rig.output('float32', count);
  rig.graph.add(
    new GPUTextureShading({
      grid,
      levelCount: settings.levelCount,
      baseSigma: settings.baseSigma,
      hasNodata: true,
      downsampleLevels: false,
      elevation: {
        id: 'elevation',
        format: 'float32',
        noDataValue: -9999,
        storage: {kind: 'buffer', values: rig.input('float32', fixture.input.elevation).view},
        validity: rig.input('uint32', fixture.input.validity).view
      },
      parameters: rig.input(
        'float32',
        getGPUTextureShadingParameterValues({detail: settings.detail, gain: settings.gain})
      ).view,
      output: {textureShadeTexture: view}
    })
  );
  new GPURasterTextureToBuffer({
    input: {id: 'readback', format: 'float32', storage: {kind: 'texture', view}},
    output: result.view,
    outputValidity: rig.output('uint32', count).view
  }).addToGraph(rig.graph);
  const compiled = rig.run();
  try {
    const actual = await result.read();
    const expected = settings.expected['textureShade'];
    const finite = expected.filter(Number.isFinite);
    const range = Math.max(...finite) - Math.min(...finite);
    expect(expected.some(value => Number.isFinite(value) && Math.abs(value) > 0.1)).toBe(true);
    for (const [index, value] of expected.entries()) {
      if (Number.isNaN(value)) expect(actual[index]).toBeNaN();
      else {
        expect(Number.isFinite(actual[index])).toBe(true);
        // Same L budget as the existing SciPy parity: range-scaled float32 absolute
        // floor plus 1e-4 relative, rather than comparison with another GPU output.
        expect(Math.abs(actual[index] - value)).toBeLessThanOrEqual(
          1e-6 * Math.max(1, range) + 1e-4 * Math.abs(value)
        );
      }
    }
  } finally {
    compiled.destroy();
    texture.destroy();
    rig.destroy();
  }
});
