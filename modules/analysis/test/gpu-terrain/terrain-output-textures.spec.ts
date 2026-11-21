// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPURasterTextureToBuffer} from '@luma.gl/graph/gpu-raster';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../src/raster-analysis/index';
import {
  GPUTerrainHorizon,
  getGPUTerrainHorizonParameterValues,
  GPUReliefShading,
  getGPUReliefShadingParameterValues,
  GPUSolarIrradiance,
  getGPUSolarIrradianceParameterValues,
  GPUSolarShadowMask,
  getGPUSolarShadowMaskParameterValues,
  GPUTerrainCastShadow,
  getGPUTerrainCastShadowParameterValues
} from '../../src/gpu-terrain/terrain-illumination/index';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {
  GPUTerrainDerivatives,
  getGPUTerrainDerivativesParameterValues
} from '../../src/gpu-terrain/terrain-analysis/index';
import {createTerrainReviewFixture} from './terrain-processing/terrain-review-fixture';

const GRID = makeGPURasterMetadata({width: 3, height: 3});
const COUNT = 9;

type Rig = ReturnType<typeof createTerrainReviewFixture>;

/** Poisoned storage and an independent channel-zero readback, including the dispatch tail. */
function createTextureOutput(
  rig: Rig,
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>
) {
  const texture = device.createTexture({
    width: 3,
    height: 3,
    format: 'rgba32float',
    usage: Texture.STORAGE | Texture.SAMPLE | Texture.COPY_DST
  });
  texture.writeData(new Float32Array(COUNT * 4).fill(0.3125));
  const view = rig.graph.createTextureView(
    rig.graph.importTexture(
      {
        id: 'texture-output',
        width: 3,
        height: 3,
        format: 'rgba32float',
        usage: texture.props.usage
      },
      texture
    ),
    {mipLevelCount: 1}
  );
  const readback = rig.output('float32', COUNT);
  return {
    texture,
    view,
    read: readback.read,
    connect() {
      new GPURasterTextureToBuffer({
        input: {id: 'texture-result', format: 'float32', storage: {kind: 'texture', view}},
        output: readback.view,
        outputValidity: rig.output('uint32', COUNT).view
      }).addToGraph(rig.graph);
    }
  };
}

const CASES = [
  'skyViewFactorTexture',
  'hillshadeTexture',
  'reliefTexture',
  'insolationTexture',
  'sunHoursTexture',
  'illuminationTexture',
  'sunVisibilityTexture',
  'derivativeHillshadeTexture',
  'maskSunVisibilityTexture'
] as const;

for (const port of CASES) {
  for (const includeBuffer of [false, true]) {
    it(`terrain texture oracle: ${port}, scalar buffer ${includeBuffer}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = createTerrainReviewFixture(device);
      const sink = createTextureOutput(rig, device);
      const scalar = includeBuffer ? rig.output('float32', COUNT) : undefined;
      const elevation = rig.input('float32', new Float32Array(COUNT).fill(10));
      const band = {
        id: 'elevation',
        format: 'float32' as const,
        storage: {kind: 'buffer' as const, values: elevation.view}
      };
      let expected: number;
      // Closed forms on horizontal ground: solid-angle SVF=1; Lambert shade=sin(30)=1/2;
      // relief=exposure*shade=1/4; two hours at DNI=100 and diffuse=10 gives 120 Wh/m2;
      // illumination=ambient(0.2)+sun(0.6)*sin(30)=0.5; unobstructed sun visibility=1.
      if (port === 'skyViewFactorTexture') {
        expected = 1;
        rig.graph.add(
          new GPUTerrainHorizon({
            grid: GRID,
            elevation: band,
            directionCount: 4,
            maximumRadius: 1,
            parameters: rig.input('float32', getGPUTerrainHorizonParameterValues({}, GRID)).view,
            output: {skyViewFactorTexture: sink.view, skyViewFactor: scalar?.view}
          })
        );
      } else if (port === 'hillshadeTexture' || port === 'reliefTexture') {
        expected = port === 'hillshadeTexture' ? 0.5 : 0.25;
        rig.graph.add(
          new GPUReliefShading({
            grid: GRID,
            elevation: band,
            parameters: rig.input(
              'float32',
              getGPUReliefShadingParameterValues(
                {
                  lights: [{azimuthDegrees: 90, altitudeDegrees: 30}],
                  exposure: 0.5
                },
                GRID
              )
            ).view,
            output:
              port === 'hillshadeTexture'
                ? {hillshadeTexture: sink.view, hillshade: scalar?.view}
                : {reliefTexture: sink.view, relief: scalar?.view}
          })
        );
      } else if (port === 'sunHoursTexture' || port === 'insolationTexture') {
        expected = port === 'sunHoursTexture' ? 2 : 120;
        rig.graph.add(
          new GPUSolarIrradiance({
            grid: GRID,
            directionCount: 4,
            horizon: rig.input('float32', new Float32Array(COUNT * 4)).view,
            sunTable: rig.input('float32', [90, 30, 2, 100]).view,
            sampleCapacity: 1,
            parameters: rig.input(
              'float32',
              getGPUSolarIrradianceParameterValues({sampleCount: 1, diffuseIrradiance: 10})
            ).view,
            output:
              port === 'sunHoursTexture'
                ? {sunHoursTexture: sink.view, sunHours: scalar?.view}
                : {insolationTexture: sink.view, insolation: scalar?.view}
          })
        );
      } else if (port === 'derivativeHillshadeTexture') {
        expected = 0.5;
        rig.graph.add(
          new GPUTerrainDerivatives({
            grid: GRID,
            elevation: band,
            parameters: rig.input(
              'float32',
              getGPUTerrainDerivativesParameterValues({altitudeDegrees: 30}, GRID)
            ).view,
            output: {hillshadeTexture: sink.view, hillshade: scalar?.view}
          })
        );
      } else if (port === 'illuminationTexture' || port === 'maskSunVisibilityTexture') {
        expected = port === 'illuminationTexture' ? 0.5 : 1;
        rig.graph.add(
          new GPUSolarShadowMask({
            grid: GRID,
            directionCount: 4,
            horizon: rig.input('float32', new Float32Array(COUNT * 4)).view,
            slope: rig.input('float32', new Float32Array(COUNT)).view,
            aspect: rig.input('float32', new Float32Array(COUNT).fill(-1)).view,
            parameters: rig.input(
              'float32',
              getGPUSolarShadowMaskParameterValues({
                azimuthDegrees: 90,
                altitudeDegrees: 30,
                angularRadiusDegrees: 0,
                sunIntensity: 0.6,
                ambientIntensity: 0.2
              })
            ).view,
            output:
              port === 'illuminationTexture'
                ? {illuminationTexture: sink.view, illumination: scalar?.view}
                : {sunVisibilityTexture: sink.view, sunVisibility: scalar?.view}
          })
        );
      } else {
        expected = 1;
        rig.graph.add(
          new GPUTerrainCastShadow({
            grid: GRID,
            elevation: band,
            parameters: rig.input(
              'float32',
              getGPUTerrainCastShadowParameterValues(
                {
                  azimuthDegrees: 90,
                  altitudeDegrees: 30,
                  angularRadiusDegrees: 0
                },
                GRID
              )
            ).view,
            output: {sunVisibilityTexture: sink.view, sunVisibility: scalar?.view}
          })
        );
      }
      sink.connect();
      let compiled: ReturnType<Rig['run']> | undefined;
      try {
        compiled = rig.run();
        for (const values of [await sink.read(), ...(scalar ? [await scalar.read()] : [])]) {
          expect(values).toHaveLength(COUNT);
          for (const value of values) {
            expect(Number.isFinite(value)).toBe(true);
            // At most a few float32 trig/multiply roundings, including accumulated energy.
            expect(Math.abs(value - expected)).toBeLessThan(2e-5);
          }
        }
      } finally {
        compiled?.destroy();
        sink.texture.destroy();
        rig.destroy();
      }
    });
  }
}
