// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {AmbientLight, DirectionalLight, Deck, MapView, type Layer} from '@deck.gl/core';
import {
  GPUSolidPolygonLayer,
  WGSLShadowLightingEffect
} from '../../../../examples/deck/kepler-compute/gpu-layers/index';
import {
  makeKeplerGeometryGPUVectors,
  prepareKeplerGeometrySource
} from '../../../../examples/deck/kepler-compute/src/geometry-preparation/geometry-gpu-vectors';
import {Buffer, Texture, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';

const SIZE = 192;

function makeRing(halfWidth: number, halfHeight: number): number[][] {
  return [
    [-halfWidth + 0.0000013, 45.0000017 - halfHeight, 0],
    [halfWidth + 0.0000013, 45.0000017 - halfHeight, 0],
    [halfWidth + 0.0000013, 45.0000017 + halfHeight, 0],
    [-halfWidth + 0.0000013, 45.0000017 + halfHeight, 0],
    [-halfWidth + 0.0000013, 45.0000017 - halfHeight, 0]
  ];
}

function makeVector<Format extends 'float32' | 'uint32'>(
  device: Device,
  format: Format,
  values: number[]
): GPUVector<Format> {
  return new GPUVector({
    type: 'data',
    format,
    data: [
      new GPUData({
        format,
        length: values.length,
        ownsBuffer: true,
        buffer: device.createBuffer({
          data: format === 'float32' ? new Float32Array(values) : new Uint32Array(values),
          usage: Buffer.VERTEX | Buffer.STORAGE | Buffer.COPY_DST
        })
      })
    ]
  });
}

// Extruded polygon walls, fill and wireframe with brushing and highlight uniforms all enabled
// (the Kepler resident configuration) must link a valid pipeline layout with the shadow module and
// must darken the ground behind a tall extruded tower.
for (const featureLevel of ['core', 'max'] as const) {
  test(`extruded polygon sides with brushing link with the shadow module and cast shadows on a ${featureLevel} device`, async () => {
    const device = await getWebGPUTestDevice(featureLevel);
    if (!device) throw new Error('Real WebGPU device required');
    const parent = document.createElement('div');
    document.body.append(parent);
    const texture = device.createTexture({
      width: SIZE,
      height: SIZE,
      format: 'rgba8unorm',
      usage: Texture.RENDER | Texture.COPY_SRC
    });
    const framebuffer = device.createFramebuffer({
      width: SIZE,
      height: SIZE,
      colorAttachments: [texture],
      depthStencilAttachment: 'depth24plus'
    });
    const ground = makeKeplerGeometryGPUVectors(
      device,
      prepareKeplerGeometrySource(
        [{id: 'ground', rows: [{type: 'Polygon', coordinates: [makeRing(0.004, 0.003)]}]}],
        {fp64: true}
      )
    );
    const tower = makeKeplerGeometryGPUVectors(
      device,
      prepareKeplerGeometrySource(
        [{id: 'tower', rows: [{type: 'Polygon', coordinates: [makeRing(0.0003, 0.0003)]}]}],
        {fp64: true}
      )
    );
    const mask = makeVector(device, 'uint32', [1]);
    const towerMask = makeVector(device, 'uint32', [1]);
    let error: Error | null = null;
    let frames = 0;
    let restore = () => {};
    const deck = new Deck({
      device,
      parent,
      width: SIZE,
      height: SIZE,
      useDevicePixels: false,
      _framebuffer: framebuffer,
      views: new MapView({id: 'main'}),
      initialViewState: {longitude: 0, latitude: 45, zoom: 17, pitch: 45, bearing: 0},
      layers: [],
      onError: next => {
        error = next;
      },
      onAfterRender: () => {
        frames++;
      }
    });
    const material = {
      ambient: 0.6,
      diffuse: 0.6,
      shininess: 0,
      specularColor: [0, 0, 0] as [number, number, number]
    };
    const render = async (effect: WGSLShadowLightingEffect | null): Promise<Uint8Array> => {
      const common = {
        extruded: true,
        wireframe: true,
        material,
        brushingEnabled: true,
        brushingRadius: 1,
        highlightedPrimitiveRange: [0, 1] as [number, number],
        getFillColor: [220, 220, 220, 255] as [number, number, number, number]
      };
      const layers: Layer[] = [
        new GPUSolidPolygonLayer({
          id: 'ground',
          ...ground.polygons,
          ...common,
          getElevation: 0,
          selectionMask: mask,
          extrusionRings: {
            positions: ground.polygonOutlines.getPath,
            sourceRowIds: ground.polygonOutlines.sourceRowIds,
            sideOrientations: ground.polygonOutlines.sideOrientations
          }
        }),
        new GPUSolidPolygonLayer({
          id: 'tower',
          ...tower.polygons,
          ...common,
          getElevation: 60,
          selectionMask: towerMask,
          extrusionRings: {
            positions: tower.polygonOutlines.getPath,
            sourceRowIds: tower.polygonOutlines.sourceRowIds,
            sideOrientations: tower.polygonOutlines.sideOrientations
          }
        })
      ];
      deck.setProps({layers, effects: effect ? [effect as never] : []});
      const deadline = performance.now() + 15000;
      while (
        performance.now() < deadline &&
        !(
          deck.isInitialized &&
          layers.every(
            layer =>
              layer.state &&
              layer.getModels().every(model => model.pipeline.linkStatus === 'success')
          )
        )
      ) {
        if (error) throw error;
        await new Promise(resolve => requestAnimationFrame(resolve));
      }
      if (error) throw error;
      for (const layer of layers) {
        expect(layer.getModels().length).toBeGreaterThanOrEqual(3);
        for (const model of layer.getModels()) expect(model.pipeline.linkStatus).toBe('success');
      }
      for (let frame = 0; frame < 2; frame++) {
        const previous = frames;
        deck.redraw('shadow sides');
        while (frames <= previous && performance.now() < deadline) {
          if (error) throw error;
          await new Promise(resolve => requestAnimationFrame(resolve));
        }
      }
      if (error) throw error;
      const layout = texture.computeMemoryLayout({width: SIZE, height: SIZE});
      const buffer = device.createBuffer({
        byteLength: layout.byteLength,
        usage: Buffer.COPY_DST | Buffer.MAP_READ
      });
      try {
        texture.readBuffer({width: SIZE, height: SIZE}, buffer);
        const bytes = await buffer.readAsync();
        const pixels = new Uint8Array(SIZE * SIZE * 4);
        for (let row = 0; row < SIZE; row++)
          pixels.set(
            bytes.subarray(row * layout.bytesPerRow, row * layout.bytesPerRow + SIZE * 4),
            row * SIZE * 4
          );
        return pixels;
      } finally {
        buffer.destroy();
      }
    };
    try {
      const lightingProps = () => ({
        ambientLight: new AmbientLight({color: [255, 255, 255], intensity: 1}),
        sun: new DirectionalLight({color: [255, 255, 255], intensity: 1, direction: [-1, 0, -1]})
      });
      const lit = await render(new WGSLShadowLightingEffect(lightingProps(), {lights: []}));
      const effect = new WGSLShadowLightingEffect(lightingProps(), {
        lights: [{direction: [-1, 0, -1]}],
        shadowColor: [0, 0, 0, 0.8]
      });
      effect.id = 'shadow-light';
      const shadowed = await render(effect);
      expect(
        effect.isShadowActive,
        `${effect.skipReason} participants=${effect.participantLayerCount} frames=${effect.renderedFrameCount}`
      ).toBe(true);
      let darker = 0;
      for (let pixel = 0; pixel < SIZE * SIZE; pixel++) {
        const before = lit[pixel * 4] + lit[pixel * 4 + 1] + lit[pixel * 4 + 2];
        const after = shadowed[pixel * 4] + shadowed[pixel * 4 + 1] + shadowed[pixel * 4 + 2];
        if (after < before - 60) darker++;
      }
      expect(darker).toBeGreaterThan(40);
    } finally {
      deck.finalize();
      restore();
      ground.destroy();
      tower.destroy();
      mask.destroy();
      towerMask.destroy();
      framebuffer.destroy();
      texture.destroy();
      parent.remove();
    }
  }, 90_000);
}
