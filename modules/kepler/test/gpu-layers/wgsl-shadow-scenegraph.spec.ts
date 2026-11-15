// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Deck, MapView, type Layer} from '@deck.gl/core';
import {ScenegraphLayer} from '@deck.gl/mesh-layers';
import {
  WGSLShadowLightingEffect,
  installScenegraphShadowPatch
} from '../../../../examples/deck/kepler-compute/gpu-layers/index';
import {AmbientLight, DirectionalLight} from '@deck.gl/core';
import {Buffer, Texture, type Device} from '@luma.gl/core';
import {areaLighting} from '@luma.gl/shaders';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';

const SIZE = 128;

/** One upright triangle, in meters, as an in-memory glTF 2.0 asset. */
function makeTriangleGltf(): unknown {
  const bytes = new ArrayBuffer(72);
  const view = new Float32Array(bytes);
  // positions then normals; the triangle stands in the x-z plane facing -y.
  view.set([-20, 0, 0, 20, 0, 0, 0, 0, 40, 0, -1, 0, 0, -1, 0, 0, -1, 0]);
  return {
    json: {
      asset: {version: '2.0'},
      scene: 0,
      scenes: [{nodes: [0]}],
      nodes: [{mesh: 0}],
      meshes: [{primitives: [{attributes: {POSITION: 0, NORMAL: 1}, material: 0}]}],
      materials: [{pbrMetallicRoughness: {baseColorFactor: [0.8, 0.8, 0.8, 1], metallicFactor: 0}}],
      accessors: [
        {
          bufferView: 0,
          componentType: 5126,
          count: 3,
          type: 'VEC3',
          min: [-20, 0, 0],
          max: [20, 0, 40]
        },
        {bufferView: 1, componentType: 5126, count: 3, type: 'VEC3'}
      ],
      bufferViews: [
        {buffer: 0, byteOffset: 0, byteLength: 36},
        {buffer: 0, byteOffset: 36, byteLength: 36}
      ],
      buffers: [{byteLength: 72}]
    },
    buffers: [{arrayBuffer: bytes, byteOffset: 0, byteLength: 72}]
  };
}

test('scenegraph layers compile with the WGSL shadow patch and join the shadow pass', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU unavailable');
  // Same workaround as the Kepler host: Deck 9.4's PBR wrapper omits the area lighting module.
  const originalGetShaders = ScenegraphLayer.prototype.getShaders;
  ScenegraphLayer.prototype.getShaders = function getShaders(this: ScenegraphLayer) {
    const shaders = originalGetShaders.call(this) as {modules: {name: string}[]};
    shaders.modules = [...shaders.modules, areaLighting];
    return shaders;
  } as typeof ScenegraphLayer.prototype.getShaders;
  installScenegraphShadowPatch();
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
  let deck: Deck<MapView> | undefined;
  let error: Error | null = null;
  let frames = 0;
  let restore = () => {};
  try {
    deck = new Deck({
      device,
      parent,
      width: SIZE,
      height: SIZE,
      useDevicePixels: false,
      _framebuffer: framebuffer,
      views: new MapView({id: 'main'}),
      initialViewState: {longitude: 0, latitude: 0, zoom: 17, pitch: 45, bearing: 0},
      layers: [],
      onError: next => {
        error = next;
      },
      onAfterRender: () => {
        frames++;
      }
    });
    const gltf = makeTriangleGltf();
    for (const lighting of ['pbr', 'flat'] as const) {
      const layers: Layer[] = [
        new ScenegraphLayer({
          id: `scenegraph-${lighting}`,
          data: [{position: [0, 0, 0]}],
          scenegraph: gltf as never,
          getPosition: (row: {position: number[]}) => row.position as [number, number, number],
          getColor: [255, 255, 255, 255],
          pickable: true,
          sizeScale: 1,
          _lighting: lighting
        })
      ];
      const effect = new WGSLShadowLightingEffect(
        {
          ambientLight: new AmbientLight({color: [255, 255, 255], intensity: 1}),
          sun: new DirectionalLight({color: [255, 255, 255], intensity: 1, direction: [-1, 0, -1]})
        },
        {lights: [{direction: [-1, 0, -1]}], shadowColor: [0, 0, 0, 0.8]}
      );
      effect.id = `shadow-${lighting}`;
      deck.setProps({layers, effects: [effect as never]});
      const deadline = performance.now() + 15000;
      const ready = () =>
        Boolean(
          deck?.isInitialized &&
            layers[0].getModels().length > 0 &&
            layers[0].getModels().every(model => model.pipeline.linkStatus === 'success')
        );
      while (!ready() && performance.now() < deadline) {
        if (error) throw error;
        await new Promise(resolve => requestAnimationFrame(resolve));
      }
      if (error) throw error;
      expect(ready(), `${lighting} scenegraph pipelines linked`).toBe(true);
      for (let frame = 0; frame < 2; frame++) {
        const previous = frames;
        deck.redraw('scenegraph shadow');
        while (frames <= previous && performance.now() < deadline) {
          if (error) throw error;
          await new Promise(resolve => requestAnimationFrame(resolve));
        }
      }
      if (error) throw error;
      expect(effect.isShadowActive, effect.skipReason ?? '').toBe(true);
      expect(effect.participantLayerCount).toBe(1);
      const pixels = await readPixels(device, texture);
      let covered = 0;
      for (let pixel = 0; pixel < SIZE * SIZE; pixel++) if (pixels[pixel * 4 + 3]) covered++;
      expect(covered, `${lighting} scenegraph drew pixels`).toBeGreaterThan(50);
      // The color/shadow passes use depth24plus; Deck's picking framebuffer uses depth16unorm.
      // Pick the triangle interior, then draw again to exercise attachment changes in both directions.
      const [pickX, pickY] = deck.getViewports()[0].project([0, 0, 12]);
      const picked = await deck.pickObjectAsync({x: pickX, y: pickY, radius: 2});
      expect(picked?.index, `${lighting} scenegraph remains pickable after shadows`).toBe(0);
      deck.redraw('scenegraph after picking');
      await readPixels(device, texture);
      if (error) throw error;
    }
    deck.finalize();
    deck = undefined;
  } finally {
    deck?.finalize();
    restore();
    ScenegraphLayer.prototype.getShaders = originalGetShaders;
    framebuffer.destroy();
    texture.destroy();
    parent.remove();
  }
}, 60_000);

async function readPixels(device: Device, texture: Texture): Promise<Uint8Array> {
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
}
