// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  AmbientLight,
  DirectionalLight,
  Deck,
  _GlobeView as GlobeView,
  MapView,
  OrbitView,
  type Layer,
  type View
} from '@deck.gl/core';
import {
  GPUMeshColumnLayer,
  WGSLShadowLightingEffect
} from '../../../../examples/deck/kepler-compute/gpu-layers/index';
import {Buffer, Texture, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';

const WIDTH = 256;
const HEIGHT = 128;
const MATERIAL = {ambient: 0.6, diffuse: 0.6, shininess: 0, specularColor: [0, 0, 0]} as const;

function makeFootprint(device: Device, radius: number): GPUData<'float32x2'> {
  const values = [0, 0];
  for (let index = 0; index < 6; index++) {
    const angle = (index / 6) * Math.PI * 2;
    values.push(Math.cos(angle) * radius, Math.sin(angle) * radius);
  }
  return new GPUData({
    format: 'float32x2',
    length: 6,
    byteOffset: 8,
    ownsBuffer: false,
    buffer: device.createBuffer({
      data: new Float32Array(values),
      usage: Buffer.STORAGE | Buffer.COPY_DST
    })
  });
}

function makeVector<Format extends 'float32x2' | 'float32' | 'unorm8x4' | 'uint32'>(
  device: Device,
  format: Format,
  row: number[]
): GPUVector<Format> {
  const data =
    format === 'uint32'
      ? new Uint32Array(row)
      : format === 'unorm8x4'
        ? new Uint8Array(row)
        : new Float32Array(row);
  return new GPUVector({
    type: 'data',
    format,
    ownsData: false,
    data: [
      new GPUData({
        format,
        length: 1,
        ownsBuffer: false,
        buffer: device.createBuffer({
          data,
          usage: Buffer.VERTEX | Buffer.STORAGE | Buffer.COPY_DST
        })
      })
    ]
  });
}

type Scene = {
  views: View | View[];
  initialViewState: Record<string, unknown>;
  groundRadius: number;
  towerRadius: number;
  towerHeight: number;
  coordinateSystem?: number;
  /** Tower layer id, to exercise the editor-layer exclusion. */
  towerId?: string;
  lightDirection: [number, number, number];
  effectOptions?: Record<string, unknown>;
};

function lightingProps(direction: number[]) {
  return {
    ambientLight: new AmbientLight({color: [255, 255, 255], intensity: 1}),
    sun: new DirectionalLight({color: [255, 255, 255], intensity: 1, direction})
  };
}

async function renderScene(device: Device, scene: Scene) {
  const parent = document.createElement('div');
  document.body.append(parent);
  const texture = device.createTexture({
    width: WIDTH,
    height: HEIGHT,
    format: 'rgba8unorm',
    usage: Texture.RENDER | Texture.COPY_SRC
  });
  const framebuffer = device.createFramebuffer({
    width: WIDTH,
    height: HEIGHT,
    colorAttachments: [texture],
    depthStencilAttachment: 'depth24plus'
  });
  const position = makeVector(device, 'float32x2', [0, 0]);
  const white = makeVector(device, 'unorm8x4', [220, 220, 220, 255]);
  const mask = makeVector(device, 'uint32', [1]);
  const coverage = makeVector(device, 'float32', [1]);
  const groundHeight = makeVector(device, 'float32', [0]);
  const towerHeight = makeVector(device, 'float32', [scene.towerHeight]);
  const groundFootprint = makeFootprint(device, scene.groundRadius);
  const towerFootprint = makeFootprint(device, scene.towerRadius);
  let error: Error | null = null;
  let frames = 0;
  let restore = () => {};
  const deck = new Deck({
    device,
    parent,
    width: WIDTH,
    height: HEIGHT,
    useDevicePixels: false,
    _framebuffer: framebuffer,
    views: scene.views,
    viewState: scene.initialViewState,
    layers: [],
    onError: next => {
      error = next;
    },
    onAfterRender: () => {
      frames++;
    }
  } as never);
  const makeLayers = (): Layer[] => [
    new GPUMeshColumnLayer({
      id: 'ground',
      vertices: groundFootprint,
      edgeDistance: scene.groundRadius * Math.cos(Math.PI / 6),
      getPosition: position,
      getFillColor: white,
      getElevation: groundHeight,
      getCoverage: coverage,
      selectionMask: mask,
      extruded: true,
      coordinateSystem: scene.coordinateSystem,
      material: MATERIAL
    } as never),
    new GPUMeshColumnLayer({
      id: scene.towerId ?? 'tower',
      vertices: towerFootprint,
      edgeDistance: scene.towerRadius * Math.cos(Math.PI / 6),
      getPosition: position,
      getFillColor: white,
      getElevation: towerHeight,
      getCoverage: coverage,
      selectionMask: mask,
      extruded: true,
      coordinateSystem: scene.coordinateSystem,
      material: MATERIAL
    } as never)
  ];
  const render = async (effect: WGSLShadowLightingEffect): Promise<Uint8Array> => {
    const layers = makeLayers();
    deck.setProps({layers, effects: [effect as never]});
    const deadline = performance.now() + 15000;
    while (
      performance.now() < deadline &&
      !(
        deck.isInitialized &&
        layers.every(
          layer =>
            layer.state && layer.getModels().every(model => model.pipeline.linkStatus === 'success')
        )
      )
    ) {
      if (error) throw error;
      await new Promise(resolve => requestAnimationFrame(resolve));
    }
    if (error) throw error;
    for (let frame = 0; frame < 2; frame++) {
      const previous = frames;
      deck.redraw('shadow views');
      while (frames <= previous && performance.now() < deadline) {
        if (error) throw error;
        await new Promise(resolve => requestAnimationFrame(resolve));
      }
    }
    if (error) throw error;
    const layout = texture.computeMemoryLayout({width: WIDTH, height: HEIGHT});
    const buffer = device.createBuffer({
      byteLength: layout.byteLength,
      usage: Buffer.COPY_DST | Buffer.MAP_READ
    });
    try {
      texture.readBuffer({width: WIDTH, height: HEIGHT}, buffer);
      const bytes = await buffer.readAsync();
      const pixels = new Uint8Array(WIDTH * HEIGHT * 4);
      for (let row = 0; row < HEIGHT; row++)
        pixels.set(
          bytes.subarray(row * layout.bytesPerRow, row * layout.bytesPerRow + WIDTH * 4),
          row * WIDTH * 4
        );
      return pixels;
    } finally {
      buffer.destroy();
    }
  };
  try {
    const lit = new WGSLShadowLightingEffect(lightingProps(scene.lightDirection), {lights: []});
    lit.id = 'lit-light';
    const litPixels = await render(lit);
    const effect = new WGSLShadowLightingEffect(lightingProps(scene.lightDirection), {
      lights: [{direction: scene.lightDirection}],
      shadowColor: [0, 0, 0, 0.8],
      ...scene.effectOptions
    });
    effect.id = 'shadow-light';
    const shadowPixels = await render(effect);
    // Snapshot before deck.finalize() cleans the effect up.
    const effectState = {
      isShadowActive: effect.isShadowActive,
      skipReason: effect.skipReason,
      activeViewportCount: effect.activeViewportCount,
      participantLayerCount: effect.participantLayerCount,
      fitting: effect.getViewportFitting(),
      shadowMapCount: effect.getShadowMaps().length
    };
    return {effect: effectState, litPixels, shadowPixels};
  } finally {
    deck.finalize();
    restore();
    framebuffer.destroy();
    texture.destroy();
    parent.remove();
  }
}

/** Counts pixels that got clearly darker inside a pixel rectangle; also counts brighter ones. */
function compareRegion(
  lit: Uint8Array,
  shadowed: Uint8Array,
  region = {x0: 0, x1: WIDTH}
): {darker: number; brighter: number} {
  let darker = 0;
  let brighter = 0;
  for (let row = 0; row < HEIGHT; row++) {
    for (let column = region.x0; column < region.x1; column++) {
      const pixel = row * WIDTH + column;
      const before = lit[pixel * 4] + lit[pixel * 4 + 1] + lit[pixel * 4 + 2];
      const after = shadowed[pixel * 4] + shadowed[pixel * 4 + 1] + shadowed[pixel * 4 + 2];
      if (after < before - 60) darker++;
      else if (after > before + 30) brighter++;
    }
  }
  return {darker, brighter};
}

const MAP_STATE = {longitude: 0, latitude: 0, zoom: 17, pitch: 45, bearing: 0};
const MAP_SCENE = {
  groundRadius: 40,
  towerRadius: 4,
  towerHeight: 30,
  lightDirection: [-1, 0, -1] as [number, number, number]
};

test('split map viewports each get their own shadow map', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU unavailable');
  const {effect, litPixels, shadowPixels} = await renderScene(device, {
    ...MAP_SCENE,
    views: [
      new MapView({id: 'left', x: 0, width: '50%'}),
      new MapView({id: 'right', x: '50%', width: '50%'})
    ],
    initialViewState: {left: MAP_STATE, right: {...MAP_STATE, bearing: 20}}
  });
  expect(effect.isShadowActive, effect.skipReason ?? '').toBe(true);
  expect(effect.activeViewportCount).toBe(2);
  expect(Object.keys(effect.fitting).sort()).toEqual(['left', 'right']);
  expect(effect.shadowMapCount).toBe(2);
  const left = compareRegion(litPixels, shadowPixels, {x0: 0, x1: WIDTH / 2});
  const right = compareRegion(litPixels, shadowPixels, {x0: WIDTH / 2, x1: WIDTH});
  expect(left.darker).toBeGreaterThan(20);
  expect(right.darker).toBeGreaterThan(20);
  expect(left.brighter + right.brighter).toBe(0);
}, 90_000);

test('orbit viewports fit the light frustum to the scene bounds', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU unavailable');
  const {effect, litPixels, shadowPixels} = await renderScene(device, {
    groundRadius: 40,
    towerRadius: 4,
    towerHeight: 30,
    lightDirection: [-1, 0, -1],
    views: new OrbitView({id: 'orbit', orbitAxis: 'Z', fovy: 30}),
    initialViewState: {target: [0, 0, 0], rotationX: 40, rotationOrbit: 20, zoom: 2.5},
    coordinateSystem: 'cartesian',
    effectOptions: {
      sceneBounds: [
        [-40, -40, 0],
        [40, 40, 30]
      ]
    }
  });
  expect(effect.isShadowActive, effect.skipReason ?? '').toBe(true);
  expect(effect.fitting).toEqual({orbit: 'scene-bounds'});
  const result = compareRegion(litPixels, shadowPixels);
  expect(result.darker).toBeGreaterThan(40);
  expect(result.brighter).toBe(0);
}, 90_000);

test('orbit viewports without bounds clamp the far plane and still shadow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU unavailable');
  const {effect, litPixels, shadowPixels} = await renderScene(device, {
    groundRadius: 40,
    towerRadius: 4,
    towerHeight: 30,
    lightDirection: [-1, 0, -1],
    views: new OrbitView({id: 'orbit', orbitAxis: 'Z', fovy: 30}),
    initialViewState: {target: [0, 0, 0], rotationX: 40, rotationOrbit: 20, zoom: 2.5},
    coordinateSystem: 'cartesian'
  });
  expect(effect.isShadowActive, effect.skipReason ?? '').toBe(true);
  expect(effect.fitting).toEqual({orbit: 'view-frustum'});
  expect(compareRegion(litPixels, shadowPixels).darker).toBeGreaterThan(20);
}, 90_000);

test('globe viewports shadow with a sphere-fitted light projection', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU unavailable');
  const {effect, litPixels, shadowPixels} = await renderScene(device, {
    groundRadius: 900_000,
    towerRadius: 60_000,
    towerHeight: 900_000,
    // Globe common space: up at longitude 0, latitude 0 is -y.
    lightDirection: [-1, 1.6, 0],
    views: new GlobeView({id: 'globe'}),
    initialViewState: {longitude: 0, latitude: 0, zoom: 2.2, pitch: 0, bearing: 0}
  });
  expect(effect.isShadowActive, effect.skipReason ?? '').toBe(true);
  expect(effect.fitting).toEqual({globe: 'globe'});
  const result = compareRegion(litPixels, shadowPixels);
  expect(result.darker).toBeGreaterThan(40);
  expect(result.brighter).toBe(0);
}, 90_000);

test('editor layers neither cast nor receive shadows, like Kepler', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU unavailable');
  const normal = await renderScene(device, {
    ...MAP_SCENE,
    views: new MapView({id: 'main'}),
    initialViewState: MAP_STATE
  });
  const excluded = await renderScene(device, {
    ...MAP_SCENE,
    views: new MapView({id: 'main'}),
    initialViewState: MAP_STATE,
    towerId: 'kepler_editor_layer-sublayer',
    effectOptions: {excludedLayerIdPrefixes: ['kepler_editor_layer']}
  });
  expect(normal.effect.participantLayerCount).toBe(2);
  expect(excluded.effect.participantLayerCount).toBe(1);
  expect(compareRegion(normal.litPixels, normal.shadowPixels).darker).toBeGreaterThan(40);
  // The excluded tower casts nothing, so the ground stays lit.
  expect(compareRegion(excluded.litPixels, excluded.shadowPixels).darker).toBeLessThan(10);
}, 90_000);
