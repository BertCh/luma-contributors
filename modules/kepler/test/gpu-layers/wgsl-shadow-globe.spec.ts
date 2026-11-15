// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  AmbientLight,
  DirectionalLight,
  Deck,
  _GlobeView as GlobeView,
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
      stats: effect.getViewportShadowStats(),
      shadowMapSizes: effect.getShadowMaps().map(map => [map.width, map.height]),
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

/** Pixels that got darker by a lot (shadow core) and by a little (soft edge). */
function measureShadow(lit: Uint8Array, shadowed: Uint8Array) {
  let core = 0;
  let edge = 0;
  for (let pixel = 0; pixel < WIDTH * HEIGHT; pixel++) {
    const before = lit[pixel * 4] + lit[pixel * 4 + 1] + lit[pixel * 4 + 2];
    const after = shadowed[pixel * 4] + shadowed[pixel * 4 + 1] + shadowed[pixel * 4 + 2];
    const drop = before - after;
    if (drop > 60) core++;
    else if (drop > 12) edge++;
  }
  return {core, edge};
}

const GLOBE_SCENE = {
  groundRadius: 900_000,
  towerRadius: 60_000,
  towerHeight: 900_000,
  // Globe common space: up at longitude 0, latitude 0 is -y.
  lightDirection: [-1, 1.6, 0] as [number, number, number],
  views: new GlobeView({id: 'globe'}),
  initialViewState: {longitude: 0, latitude: 0, zoom: 2.2, pitch: 0, bearing: 0}
};

test('globe shadow maps use the resolution budget and sharpen the tower shadow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU unavailable');
  const legacy = await renderScene(device, {
    ...GLOBE_SCENE,
    effectOptions: {globeShadowMapSize: 0}
  });
  const budget = await renderScene(device, GLOBE_SCENE);
  expect(legacy.effect.isShadowActive, legacy.effect.skipReason ?? '').toBe(true);
  expect(budget.effect.isShadowActive, budget.effect.skipReason ?? '').toBe(true);
  expect(budget.effect.fitting).toEqual({globe: 'globe'});

  const legacyStats = legacy.effect.stats['globe'];
  const budgetStats = budget.effect.stats['globe'];
  expect(legacyStats.shadowMapSize).toEqual([WIDTH, HEIGHT]);
  expect(legacy.effect.shadowMapSizes[0]).toEqual([WIDTH, HEIGHT]);
  // 256 x 128 viewport: the default 2048 budget enlarges the map 8x per axis.
  expect(budgetStats.resolutionScale).toBeGreaterThanOrEqual(7.9);
  expect(budget.effect.shadowMapSizes[0][0]).toBe(budgetStats.shadowMapSize[0]);
  expect(budget.effect.shadowMapSizes[0][0]).toBeGreaterThanOrEqual(WIDTH * 7.9);
  // Density versus the unrotated, viewport-sized map the effect used before this change.
  const baseline = budgetStats.baselineTexelsPerMeter!;
  expect(legacyStats.texelsPerMeter!).toBeGreaterThanOrEqual(baseline * 0.999);
  expect(budgetStats.texelsPerMeter!).toBeGreaterThan(baseline * 7.9);

  const legacyShadow = measureShadow(legacy.litPixels, legacy.shadowPixels);
  const budgetShadow = measureShadow(budget.litPixels, budget.shadowPixels);
  console.log(
    `globe shadow: legacy ${JSON.stringify(legacyShadow)} ${legacyStats.texelsPerMeter!.toFixed(5)} tx/m; ` +
      `budget ${JSON.stringify(budgetShadow)} ${budgetStats.texelsPerMeter!.toFixed(5)} tx/m`
  );
  // The tower shadow is still there with the same footprint (within 15%) ...
  expect(budgetShadow.core).toBeGreaterThan(40);
  expect(Math.abs(budgetShadow.core - legacyShadow.core)).toBeLessThan(
    legacyShadow.core * 0.15 + 8
  );
  // ... and its blended edge band is not wider than with the coarse map.
  expect(budgetShadow.edge).toBeLessThanOrEqual(legacyShadow.edge);
}, 120_000);
