// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  Deck,
  MapView,
  AmbientLight,
  DirectionalLight,
  LightingEffect,
  type Layer
} from '@deck.gl/core';
import {SolidPolygonLayer} from '@deck.gl/layers';
import {GPUSolidPolygonLayer} from '../../../../examples/deck/kepler-compute/gpu-layers/index';
import {
  makeKeplerGeometryGPUVectors,
  prepareKeplerGeometrySource
} from '../../../../examples/deck/kepler-compute/src/geometry-preparation/geometry-gpu-vectors';
import {Buffer, Texture, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test, vi} from 'vitest';

const WIDTH = 256;
const HEIGHT = 128;

for (const vectorInputs of [false, true]) {
  test(`extruded rings preserve geometry, lighting and picking with ${vectorInputs ? 'vector' : 'constant'} inputs`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) throw new Error('Real WebGPU device required');
    const ring = (longitude: number, latitude: number, halfWidth: number, halfHeight: number) =>
      [
        [longitude - halfWidth, latitude - halfHeight, 10],
        [longitude + halfWidth, latitude - halfHeight, 10],
        [longitude + halfWidth, latitude + halfHeight, 10],
        [longitude - halfWidth, latitude + halfHeight, 10],
        [longitude - halfWidth, latitude - halfHeight, 10]
      ].map(point => point.map(Math.fround));
    const shell = ring(0, 45, 0.0012, 0.0008);
    // Deliberately use the shell's winding for the hole; explicit orientation metadata corrects it.
    const hole = ring(0, 45, 0.0004, 0.00025);
    const left = ring(-0.0035, 45, 0.00055, 0.00065);
    const right = ring(0.0035, 45, 0.00055, 0.00065);
    const polygons = [[shell, hole], [left], [right]];
    const geometry = makeKeplerGeometryGPUVectors(
      device,
      prepareKeplerGeometrySource([
        {id: 'first', rows: [{type: 'Polygon', coordinates: polygons[0]}]},
        {id: 'second', rows: [{type: 'MultiPolygon', coordinates: [polygons[1], polygons[2]]}]}
      ])
    );
    const elevationOffsets = makeVector(device, 'float32', [[20], [30]]);
    const elevations = makeVector(device, 'float32', [[50], [30]]);
    const colors = makeVector(device, 'unorm8x4', [
      [200, 70, 50, 255],
      [50, 140, 210, 255]
    ]);
    const mask = makeVector(device, 'uint32', [[1], [1]]);
    const borrowedBuffers = [
      ...geometry.buffers,
      ...elevationOffsets.data.map(chunk => chunk.buffer as Buffer),
      ...elevations.data.map(chunk => chunk.buffer as Buffer),
      ...colors.data.map(chunk => chunk.buffer as Buffer),
      ...mask.data.map(chunk => chunk.buffer as Buffer)
    ];
    const reads = borrowedBuffers.map(buffer => vi.spyOn(buffer, 'readAsync'));
    const material = {
      ambient: 0.25,
      diffuse: 0.75,
      shininess: 32,
      specularColor: [0, 0, 0] as [number, number, number]
    };
    const reference = new SolidPolygonLayer({
      id: 'native-extrusion',
      data: polygons,
      getPolygon: (polygon, {index}) =>
        polygon.map(path =>
          path.map(([longitude, latitude, altitude]) => [
            longitude,
            latitude,
            altitude + (vectorInputs && index !== 0 ? 30 : 20)
          ])
        ),
      getElevation: (_polygon, {index}) => (vectorInputs && index !== 0 ? 60 : 100),
      getFillColor: (_polygon, {index}) =>
        vectorInputs && index !== 0 ? [50, 140, 210, 255] : [200, 70, 50, 255],
      extruded: true,
      material,
      pickable: true
    });
    const layer = new GPUSolidPolygonLayer({
      id: 'resident-extrusion',
      ...geometry.polygons,
      getFillColor: vectorInputs ? colors : [200, 70, 50, 255],
      getElevation: vectorInputs ? elevations : 50,
      elevationScale: 2,
      getElevationOffset: vectorInputs ? elevationOffsets : 20,
      selectionMask: mask,
      extruded: true,
      material,
      pickable: true,
      extrusionRings: {
        positions: geometry.polygonOutlines.getPath,
        sourceRowIds: geometry.polygonOutlines.sourceRowIds,
        sideOrientations: geometry.polygonOutlines.sideOrientations
      }
    });
    const pickLocations = [
      [-0.0008, 45, 130],
      [-0.0035, 45, vectorInputs ? 100 : 130],
      [0.0035, 45, vectorInputs ? 100 : 130],
      [0, Math.fround(45 - 0.0008), 80]
    ];
    const nativePicks: Array<number | undefined> = [];
    try {
      const nativePixels = await render(device, reference, async deck => {
        for (const point of pickLocations) {
          const [horizontal, vertical] = deck.getViewports()[0].project(point);
          nativePicks.push((await deck.pickObjectAsync({x: horizontal, y: vertical}))?.index);
        }
      });
      expect(nativePicks).toEqual([0, 1, 2, 0]);
      const residentPixels = await render(device, layer, async deck => {
        expect(layer.getModels()).toHaveLength(4);
        expect(layer.extrusionByteLength).toBeGreaterThan(0);
        const models = layer.getModels();
        for (const [index, point] of pickLocations.entries()) {
          const [horizontal, vertical] = deck.getViewports()[0].project(point);
          const picked = await deck.pickObjectAsync({x: horizontal, y: vertical});
          const sourceRow = index === 1 || index === 2 ? 1 : 0;
          expect(picked?.index).toBe(sourceRow);
          expect(picked).toHaveProperty('gpuVector', {
            rowIndex: sourceRow,
            batchIndex: sourceRow,
            batchRowIndex: 0
          });
        }
        let activeLayer = layer;
        if (vectorInputs) {
          (elevations.data[0].buffer as Buffer).write(new Float32Array([65]));
        } else {
          activeLayer = layer.clone({getElevation: 65});
          deck.setProps({layers: [activeLayer]});
          await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
        }
        const [raisedHorizontal, raisedVertical] = deck
          .getViewports()[0]
          .project([-0.0008, 45, 160]);
        expect((await deck.pickObjectAsync({x: raisedHorizontal, y: raisedVertical}))?.index).toBe(
          0
        );
        expect(activeLayer.getModels()).toEqual(models);
        (mask.data[0].buffer as Buffer).write(new Uint32Array([0]));
        expect(await deck.pickObjectAsync({x: raisedHorizontal, y: raisedVertical})).toBeNull();
        const [secondHorizontal, secondVertical] = deck
          .getViewports()[0]
          .project([-0.0035, 45, vectorInputs ? 100 : 160]);
        expect((await deck.pickObjectAsync({x: secondHorizontal, y: secondVertical}))?.index).toBe(
          1
        );
        for (const read of reads) expect(read).not.toHaveBeenCalled();
      });
      let intersection = 0;
      let union = 0;
      let error = 0;
      for (let offset = 0; offset < nativePixels.length; offset += 4) {
        const nativeVisible = nativePixels[offset + 3] > 100;
        const residentVisible = residentPixels[offset + 3] > 100;
        if (nativeVisible || residentVisible) union++;
        if (nativeVisible && residentVisible) {
          intersection++;
          for (let channel = 0; channel < 3; channel++)
            error += Math.abs(nativePixels[offset + channel] - residentPixels[offset + channel]);
        }
      }
      expect(intersection).toBeGreaterThan(200);
      expect(intersection / union, 'native/resident lit solid footprint').toBeGreaterThan(0.85);
      expect(error / (intersection * 3), 'native Gouraud lighting agreement').toBeLessThan(15);
      for (const buffer of borrowedBuffers) expect(buffer.destroyed).toBe(false);
    } finally {
      reads.forEach(read => read.mockRestore());
      const fence = device.createFence();
      await fence.signaled;
      fence.destroy();
      geometry.destroy();
      elevations.destroy();
      elevationOffsets.destroy();
      colors.destroy();
      mask.destroy();
    }
  }, 30000);
}

for (const {wireframe, filled} of [
  {wireframe: false, filled: true},
  {wireframe: true, filled: false},
  {wireframe: true, filled: true}
]) {
  test(`polygon offset supports ${wireframe ? `independent lit wireframe with filled=${filled}` : 'flat XYZ surfaces'}`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) throw new Error('Real WebGPU device required');
    const ring = [
      [-0.0012, 44.9992, 10],
      [0.0012, 44.9992, 10],
      [0.0012, 45.0008, 10],
      [-0.0012, 45.0008, 10],
      [-0.0012, 44.9992, 10]
    ].map(point => point.map(Math.fround));
    const geometry = makeKeplerGeometryGPUVectors(
      device,
      prepareKeplerGeometrySource([{id: 'offset', rows: [{type: 'Polygon', coordinates: [ring]}]}])
    );
    const elevationOffsets = makeVector(device, 'float32', [[30]]);
    const lineColors = makeVector(device, 'unorm8x4', [[20, 180, 60, 255]]);
    const mask = makeVector(device, 'uint32', [[1]]);
    const borrowedBuffers = [
      ...geometry.buffers,
      ...elevationOffsets.data.map(chunk => chunk.buffer as Buffer),
      ...lineColors.data.map(chunk => chunk.buffer as Buffer),
      ...mask.data.map(chunk => chunk.buffer as Buffer)
    ];
    const reads = borrowedBuffers.map(buffer => vi.spyOn(buffer, 'readAsync'));
    const material = {
      ambient: 0.25,
      diffuse: 0.75,
      shininess: 32,
      specularColor: [0, 0, 0] as [number, number, number]
    };
    const native = new SolidPolygonLayer({
      id: 'native-offset',
      data: [ring],
      getPolygon: path =>
        path.map(([longitude, latitude, altitude]) => [longitude, latitude, altitude + 30]),
      getElevation: 50,
      elevationScale: 2,
      getFillColor: [200, 70, 50, 255],
      getLineColor: [20, 180, 60, 255],
      filled,
      extruded: wireframe,
      wireframe,
      material,
      pickable: true
    });
    const layer = new GPUSolidPolygonLayer({
      id: 'resident-offset',
      ...geometry.polygons,
      getElevation: 50,
      elevationScale: 2,
      getElevationOffset: elevationOffsets,
      getFillColor: [200, 70, 50, 255],
      getLineColor: lineColors,
      filled,
      extruded: wireframe,
      wireframe,
      material,
      selectionMask: mask,
      pickable: true,
      extrusionRings: {
        positions: geometry.polygonOutlines.getPath,
        sourceRowIds: geometry.polygonOutlines.sourceRowIds,
        sideOrientations: geometry.polygonOutlines.sideOrientations
      }
    });
    const location = wireframe ? [ring[0][0], ring[0][1], 90] : [0, 45, 40];
    const pick = async (deck: Deck<MapView>, point: number[]) => {
      const [horizontal, vertical] = deck.getViewports()[0].project(point);
      return deck.pickObjectAsync({x: horizontal, y: vertical, radius: wireframe ? 2 : 0});
    };
    try {
      const nativePixels = await render(device, native, async deck => {
        expect((await pick(deck, location))?.index).toBe(0);
      });
      const residentPixels = await render(device, layer, async deck => {
        expect((await pick(deck, location))?.index).toBe(0);
        const models = layer.getModels();
        (elevationOffsets.data[0].buffer as Buffer).write(new Float32Array([60]));
        const raised = [location[0], location[1], location[2] + 30];
        expect((await pick(deck, raised))?.index).toBe(0);
        expect(layer.getModels()).toEqual(models);
        (mask.data[0].buffer as Buffer).write(new Uint32Array([0]));
        expect(await pick(deck, raised)).toBeNull();
        for (const read of reads) expect(read).not.toHaveBeenCalled();
      });
      let intersection = 0;
      let union = 0;
      let error = 0;
      for (let offset = 0; offset < nativePixels.length; offset += 4) {
        const nativeVisible = nativePixels[offset + 3] > 100;
        const residentVisible = residentPixels[offset + 3] > 100;
        if (nativeVisible || residentVisible) union++;
        if (nativeVisible && residentVisible) {
          intersection++;
          for (let channel = 0; channel < 3; channel++)
            error += Math.abs(nativePixels[offset + channel] - residentPixels[offset + channel]);
        }
      }
      expect(intersection).toBeGreaterThan(wireframe ? 40 : 200);
      expect(intersection / union).toBeGreaterThan(wireframe ? 0.7 : 0.85);
      expect(error / (intersection * 3)).toBeLessThan(15);
      for (const buffer of borrowedBuffers) expect(buffer.destroyed).toBe(false);
    } finally {
      reads.forEach(read => read.mockRestore());
      const fence = device.createFence();
      await fence.signaled;
      fence.destroy();
      geometry.destroy();
      elevationOffsets.destroy();
      lineColors.destroy();
      mask.destroy();
    }
  }, 30000);
}

function makeVector<Format extends 'float32' | 'uint32' | 'unorm8x4'>(
  device: Device,
  format: Format,
  chunks: number[][]
): GPUVector<Format> {
  return new GPUVector({
    type: 'data',
    format,
    data: chunks.map(
      values =>
        new GPUData({
          format,
          length: format === 'unorm8x4' ? values.length / 4 : values.length,
          ownsBuffer: true,
          buffer: device.createBuffer({
            data:
              format === 'float32'
                ? new Float32Array(values)
                : format === 'uint32'
                  ? new Uint32Array(values)
                  : new Uint8Array(values),
            usage: Buffer.VERTEX | Buffer.STORAGE | Buffer.COPY_DST
          })
        })
    )
  });
}

async function render(
  device: Device,
  layer: Layer,
  inspect?: (deck: Deck<MapView>) => Promise<void>
): Promise<Uint8Array> {
  const parent = document.createElement('div');
  document.body.append(parent);
  const color = device.createTexture({
    width: WIDTH,
    height: HEIGHT,
    format: 'rgba8unorm',
    usage: Texture.RENDER | Texture.COPY_SRC
  });
  const framebuffer = device.createFramebuffer({
    width: WIDTH,
    height: HEIGHT,
    colorAttachments: [color],
    depthStencilAttachment: 'depth24plus'
  });
  let error: Error | null = null;
  let frameCount = 0;
  let restore = () => {};
  const deck = new Deck({
    device,
    parent,
    width: WIDTH,
    height: HEIGHT,
    useDevicePixels: false,
    _framebuffer: framebuffer,
    views: new MapView({id: 'map'}),
    initialViewState: {longitude: 0, latitude: 45, zoom: 14, pitch: 50, bearing: 20},
    layers: [layer],
    effects: [
      new LightingEffect({
        ambientLight: new AmbientLight({color: [255, 255, 255], intensity: 0.8}),
        directionalLight: new DirectionalLight({
          color: [255, 255, 255],
          intensity: 1,
          direction: [-1, -2, -3]
        })
      })
    ],
    onError: failure => {
      error = failure;
    },
    onAfterRender: () => {
      frameCount++;
    }
  });
  try {
    const deadline = performance.now() + 10000;
    while (
      (!deck.isInitialized ||
        !frameCount ||
        layer.getModels().some(model => model.pipeline.linkStatus !== 'success')) &&
      performance.now() < deadline
    ) {
      if (error) throw error;
      await new Promise(resolve => requestAnimationFrame(resolve));
    }
    if (error) throw error;
    expect(frameCount).toBeGreaterThan(0);
    const previous = frameCount;
    deck.redraw('capture native projection');
    while (frameCount === previous && performance.now() < deadline)
      await new Promise(resolve => requestAnimationFrame(resolve));
    const layout = color.computeMemoryLayout({width: WIDTH, height: HEIGHT});
    const buffer = device.createBuffer({
      byteLength: layout.byteLength,
      usage: Buffer.MAP_READ | Buffer.COPY_DST
    });
    try {
      color.readBuffer({width: WIDTH, height: HEIGHT}, buffer);
      const bytes = await buffer.readAsync();
      const pixels = new Uint8Array(WIDTH * HEIGHT * 4);
      for (let row = 0; row < HEIGHT; row++)
        pixels.set(
          bytes.subarray(row * layout.bytesPerRow, row * layout.bytesPerRow + WIDTH * 4),
          row * WIDTH * 4
        );
      await inspect?.(deck);
      return pixels;
    } finally {
      buffer.destroy();
    }
  } finally {
    deck.finalize();
    restore();
    framebuffer.destroy();
    color.destroy();
    parent.remove();
  }
}
