// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Deck, OrbitView, type Layer} from '@deck.gl/core';
import {ColumnLayer} from '@deck.gl/layers';
import {GPUMeshColumnLayer} from '../../../../examples/deck/kepler-compute/gpu-layers/index';
import {Buffer, Texture, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test, vi} from 'vitest';

const SIZE = 128;
// Deliberately irregular convex custom footprint, not a regular circle/hexagon approximation.
const VERTICES: [number, number][] = [
  [9, 0],
  [5, 7],
  [-4, 8],
  [-8, 0],
  [-5, -6],
  [4, -7]
];
const EDGE_DISTANCE =
  (Math.cos(Math.PI / VERTICES.length) *
    VERTICES.reduce((total, vertex) => total + Math.hypot(...vertex), 0)) /
  VERTICES.length;

test('resident custom-meter columns match native projected geometry and lighting across fill, stroke and extrusion', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU unavailable');
  const vertices = new GPUData({
    format: 'float32x2',
    length: VERTICES.length,
    byteOffset: 8,
    ownsBuffer: false,
    buffer: device.createBuffer({
      data: new Float32Array([0, 0, ...VERTICES.flat()]),
      usage: Buffer.STORAGE | Buffer.COPY_DST
    })
  });
  const positions = vector(device, 'float32x2', [
    [-15, 0],
    [15, 0]
  ]);
  const colors = vector(device, 'unorm8x4', [
    [200, 60, 20, 255],
    [20, 170, 70, 255]
  ]);
  const lineColors = vector(device, 'unorm8x4', [
    [30, 70, 230, 255],
    [230, 170, 20, 255]
  ]);
  const elevations = vector(device, 'float32', [[8], [14]]);
  const coverages = vector(device, 'float32', [[1], [1]]);
  const mask = vector(device, 'uint32', [[1], [1]]);
  const borrowed = [
    vertices,
    ...positions.data,
    ...colors.data,
    ...lineColors.data,
    ...elevations.data,
    ...coverages.data,
    ...mask.data
  ].map(data => data.buffer as Buffer);
  const reads = borrowed.map(buffer => vi.spyOn(buffer, 'readAsync'));
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
  let deck: Deck<OrbitView> | undefined;
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
      views: new OrbitView({id: 'main', orbitAxis: 'Z'}),
      initialViewState: {target: [0, 0, 4], rotationX: 30, rotationOrbit: 20, zoom: 1},
      layers: [],
      onError: next => {
        error = next;
      },
      onAfterRender: () => {
        frames++;
      }
    });
    const render = async (layers: Layer[]) => {
      deck!.setProps({layers});
      await waitUntil(
        () =>
          Boolean(
            deck?.isInitialized &&
              layers.every(
                layer =>
                  layer.state &&
                  layer.getModels().every(model => model.pipeline.linkStatus === 'success')
              )
          ),
        () => error
      );
      const previous = frames;
      deck!.redraw('column oracle');
      await waitUntil(
        () => frames > previous,
        () => error
      );
      return readPixels(device, texture);
    };
    const nativeData = [
      {
        position: [-15, 0, 0] as [number, number, number],
        color: [200, 60, 20, 255] as [number, number, number, number],
        lineColor: [30, 70, 230, 255] as [number, number, number, number],
        elevation: 8
      },
      {
        position: [15, 0, 0] as [number, number, number],
        color: [20, 170, 70, 255] as [number, number, number, number],
        lineColor: [230, 170, 20, 255] as [number, number, number, number],
        elevation: 14
      }
    ];
    for (const settings of [
      {extruded: false, filled: true, stroked: false, flatShading: false},
      {extruded: false, filled: true, stroked: true, flatShading: false},
      {extruded: false, filled: false, stroked: true, flatShading: false},
      {extruded: true, filled: true, stroked: true, flatShading: false},
      {extruded: true, filled: true, stroked: false, flatShading: true}
    ]) {
      const common = {
        ...settings,
        coordinateSystem: 'cartesian',
        coverage: 0.85,
        elevationScale: 1.5,
        lineWidthUnits: 'pixels' as const,
        lineWidthScale: 2,
        opacity: 0.8,
        material: true,
        pickable: true
      };
      const native = new ColumnLayer({
        id: 'native-column',
        ...common,
        radius: 1,
        radiusUnits: 'meters',
        diskResolution: 6,
        vertices: VERTICES,
        data: nativeData,
        getPosition: row => row.position,
        getFillColor: row => row.color,
        getLineColor: row => row.lineColor,
        getElevation: row => row.elevation,
        getLineWidth: 1
      });
      const expected = await render([native]);
      const resident = new GPUMeshColumnLayer({
        id: 'resident-column',
        ...common,
        vertices,
        edgeDistance: EDGE_DISTANCE,
        getPosition: positions,
        getFillColor: colors,
        getLineColor: lineColors,
        getElevation: elevations,
        getCoverage: coverages,
        selectionMask: mask,
        getLineWidth: 1
      });
      const actual = await render([resident]);
      let differingPixels = 0;
      let expectedPixels = 0;
      for (let pixel = 0; pixel < SIZE * SIZE; pixel++) {
        if (expected[pixel * 4 + 3]) expectedPixels++;
        if (
          [0, 1, 2, 3].some(
            channel => Math.abs(expected[pixel * 4 + channel] - actual[pixel * 4 + channel]) > 2
          )
        )
          differingPixels++;
      }
      expect(expectedPixels, JSON.stringify(settings)).toBeGreaterThan(100);
      expect(differingPixels, JSON.stringify(settings)).toBeLessThan(10);
    }
    const resident = new GPUMeshColumnLayer({
      id: 'resident-mask',
      vertices,
      edgeDistance: EDGE_DISTANCE,
      getPosition: positions,
      getFillColor: colors,
      getCoverage: coverages,
      getElevation: 0,
      selectionMask: mask,
      coordinateSystem: 'cartesian',
      extruded: false,
      pickable: true
    });
    await render([resident]);
    const model = resident.getModels()[0];
    const screen = deck.getViewports()[0].project([15, 0, 0]);
    expect((await deck.pickObjectAsync({x: screen[0], y: screen[1]}))?.index).toBe(1);
    expect(await deck.pickObjectAsync({x: screen[0], y: screen[1]})).toHaveProperty('gpuVector', {
      rowIndex: 1,
      batchIndex: 1,
      batchRowIndex: 0
    });
    (mask.data[1].buffer as Buffer).write(new Uint32Array([0]));
    (coverages.data[0].buffer as Buffer).write(new Float32Array([0.5]));
    await render([resident]);
    expect(await deck.pickObjectAsync({x: screen[0], y: screen[1]})).toBeNull();
    expect(resident.getModels()[0]).toBe(model);
    for (const read of reads) expect(read).not.toHaveBeenCalled();
    deck.finalize();
    deck = undefined;
    for (const buffer of borrowed) expect(buffer.destroyed).toBe(false);
  } finally {
    deck?.finalize();
    restore();
    reads.forEach(read => read.mockRestore());
    framebuffer.destroy();
    texture.destroy();
    borrowed.forEach(buffer => buffer.destroy());
    parent.remove();
  }
}, 60_000);

function vector<Format extends 'float32x2' | 'float32' | 'unorm8x4' | 'uint32'>(
  device: Device,
  format: Format,
  rows: number[][]
): GPUVector<Format> {
  return new GPUVector({
    type: 'data',
    format,
    ownsData: false,
    data: rows.map(
      row =>
        new GPUData({
          format,
          length: 1,
          ownsBuffer: false,
          buffer: device.createBuffer({
            data:
              format === 'uint32'
                ? new Uint32Array(row)
                : format === 'unorm8x4'
                  ? new Uint8Array(row)
                  : new Float32Array(row),
            usage: Buffer.VERTEX | Buffer.STORAGE | Buffer.COPY_DST
          })
        })
    )
  });
}
async function waitUntil(predicate: () => boolean, getError: () => Error | null): Promise<void> {
  const deadline = performance.now() + 10000;
  while (!predicate() && performance.now() < deadline) {
    if (getError()) throw getError();
    await new Promise(resolve => requestAnimationFrame(resolve));
  }
  if (getError()) throw getError();
  expect(predicate(), 'Deck completed requested native-column frame').toBe(true);
}
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
