// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Deck, MapView, type Layer} from '@deck.gl/core';
import {ArcLayer, PathLayer, ScatterplotLayer} from '@deck.gl/layers';
import {
  GPUArcLayer,
  GPUPathLayer,
  GPUScatterplotLayer
} from '../../../../examples/deck/kepler-compute/gpu-layers/index';
import {Buffer, Texture, type Device} from '@luma.gl/core';
import {GPUData, GPUVector, type VertexList} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test, vi} from 'vitest';

const WIDTH = 256;
const HEIGHT = 128;

for (const billboard of [false, true]) {
  test(`resident meter circles match Deck under pitch (billboard=${billboard})`, async () => {
    const device = await requireDevice();
    const positions = makePositions(device, [0, 45]);
    const props = {
      getRadius: 100,
      radiusScale: 2,
      radiusUnits: 'meters' as const,
      radiusMaxPixels: 18,
      billboard,
      getFillColor: [255, 0, 0, 255] as [number, number, number, number]
    };
    try {
      const nativePixels = await render(
        device,
        new ScatterplotLayer({
          id: 'native-point',
          data: [{position: [0, 45]}],
          getPosition: row => row.position,
          ...props
        })
      );
      const residentPixels = await render(
        device,
        new GPUScatterplotLayer({id: 'resident-point', getPosition: positions, ...props})
      );
      expectOverlap(nativePixels, residentPixels, 0.82);
    } finally {
      positions.destroy();
    }
  }, 30_000);
}

for (const billboard of [false, true]) {
  for (const lineWidthUnits of ['meters', 'pixels'] as const) {
    test(`resident hollow outlines match Deck under pitch (${lineWidthUnits}, billboard=${billboard})`, async () => {
      const device = await requireDevice();
      const positions = makePositions(device, [0, 45]);
      const props = {
        getRadius: 100,
        radiusUnits: 'meters' as const,
        radiusMaxPixels: 18,
        radiusMinPixels: 18,
        billboard,
        stroked: true,
        filled: false,
        getLineColor: [255, 0, 0, 255] as [number, number, number, number],
        getLineWidth: lineWidthUnits === 'meters' ? 10 : 3,
        lineWidthUnits,
        lineWidthScale: 2,
        lineWidthMinPixels: 4,
        lineWidthMaxPixels: 8
      };
      try {
        const nativePixels = await render(
          device,
          new ScatterplotLayer({
            id: 'native-outline',
            data: [{}],
            getPosition: (): [number, number] => [0, 45],
            ...props
          })
        );
        const residentPixels = await render(
          device,
          new GPUScatterplotLayer({
            id: 'resident-outline',
            getPosition: positions,
            ...props
          })
        );
        expectOverlap(nativePixels, residentPixels, 0.88);
        expect(residentPixels[(Math.floor(HEIGHT / 2) * WIDTH + WIDTH / 2) * 4]).toBe(0);
      } finally {
        positions.destroy();
      }
    }, 30_000);
  }
}

for (const billboard of [false, true]) {
  test(`resident XYZ circles preserve positive and negative altitude and picking under pitch (billboard=${billboard})`, async () => {
    const device = await requireDevice();
    // Native and resident receive the same float32 coordinates; no hidden XYZ conversion occurs.
    const coordinates: [number, number, number][] = [
      [Math.fround(-0.001), 45, 140],
      [Math.fround(0.001), 45, -120]
    ];
    const positions = new GPUVector({
      type: 'data',
      name: 'elevated-points',
      format: 'float32x3',
      data: coordinates.map(
        position =>
          new GPUData({
            format: 'float32x3',
            length: 1,
            ownsBuffer: true,
            buffer: device.createBuffer({
              data: Float32Array.from(position),
              usage: Buffer.VERTEX | Buffer.STORAGE
            })
          })
      )
    });
    const borrowedBuffers = positions.data.map(chunk => chunk.buffer as Buffer);
    const reads = borrowedBuffers.map(buffer => vi.spyOn(buffer, 'readAsync'));
    const props = {
      getRadius: 30,
      radiusUnits: 'meters' as const,
      radiusMinPixels: 7,
      radiusMaxPixels: 7,
      billboard,
      stroked: true,
      getLineWidth: 2,
      lineWidthUnits: 'pixels' as const,
      getFillColor: [255, 0, 0, 255] as [number, number, number, number],
      getLineColor: [255, 0, 0, 255] as [number, number, number, number],
      pickable: true
    };
    const inspectPicking = (resident: boolean) => async (deck: Deck<MapView>) => {
      const viewport = deck.getViewports()[0];
      for (const [rowIndex, position] of coordinates.entries()) {
        const [horizontal, vertical] = viewport.project(position);
        const groundPosition = viewport.project([position[0], position[1], 0]);
        expect(
          Math.abs(vertical - groundPosition[1]),
          'altitude moves the projected center'
        ).toBeGreaterThan(8);
        const picked = await deck.pickObjectAsync({x: horizontal, y: vertical});
        expect(picked?.index).toBe(rowIndex);
        if (resident)
          expect(picked).toHaveProperty('gpuVector', {
            rowIndex,
            batchIndex: rowIndex,
            batchRowIndex: 0
          });
      }
    };
    try {
      const nativePixels = await render(
        device,
        new ScatterplotLayer({
          id: 'native-xyz',
          data: coordinates,
          getPosition: position => position,
          ...props
        }),
        inspectPicking(false)
      );
      const residentPixels = await render(
        device,
        new GPUScatterplotLayer({
          id: 'resident-xyz',
          getPosition: positions,
          ...props
        }),
        inspectPicking(true)
      );
      expectOverlap(nativePixels, residentPixels, 0.88);
      for (const read of reads) expect(read).not.toHaveBeenCalled();
      for (const buffer of borrowedBuffers) expect(buffer.destroyed).toBe(false);
    } finally {
      reads.forEach(read => read.mockRestore());
      positions.destroy();
    }
  }, 30_000);
}

test('resident projected circular arc matches Deck at nonzero pitch', async () => {
  const device = await requireDevice();
  const source = makePositions(device, [-0.003, 45]);
  const target = makePositions(device, [0.003, 45]);
  const props = {
    getHeight: 0.8,
    getWidth: 6,
    numSegments: 50,
    getSourceColor: [255, 0, 0, 255] as [number, number, number, number],
    getTargetColor: [255, 0, 0, 255] as [number, number, number, number]
  };
  try {
    const nativePixels = await render(
      device,
      new ArcLayer({
        id: 'native-arc',
        data: [{}],
        getSourcePosition: (): [number, number] => [-0.003, 45],
        getTargetPosition: (): [number, number] => [0.003, 45],
        ...props
      })
    );
    const residentPixels = await render(
      device,
      new GPUArcLayer({
        id: 'resident-arc',
        getSourcePosition: source,
        getTargetPosition: target,
        arcMode: 'deck',
        ...props
      })
    );
    expectOverlap(nativePixels, residentPixels, 0.85);
  } finally {
    source.destroy();
    target.destroy();
  }
}, 30_000);

test('resident ground-plane meter path width and scale match Deck on a rectangular pitched viewport', async () => {
  const device = await requireDevice();
  const coordinates = [-0.003, 44.999, 0.003, 45.001];
  const paths = new GPUVector<VertexList<'float32x2'>>({
    type: 'data',
    name: 'path',
    format: 'vertex-list<float32x2>',
    data: [
      new GPUData({
        format: 'vertex-list<float32x2>',
        length: 1,
        valueLength: 2,
        valueOffsets: new Int32Array([0, 2]),
        ownsBuffer: true,
        buffer: device.createBuffer({
          data: Float32Array.from(coordinates),
          usage: Buffer.STORAGE | Buffer.VERTEX
        })
      })
    ]
  });
  const props = {
    getWidth: 20,
    widthScale: 2,
    widthUnits: 'meters' as const,
    billboard: false,
    getColor: [255, 0, 0, 255] as [number, number, number, number]
  };
  try {
    const nativePixels = await render(
      device,
      new PathLayer({
        id: 'native-path',
        data: [{}],
        getPath: (): [number, number][] => [
          [coordinates[0], coordinates[1]],
          [coordinates[2], coordinates[3]]
        ],
        ...props
      })
    );
    const residentPixels = await render(
      device,
      new GPUPathLayer({id: 'resident-path', getPath: paths, ...props})
    );
    expectOverlap(nativePixels, residentPixels, 0.85);
  } finally {
    paths.destroy();
  }
}, 30_000);

async function requireDevice(): Promise<Device> {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('Real WebGPU device required');
  return device;
}

function makePositions(device: Device, values: number[]): GPUVector<'float32x2'> {
  return new GPUVector({
    type: 'data',
    name: 'positions',
    format: 'float32x2',
    data: [
      new GPUData({
        format: 'float32x2',
        length: values.length / 2,
        ownsBuffer: true,
        buffer: device.createBuffer({
          data: Float32Array.from(values),
          usage: Buffer.VERTEX | Buffer.STORAGE
        })
      })
    ]
  });
}

function expectOverlap(reference: Uint8Array, actual: Uint8Array, minimum: number): void {
  let intersection = 0;
  let union = 0;
  let referenceCount = 0;
  for (let index = 0; index < reference.length; index += 4) {
    const referenceVisible = reference[index] > 100;
    const actualVisible = actual[index] > 100;
    if (referenceVisible) referenceCount++;
    if (referenceVisible && actualVisible) intersection++;
    if (referenceVisible || actualVisible) union++;
  }
  expect(referenceCount, 'native reference draws a substantial visible footprint').toBeGreaterThan(
    50
  );
  expect(
    intersection / union,
    'resident/native visible footprint intersection over union'
  ).toBeGreaterThan(minimum);
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
