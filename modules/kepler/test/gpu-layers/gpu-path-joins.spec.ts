// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Deck, OrbitView, MapView, type Layer} from '@deck.gl/core';
import {PathLayer} from '@deck.gl/layers';
// Import the pinned oracle directly; the geo-layers barrel also loads unrelated legacy WebGL layers.
import TripsLayer from '../../../../node_modules/@deck.gl/geo-layers/src/trips-layer/trips-layer';
import {GPUPathLayer} from '../../../../examples/deck/kepler-compute/gpu-layers/index';
import {Buffer, Texture, type Device} from '@luma.gl/core';
import {GPUData, GPUVector, type VertexList} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test, vi} from 'vitest';

const SIZE = 128;
const PATHS = [
  [
    [-23, -15, 0],
    [-23, 10, 4],
    [-7, -10, 8],
    [-7, -10, 8],
    [-5, 18, 2]
  ],
  [
    [8, -14, 0],
    [25, -14, 2],
    [21, 15, 8],
    [8, -14, 0]
  ]
];

for (const {format, geographic} of [
  {format: 'float32x2', geographic: false},
  {format: 'float32x3', geographic: false},
  {format: 'float32x4', geographic: false},
  {format: 'float32x3', geographic: true}
] as const) {
  test(`resident ${geographic ? 'geographic' : 'cartesian'} ${format} paths match native joins, caps, widths and Trips without source readback`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) throw new Error('WebGPU unavailable');
    const sourcePaths = geographic
      ? PATHS.map(path =>
          path.map(([longitude, latitude, altitude]) => [
            Math.fround(longitude * 0.0001),
            Math.fround(45 + latitude * 0.0001),
            altitude * 20
          ])
        )
      : PATHS;
    const coordinateSystem = geographic ? 'lnglat' : 'cartesian';
    const points = sourcePaths.map(path =>
      path.map((point, index) =>
        format === 'float32x2'
          ? point.slice(0, 2)
          : format === 'float32x4'
            ? [...point, index * 10]
            : point
      )
    );
    const paths = makeList(device, format, points);
    const timestamps = makeList(
      device,
      'float32',
      PATHS.map(path => path.map((_point, index) => [index * 10]))
    );
    const masks = new GPUVector({
      type: 'data',
      format: 'uint32',
      ownsData: false,
      data: PATHS.map(
        () =>
          new GPUData({
            format: 'uint32',
            length: 1,
            ownsBuffer: false,
            buffer: device.createBuffer({
              data: new Uint32Array([1]),
              usage: Buffer.STORAGE | Buffer.COPY_DST
            })
          })
      )
    });
    const buffers = [...paths.data, ...timestamps.data, ...masks.data].map(
      chunk => chunk.buffer as Buffer
    );
    const reads = buffers.map(buffer => vi.spyOn(buffer, 'readAsync'));
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
    let deck: Deck<OrbitView | MapView> | undefined;
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
        views: geographic ? new MapView({id: 'main'}) : new OrbitView({id: 'main', orbitAxis: 'Z'}),
        initialViewState: geographic
          ? {longitude: 0, latitude: 45, zoom: 14, pitch: 45, bearing: 20}
          : {target: [0, 0, 3], rotationX: 30, rotationOrbit: 20, zoom: 1},
        layers: [],
        onError: next => {
          error = next;
        },
        onAfterRender: () => {
          frames++;
        }
      });
      const render = async (layer: Layer) => {
        deck!.setProps({layers: [layer]});
        await waitUntil(
          () =>
            Boolean(
              deck?.isInitialized &&
                layer.state &&
                layer.getModels().every(model => model.pipeline.linkStatus === 'success')
            ),
          () => error
        );
        const previous = frames;
        deck!.redraw('path oracle');
        await waitUntil(
          () => frames > previous,
          () => error
        );
        return readPixels(device, texture);
      };
      const metrics: ReturnType<typeof comparePixels>[] = [];
      for (const settings of [
        {
          billboard: true,
          jointRounded: false,
          capRounded: false,
          miterLimit: 4,
          antialiasing: false,
          widthUnits: 'pixels' as const
        },
        {
          billboard: false,
          jointRounded: false,
          capRounded: false,
          miterLimit: 2,
          antialiasing: false,
          widthUnits: 'meters' as const
        },
        {
          billboard: true,
          jointRounded: false,
          capRounded: false,
          miterLimit: 1,
          antialiasing: false,
          widthUnits: 'pixels' as const
        },
        {
          billboard: true,
          jointRounded: true,
          capRounded: true,
          miterLimit: 2,
          antialiasing: false,
          widthUnits: 'pixels' as const
        },
        {
          billboard: false,
          jointRounded: true,
          capRounded: true,
          miterLimit: 2,
          antialiasing: false,
          widthUnits: 'meters' as const
        },
        {
          billboard: false,
          jointRounded: false,
          capRounded: false,
          miterLimit: 4,
          antialiasing: false,
          widthUnits: 'common' as const
        },
        {
          billboard: true,
          jointRounded: true,
          capRounded: true,
          miterLimit: 2,
          antialiasing: true,
          widthUnits: 'pixels' as const
        },
        {
          billboard: false,
          jointRounded: true,
          capRounded: true,
          miterLimit: 2,
          antialiasing: true,
          widthUnits: 'pixels' as const
        }
      ]) {
        const common = {
          ...settings,
          coordinateSystem,
          getWidth: 5,
          widthScale: 1.5,
          widthMinPixels: 2,
          widthMaxPixels: 12,
          getColor: [200, 70, 30, 180] as [number, number, number, number],
          opacity: 0.8,
          pickable: true
        };
        const data = points.map(path => path.map(point => [point[0], point[1], point[2] ?? 0]));
        const expected = await render(
          new PathLayer({id: 'native-path', ...common, data, getPath: path => path})
        );
        const resident = new GPUPathLayer({
          id: 'resident-path',
          ...common,
          getPath: paths,
          selectionMask: masks
        });
        const actual = await render(resident);
        metrics.push(
          comparePixels(actual, expected, JSON.stringify(settings), settings.antialiasing)
        );
      }
      for (const fadeTrail of [true, false]) {
        const tripCommon = {
          coordinateSystem,
          billboard: true,
          jointRounded: true,
          capRounded: true,
          antialiasing: true,
          widthUnits: 'pixels' as const,
          getWidth: 8,
          getColor: [50, 160, 220, 200] as [number, number, number, number],
          currentTime: 25,
          trailLength: 20,
          fadeTrail
        };
        const tripData = points.map(path => path.map(point => [point[0], point[1], point[2] ?? 0]));
        const expectedTrip = await render(
          new TripsLayer({
            id: 'native-trip',
            ...tripCommon,
            data: tripData,
            getPath: path => path,
            getTimestamps: path => path.map((_point, index) => index * 10)
          })
        );
        const actualTrip = await render(
          new GPUPathLayer({
            id: 'resident-trip',
            timestampMode: 'trips',
            ...tripCommon,
            getPath: paths,
            getTimestamps: timestamps
          })
        );
        metrics.push(comparePixels(actualTrip, expectedTrip, `Trips fade=${fadeTrail}`, true));
      }
      console.info(
        'GPU path native oracle',
        JSON.stringify({
          format,
          geographic,
          nonAntialiasedMaximumChannelError: Math.max(
            ...metrics
              .filter(metric => !metric.antialiasing)
              .map(metric => metric.maximumChannelError)
          ),
          maximumChannelError: Math.max(...metrics.map(metric => metric.maximumChannelError)),
          maximumSilhouetteDifferences: Math.max(
            ...metrics.map(metric => metric.silhouetteDifferences)
          ),
          maximumNormalizedTotalError: Math.max(
            ...metrics.map(metric => metric.normalizedTotalError)
          ),
          maximumNormalizedPaintedError: Math.max(
            ...metrics.map(metric => metric.normalizedPaintedError)
          )
        })
      );
      const resident = new GPUPathLayer({
        id: 'resident-picking',
        coordinateSystem,
        getPath: paths,
        selectionMask: masks,
        getWidth: 10,
        capRounded: true,
        jointRounded: true,
        pickable: true
      });
      await render(resident);
      const models = resident.getModels();
      const screen = deck
        .getViewports()[0]
        .project(
          geographic
            ? [Math.fround(16 * 0.0001), Math.fround(45 - 14 * 0.0001), 20]
            : [16, -14, format === 'float32x2' ? 0 : 1]
        );
      expect(await deck.pickObjectAsync({x: screen[0], y: screen[1], radius: 1})).toHaveProperty(
        'gpuVector',
        {rowIndex: 1, batchIndex: 1, batchRowIndex: 0}
      );
      (masks.data[1].buffer as Buffer).write(new Uint32Array([0]));
      await render(resident);
      expect(await deck.pickObjectAsync({x: screen[0], y: screen[1], radius: 1})).toBeNull();
      expect(resident.getModels()).toEqual(models);
      for (const read of reads) expect(read).not.toHaveBeenCalled();
      deck.finalize();
      deck = undefined;
      for (const buffer of buffers) expect(buffer.destroyed).toBe(false);
    } finally {
      deck?.finalize();
      restore();
      reads.forEach(read => read.mockRestore());
      framebuffer.destroy();
      texture.destroy();
      buffers.forEach(buffer => buffer.destroy());
      parent.remove();
    }
  }, 60_000);
}

function comparePixels(
  actual: Uint8Array,
  expected: Uint8Array,
  label: string,
  antialiasing = false
) {
  let differences = 0;
  let expectedPixels = 0;
  let silhouetteDifferences = 0;
  let paintedUnionPixels = 0;
  let maximumChannelError = 0;
  let totalChannelError = 0;
  for (let pixel = 0; pixel < SIZE * SIZE; pixel++) {
    if (expected[pixel * 4 + 3]) expectedPixels++;
    if (expected[pixel * 4 + 3] || actual[pixel * 4 + 3]) paintedUnionPixels++;
    if (Boolean(actual[pixel * 4 + 3]) !== Boolean(expected[pixel * 4 + 3]))
      silhouetteDifferences++;
    let differs = false;
    for (let channel = 0; channel < 4; channel++) {
      const error = Math.abs(actual[pixel * 4 + channel] - expected[pixel * 4 + channel]);
      maximumChannelError = Math.max(maximumChannelError, error);
      totalChannelError += error;
      differs ||= error > 3;
    }
    if (differs) differences++;
  }
  const normalizedTotalError = totalChannelError / (SIZE * SIZE * 4 * 255);
  const normalizedPaintedError = totalChannelError / (Math.max(paintedUnionPixels, 1) * 4 * 255);
  expect(expectedPixels, label).toBeGreaterThan(100);
  if (antialiasing) {
    // Native attribute interpolation and resident storage interpolation can yield different
    // derivatives at a few rounded edge pixels. Bound local error AND total image error,
    // retain a strict silhouette check, and require bit-exact geometry without analytic AA.
    const diagnostic = `${label}: ${JSON.stringify({differences, expectedPixels, silhouetteDifferences, maximumChannelError, normalizedTotalError, normalizedPaintedError})}`;
    expect(silhouetteDifferences, diagnostic).toBeLessThanOrEqual(2);
    expect(differences, diagnostic).toBeLessThanOrEqual(Math.ceil(expectedPixels * 0.02));
    expect(maximumChannelError, diagnostic).toBeLessThanOrEqual(64);
    expect(normalizedTotalError, diagnostic).toBeLessThan(0.0001);
    // A painted-union bound prevents empty framebuffer background from diluting the error.
    expect(normalizedPaintedError, diagnostic).toBeLessThan(0.0005);
  } else {
    expect(maximumChannelError, label).toBe(0);
    expect(silhouetteDifferences, label).toBe(0);
  }
  return {
    antialiasing,
    maximumChannelError,
    silhouetteDifferences,
    normalizedTotalError,
    normalizedPaintedError
  };
}
function makeList<Format extends 'float32' | 'float32x2' | 'float32x3' | 'float32x4'>(
  device: Device,
  format: Format,
  rows: number[][][]
): GPUVector<VertexList<Format>> {
  return new GPUVector({
    type: 'data',
    format: `vertex-list<${format}>`,
    ownsData: false,
    data: rows.map(
      row =>
        new GPUData({
          format: `vertex-list<${format}>`,
          length: 1,
          valueLength: row.length,
          valueOffsets: new Int32Array([0, row.length]),
          ownsBuffer: false,
          buffer: device.createBuffer({
            data: new Float32Array(row.flat()),
            usage: Buffer.STORAGE | Buffer.COPY_DST
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
  expect(predicate(), 'Deck completed requested native-path frame').toBe(true);
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
