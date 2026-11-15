// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Deck, _GlobeView as GlobeView, type Layer} from '@deck.gl/core';
import {PathLayer, ScatterplotLayer, SolidPolygonLayer} from '@deck.gl/layers';
import {
  GPUGlobeDepthDiskLayer,
  GPUMeshIconLayer,
  GPUPathLayer,
  GPUScatterplotLayer,
  GPUSolidPolygonLayer
} from '../../../../examples/deck/kepler-compute/gpu-layers/index';
import {Buffer, Texture, type Device} from '@luma.gl/core';
import {GPUData, GPUVector, type VertexList} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';

const SIZE = 192;
const NEAR = [0, 0];
const FAR = [180, 0];
const BACK_FACING = [165, 0];

type LngLat = number[];

/** Native GlobeView: resident layers draw the near hemisphere, clip at the horizon and track the native oracle. */
test('resident GPU layers render on GlobeView, clip at the horizon and track native Deck', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU unavailable');
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
  const buffers: Buffer[] = [];
  let deck: Deck<GlobeView> | undefined;
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
      // Deck 10 counts earth intervals; 72 preserves deck 9's five-degree tessellation.
      views: new GlobeView({id: 'main', resolution: 72}),
      initialViewState: {longitude: NEAR[0], latitude: NEAR[1], zoom: 0},
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
      deck!.redraw('globe');
      await waitUntil(
        () => frames > previous,
        () => error
      );
      return readPixels(device, texture);
    };
    const own = <T extends Buffer>(buffer: T): T => {
      buffers.push(buffer);
      return buffer;
    };
    const list = <Format extends 'uint32' | 'float32x3' | 'float32x4'>(
      rows: number[][][],
      format: Format
    ) =>
      new GPUVector({
        name: 'globe-oracle',
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
              buffer: own(
                device.createBuffer({
                  data:
                    format === 'uint32'
                      ? new Uint32Array(row.flat())
                      : new Float32Array(row.flat()),
                  usage:
                    Buffer.STORAGE |
                    Buffer.VERTEX |
                    Buffer.COPY_DST |
                    (format === 'uint32' ? Buffer.INDEX : 0)
                })
              )
            })
        )
      }) as unknown as GPUVector<VertexList<Format>>;
    const flat = <Format extends 'float32x2' | 'uint32'>(rows: number[][], format: Format) =>
      new GPUVector({
        name: 'globe-oracle',
        type: 'data',
        format,
        ownsData: false,
        data: rows.map(
          row =>
            new GPUData({
              format,
              length: row.length / (format === 'float32x2' ? 2 : 1),
              ownsBuffer: false,
              buffer: own(
                device.createBuffer({
                  data: format === 'uint32' ? new Uint32Array(row) : new Float32Array(row),
                  usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_DST
                })
              )
            })
        )
      }) as GPUVector<typeof format>;
    const at = (center: LngLat, offsets: LngLat[]) =>
      offsets.map(([east, north]) => [center[0]! + east!, center[1]! + north!]);
    const pathFor = (center: LngLat) =>
      at(center, [
        [-12, -6],
        [0, 8],
        [12, -6]
      ]);
    const triangleFor = (center: LngLat) =>
      at(center, [
        [-12, -8],
        [12, -8],
        [0, 12]
      ]);
    const view = (center: LngLat) =>
      deck!.setProps({
        initialViewState: {longitude: center[0], latitude: center[1], zoom: 0},
        viewState: {longitude: center[0], latitude: center[1], zoom: 0}
      } as never);
    const painted = (pixels: Uint8Array) => {
      let count = 0;
      for (let pixel = 0; pixel < SIZE * SIZE; pixel++) if (pixels[pixel * 4 + 3]) count++;
      return count;
    };
    const color = [220, 60, 40, 255] as [number, number, number, number];
    const cases = [
      {
        name: 'path',
        // Deck 10.0.0-alpha.1's WGSL path shader multiplies deltas by the globe tangent basis instead
        // of its transpose, so a non-billboard native path is far thinner than the resident one (38
        // against 254 painted pixels in this frame; billboard paths agree, 384 against 386). The
        // resident layer follows Deck's GLSL (transpose), so the native silhouette is no oracle here.
        nativeSilhouetteIsOracle: false,
        native: (center: LngLat) =>
          new PathLayer({
            id: 'native',
            data: [pathFor(center)],
            getPath: path => path,
            getWidth: 5,
            widthUnits: 'pixels',
            billboard: false,
            getColor: color
          }),
        resident: (center: LngLat) =>
          new GPUPathLayer({
            id: 'resident',
            getPath: list([pathFor(center).map(([lng, lat]) => [lng!, lat!, 0])], 'float32x3'),
            getWidth: 5,
            widthUnits: 'pixels',
            billboard: false,
            getColor: color
          })
      },
      {
        name: 'polygon',
        native: (center: LngLat) =>
          new SolidPolygonLayer({
            id: 'native',
            data: [triangleFor(center)],
            getPolygon: polygon => polygon,
            getFillColor: color,
            material: false
          }),
        resident: (center: LngLat) =>
          new GPUSolidPolygonLayer({
            id: 'resident',
            positions: list(
              [triangleFor(center).map(([lng, lat]) => [lng!, lat!, 0, 1])],
              'float32x4'
            ),
            rowIndices: list([[[0], [0], [0]]], 'uint32'),
            indices: list([[[0], [1], [2]]], 'uint32'),
            getFillColor: color,
            material: false
          })
      },
      {
        // Resident aggregate cells are preprojected Web Mercator common space (Deck CARTESIAN).
        name: 'polygon-mercator-common',
        native: (center: LngLat) =>
          new SolidPolygonLayer({
            id: 'native',
            data: [triangleFor(center)],
            getPolygon: polygon => polygon,
            getFillColor: color,
            material: false
          }),
        resident: (center: LngLat) =>
          new GPUSolidPolygonLayer({
            id: 'resident',
            coordinateSystem: 'cartesian',
            positions: list(
              [
                triangleFor(center).map(([lng, lat]) => {
                  const scale = 512 / (2 * Math.PI);
                  const y = Math.PI + Math.log(Math.tan(Math.PI / 4 + (lat! * Math.PI) / 360));
                  return [((lng! + 180) / 360) * 512, y * scale, 0, 1];
                })
              ],
              'float32x4'
            ),
            rowIndices: list([[[0], [0], [0]]], 'uint32'),
            indices: list([[[0], [1], [2]]], 'uint32'),
            getFillColor: color,
            material: false
          })
      },
      ...[true, false].map(billboard => ({
        name: billboard ? 'point-billboard' : 'point-ground',
        native: (center: LngLat) =>
          new ScatterplotLayer({
            id: 'native',
            data: [center],
            getPosition: point => point as [number, number],
            getFillColor: color,
            billboard,
            radiusUnits: billboard ? 'pixels' : 'meters',
            getRadius: billboard ? 10 : 600_000
          }),
        resident: (center: LngLat) =>
          new GPUScatterplotLayer({
            id: 'resident',
            getPosition: flat([center], 'float32x2'),
            getFillColor: color,
            billboard,
            radiusUnits: billboard ? 'pixels' : 'meters',
            getRadius: billboard ? 10 : 600_000
          })
      }))
    ];
    const results: Record<string, unknown> = {};
    for (const entry of cases) {
      view(NEAR);
      const nativeNear = painted(await render(entry.native(NEAR)));
      expect(nativeNear, `${entry.name}: native near oracle must draw`).toBeGreaterThan(20);
      const residentNear = painted(await render(entry.resident(NEAR)));
      // The oracle and the resident layer agree on the near-side silhouette to within 5%.
      if ('nativeSilhouetteIsOracle' in entry && entry.nativeSilhouetteIsOracle === false) {
        expect(residentNear, `${entry.name}: resident near silhouette must draw`).toBeGreaterThan(
          100
        );
      } else {
        expect(
          Math.abs(residentNear - nativeNear),
          `${entry.name} near ${residentNear} vs ${nativeNear}`
        ).toBeLessThanOrEqual(Math.ceil(nativeNear * 0.05));
      }
      const residentFar = painted(await render(entry.resident(FAR)));
      expect(residentFar, `${entry.name}: antipode must be culled`).toBe(0);
      // A feature just past the horizon (165 degrees of longitude) must not leak onto the near side.
      const residentBack = painted(await render(entry.resident(BACK_FACING)));
      expect(residentBack, `${entry.name}: back hemisphere must be culled`).toBe(0);
      view(FAR);
      const residentRotated = painted(await render(entry.resident(FAR)));
      expect(residentRotated, `${entry.name}: rotating the globe reveals it`).toBeGreaterThan(20);
      results[entry.name] = {nativeNear, residentNear, residentFar, residentBack, residentRotated};
    }
    // Mesh icons have no native oracle (a diamond is not a circle); require the same near/horizon behavior.
    const diamond = own(
      device.createBuffer({
        data: new Float32Array([0, -1, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 1, 0, -1, 0, 0]),
        usage: Buffer.VERTEX | Buffer.COPY_DST
      })
    );
    for (const billboard of [true, false]) {
      const name = billboard ? 'icon-billboard' : 'icon-ground';
      const icon = (center: LngLat) =>
        new GPUMeshIconLayer({
          id: 'resident',
          getPosition: flat([center], 'float32x2'),
          getFillColor: color,
          iconGeometry: new GPUData({
            format: 'float32x3',
            length: 6,
            buffer: diamond,
            ownsBuffer: false
          }),
          billboard,
          radiusUnits: 'meters',
          getRadius: 600_000
        });
      view(NEAR);
      const residentNear = painted(await render(icon(NEAR)));
      expect(residentNear, `${name}: mesh icon must draw on the globe`).toBeGreaterThan(20);
      const residentFar = painted(await render(icon(FAR)));
      expect(residentFar, `${name}: antipode must be culled`).toBe(0);
      view(FAR);
      const residentRotated = painted(await render(icon(FAR)));
      expect(residentRotated, `${name}: rotating the globe reveals it`).toBeGreaterThan(20);
      results[name] = {residentNear, residentFar, residentRotated};
    }
    // The depth disk hides geometry that does not horizon-cull itself, like native Deck layers and basemap tiles.
    const farNative = (center: LngLat) =>
      new ScatterplotLayer({
        id: 'native-far',
        data: [center],
        getPosition: point => point as [number, number],
        getFillColor: color,
        billboard: true,
        radiusUnits: 'pixels',
        getRadius: 12,
        parameters: {depthCompare: 'less-equal', depthWriteEnabled: true}
      });
    const renderLayers = async (layers: Layer[]) => {
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
      deck!.redraw('globe');
      await waitUntil(
        () => frames > previous,
        () => error
      );
      return painted(await readPixels(device, texture));
    };
    view(NEAR);
    const withoutDisk = await renderLayers([farNative(FAR)]);
    const withDisk = await renderLayers([
      new GPUGlobeDepthDiskLayer({id: 'disk', fillColor: [0, 0, 0, 0]}),
      farNative(FAR)
    ]);
    // Deck 10 culls far-side geometry itself (project_globe_is_occluded), in scatterplot, path and
    // solid polygon alike (probed), so the disk can no longer be shown to hide native geometry.
    // What is left to check is that the disk is a real occluder: drawn opaque it fills a large
    // part of the frame, and with a transparent fill it still adds no visible pixels.
    expect(withoutDisk, 'native far-side geometry stays culled by Deck 10').toBe(0);
    expect(withDisk, 'a transparent depth disk paints nothing').toBe(0);
    const opaqueDisk = await renderLayers([
      new GPUGlobeDepthDiskLayer({id: 'disk-opaque', fillColor: [255, 0, 255, 255]})
    ]);
    results['depth-disk'] = {withoutDisk, withDisk, opaqueDisk};
    expect(opaqueDisk, 'the depth disk covers the globe silhouette').toBeGreaterThan(
      SIZE * SIZE * 0.1
    );
    console.info('GPU globe native oracle', JSON.stringify(results));
    deck.finalize();
    deck = undefined;
  } finally {
    deck?.finalize();
    restore();
    framebuffer.destroy();
    texture.destroy();
    buffers.forEach(buffer => buffer.destroy());
    parent.remove();
  }
}, 120_000);

async function waitUntil(predicate: () => boolean, getError: () => Error | null): Promise<void> {
  const deadline = performance.now() + 10_000;
  while (!predicate() && performance.now() < deadline) {
    if (getError()) throw getError();
    await new Promise(resolve => requestAnimationFrame(resolve));
  }
  if (getError()) throw getError();
  expect(predicate(), 'Deck completed the requested globe frame').toBe(true);
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
