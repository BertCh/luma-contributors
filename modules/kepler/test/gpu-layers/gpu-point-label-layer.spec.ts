// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Deck, OrbitView, MapView, type Layer} from '@deck.gl/core';
import {TextLayer} from '@deck.gl/layers';
import {GPUPointLabelLayer} from '../../../../examples/deck/kepler-compute/gpu-layers/index';
import {Buffer, Texture, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test, vi} from 'vitest';
import {preparePointLabelSource} from '../../../../examples/deck/kepler-compute/src/label-source';
import {makePointLabelRuntime} from '../../../../examples/deck/kepler-compute/src/label-runtime';

const SIZE = 256;
for (const geographic of [false, true]) {
  test(`resident ${geographic ? 'geographic' : 'cartesian'} labels retain native bitmap metrics, controls and source selection`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) throw new Error('WebGPU unavailable');
    const source = preparePointLabelSource({
      fieldName: 'name',
      chunks: [
        {id: 'first', values: ['A🙂', null]},
        {id: 'second', values: ['e\u0301B']}
      ],
      maxCharactersPerLabel: 32,
      maxGlyphs: 100,
      maxSourceBytes: 4096
    });
    const sourcePositions = geographic
      ? [
          [Math.fround(-0.001), 45, 0],
          [0, 45, 0],
          [Math.fround(0.001), Math.fround(45.0004), 30]
        ]
      : [
          [-36, 0, 0],
          [0, 20, 0],
          [30, -5, 5]
        ];
    const makeVector = <Format extends 'float32x3' | 'uint32' | 'float32'>(
      format: Format,
      values: number[][]
    ) =>
      new GPUVector({
        type: 'data',
        format,
        ownsData: true,
        data: [values.slice(0, 2), values.slice(2)].map(
          rows =>
            new GPUData({
              format,
              length: rows.length,
              ownsBuffer: true,
              buffer: device.createBuffer({
                data:
                  format === 'uint32'
                    ? new Uint32Array(rows.flat())
                    : new Float32Array(rows.flat()),
                usage: Buffer.STORAGE | Buffer.COPY_DST
              })
            })
        )
      });
    const positions = makeVector('float32x3', sourcePositions),
      mask = makeVector('uint32', [[1], [1], [1]]),
      radii = makeVector('float32', [[2], [4], [3]]);
    const reads = [...positions.data, ...mask.data, ...radii.data].map(chunk =>
      vi.spyOn(chunk.buffer, 'readAsync')
    );
    const runtime = makePointLabelRuntime(device, {
      source,
      positions,
      workingMemoryBudgetBytes: 4 << 20
    });
    await runtime.ready;
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
    const mapState = {longitude: 0, latitude: 45, zoom: 14};
    const coordinateSystem = geographic ? 'lnglat' : 'cartesian';
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
          ? {...mapState, pitch: 35, bearing: 15}
          : {target: [0, 0, 0], rotationX: 20, rotationOrbit: 15, zoom: 1},
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
                layer.isLoaded &&
                (layer instanceof TextLayer
                  ? layer.getSubLayers().length > 0
                  : layer.getModels().every(model => model.pipeline.linkStatus === 'success'))
            ),
          () => error
        );
        const previous = frames;
        deck!.redraw('label oracle');
        await waitUntil(
          () => frames > previous,
          () => error
        );
        return readPixels(device, texture);
      };
      for (const setting of [
        {size: 18, anchor: 'start', alignment: 'center'},
        {size: 24, anchor: 'middle', alignment: 'top'},
        {size: 15, anchor: 'end', alignment: 'bottom'}
      ] as const) {
        const result = runtime.resolve({
          label: {field: {name: 'name'}, ...setting, color: [220, 80, 40]},
          selectionMask: mask,
          radius: radii,
          radiusScale: 2,
          mapState
        });
        if (!('layerProps' in result)) throw new Error(result.diagnostic);
        const offsetScale = result.layerProps.radiusPixelScale!;
        const reference = new TextLayer({
          id: 'native-label',
          data: [
            {text: 'A🙂', position: sourcePositions[0], radius: 2},
            {text: '', position: sourcePositions[1], radius: 4},
            {text: 'e\u0301B', position: sourcePositions[2], radius: 3}
          ],
          getText: row => row.text,
          getPosition: row => row.position as [number, number, number],
          fontFamily: 'Monaco, monospace',
          fontWeight: 400,
          characterSet: [...source.characterSet],
          getColor: [220, 80, 40],
          getSize: 2 / 3,
          sizeScale: setting.size,
          getTextAnchor: setting.anchor,
          getAlignmentBaseline: setting.alignment,
          getPixelOffset: row => [
            (setting.anchor === 'middle' ? 0 : setting.anchor === 'start' ? 1 : -1) *
              (row.radius * offsetScale + 20),
            (setting.alignment === 'center' ? 0 : setting.alignment === 'bottom' ? 1 : -1) *
              (row.radius * offsetScale + 20 + setting.size)
          ],
          coordinateSystem,
          parameters: {depthCompare: 'always', depthWriteEnabled: false, cullMode: 'none'}
        });
        const expected = await render(reference);
        const layer = new GPUPointLabelLayer({
          id: 'resident-label',
          ...result.layerProps,
          coordinateSystem
        });
        const actual = await render(layer);
        let painted = 0,
          differing = 0,
          maximumError = 0,
          totalError = 0;
        for (let pixel = 0; pixel < SIZE * SIZE; pixel++) {
          if (expected[pixel * 4 + 3]) painted++;
          let different = false;
          for (let channel = 0; channel < 4; channel++) {
            const delta = Math.abs(expected[pixel * 4 + channel] - actual[pixel * 4 + channel]);
            maximumError = Math.max(maximumError, delta);
            totalError += delta;
            different ||= delta > 3;
          }
          if (different) differing++;
        }
        const metrics = {
          ...setting,
          painted,
          differing,
          maximumError,
          normalizedError: totalError / (SIZE * SIZE * 4 * 255),
          expectedBounds: measurePaint(expected),
          actualBounds: measurePaint(actual)
        };
        expect(painted, JSON.stringify(metrics)).toBeGreaterThan(50);
        expect(differing, JSON.stringify(metrics)).toBeLessThan(12);
        expect(maximumError, JSON.stringify(metrics)).toBeLessThanOrEqual(12);
        expect(metrics.normalizedError, JSON.stringify(metrics)).toBeLessThan(0.0001);
        console.info('point-label native raster metrics', JSON.stringify(metrics));
      }
      const resolved = runtime.resolve({
        label: {field: {name: 'name'}, anchor: 'middle', alignment: 'center', size: 28},
        selectionMask: mask,
        radiusScale: 1,
        mapState
      });
      if (!('layerProps' in resolved)) throw new Error(resolved.diagnostic);
      const layer = new GPUPointLabelLayer({
        id: 'label-selection',
        ...resolved.layerProps,
        coordinateSystem
      });
      const before = await render(layer);
      const models = layer.getModels();
      expect(layer.props.pickable).toBe(false);
      (mask.data[0].buffer as Buffer).write(new Uint32Array([0, 0]));
      (mask.data[1].buffer as Buffer).write(new Uint32Array([0]));
      const after = await render(layer);
      expect(before.some(value => value > 0)).toBe(true);
      expect(after.every(value => value === 0)).toBe(true);
      expect(layer.getModels()).toEqual(models);
      expect(layer.getPickingInfo({info: {index: 2} as never})).toHaveProperty('gpuVector', {
        rowIndex: 2,
        batchIndex: 1,
        batchRowIndex: 0
      });
      reads.forEach(read => expect(read).not.toHaveBeenCalled());
      deck.finalize();
      deck = undefined;
      await runtime.destroy(Promise.resolve());
      expect(positions.data.every(chunk => !chunk.buffer.destroyed)).toBe(true);
    } finally {
      deck?.finalize();
      restore();
      reads.forEach(read => read.mockRestore());
      await runtime.destroy(Promise.resolve());
      framebuffer.destroy();
      texture.destroy();
      positions.destroy();
      mask.destroy();
      radii.destroy();
      parent.remove();
    }
  }, 60_000);
}

function measurePaint(pixels: Uint8Array) {
  let minimumX = SIZE,
    minimumY = SIZE,
    maximumX = -1,
    maximumY = -1;
  let alphaSum = 0,
    weightedX = 0,
    weightedY = 0;
  for (let pixel = 0; pixel < SIZE * SIZE; pixel++) {
    const alpha = pixels[pixel * 4 + 3];
    if (!alpha) continue;
    const x = pixel % SIZE,
      y = Math.floor(pixel / SIZE);
    minimumX = Math.min(minimumX, x);
    minimumY = Math.min(minimumY, y);
    maximumX = Math.max(maximumX, x);
    maximumY = Math.max(maximumY, y);
    alphaSum += alpha;
    weightedX += x * alpha;
    weightedY += y * alpha;
  }
  return {
    bounds: [minimumX, minimumY, maximumX, maximumY],
    centroid: [weightedX / alphaSum, weightedY / alphaSum]
  };
}

async function waitUntil(predicate: () => boolean, getError: () => Error | null): Promise<void> {
  const deadline = performance.now() + 10000;
  while (!predicate() && performance.now() < deadline) {
    if (getError()) throw getError();
    await new Promise(resolve => requestAnimationFrame(resolve));
  }
  if (getError()) throw getError();
  expect(predicate(), 'Deck completed requested native-label frame').toBe(true);
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
