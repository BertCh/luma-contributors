// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import type {FontAtlas} from '@luma.gl/engine/text';
import {createNullWebGPUDevice} from '../../analysis/test/utils/gpu-contributor-test-utils';
import {
  computeLabelVisibilityWords,
  getAnchorValidityWords
} from '../../../examples/deck/kepler-compute/src/label-collision-runtime';
import {preparePointLabelSource} from '../../../examples/deck/kepler-compute/src/label-source';
import {
  getArcLabelAnchor,
  getLineLabelAnchor,
  getMultiPolygonLabelAnchor
} from '../../../examples/deck/kepler-compute/src/label-anchors';

vi.mock('@luma.gl/engine/text', async importOriginal => {
  const actual = await importOriginal<typeof import('@luma.gl/engine/text')>();
  return {
    ...actual,
    buildBitmapFontAtlas: vi.fn(() => makeAtlas()),
    GPUTextResources: class {
      readonly atlasTexture = {ready: Promise.resolve(), isReady: true};
      readonly fontAtlas: FontAtlas;
      constructor(
        readonly device: Device,
        props: {fontAtlas: FontAtlas}
      ) {
        this.fontAtlas = props.fontAtlas;
      }
      destroy = vi.fn();
    }
  };
});
// Node workers share a module graph; reload so label-runtime binds to the text mock above.
vi.resetModules();
const {makePointLabelRuntime} = await import(
  '../../../examples/deck/kepler-compute/src/label-runtime'
);

function makeAtlas(): FontAtlas {
  return {
    baselineOffset: 5,
    lineHeight: 64,
    width: 128,
    height: 64,
    pages: [{} as HTMLCanvasElement],
    xOffset: 0,
    yOffsetMin: 0,
    yOffsetMax: 0,
    renderSettings: {mode: 'bitmap', threshold: 0, smoothing: 0},
    mapping: {
      A: {
        x: 0,
        y: 0,
        width: 10,
        height: 12,
        advance: 10.5,
        anchorX: 5,
        anchorY: 9,
        layoutOffsetX: 0,
        layoutOffsetY: -9
      }
    }
  };
}

const viewport = {longitude: 0, latitude: 0, zoom: 4, width: 800, height: 600};
const style = {
  size: 64,
  fontSize: 64,
  anchor: 'middle' as const,
  alignment: 'center' as const,
  radius: 0,
  radiusPixelScale: 0,
  radiusPadding: 0,
  verticalPadding: 0,
  padding: 0
};

test('label visibility keeps the higher priority of two overlapping on-screen labels', () => {
  // 100x64 px labels at the same anchor overlap; the far one and the off-screen one differ.
  const words = computeLabelVisibilityWords({
    ...style,
    viewport,
    rowCount: 4,
    anchors: {positions: new Float32Array([0, 0, 0, 0, 20, 0, 0, 80])},
    rowBounds: new Float32Array([100, 64, 0, 0, 100, 64, 0, 0, 100, 64, 0, 0, 100, 64, 0, 0]),
    priorities: new Float32Array([1, 2, 0, 0])
  });
  // Row 1 outranks row 0 at the shared anchor; row 2 is far to the right; row 3 is off screen.
  expect(words[0]).toBe((1 << 1) | (1 << 2));
});

test('label visibility honors anchor shifts and hides invalid anchors', () => {
  const rowBounds = new Float32Array([50, 64, 0, 0, 50, 64, 0, 0]);
  const anchors = {positions: new Float32Array([0, 0, 0, 0]), validity: new Uint32Array([1, 0])};
  expect(
    computeLabelVisibilityWords({...style, viewport, rowCount: 2, anchors, rowBounds})[0]
  ).toBe(1);
  expect([
    ...getAnchorValidityWords(34, new Uint32Array([1, 0, 1, ...new Array(30).fill(0), 1]))
  ]).toEqual([5, 2]);
});

test('label anchors cover lines, arcs and multipolygons', () => {
  expect(
    getLineLabelAnchor([
      [0, 0],
      [2, 0]
    ])
  ).toEqual([1, 0]);
  expect(getArcLabelAnchor([0, 0], [0, 60])[1]).toBeCloseTo(30);
  const square = (x: number, size: number) => [
    [
      [x, 0],
      [x + size, 0],
      [x + size, size],
      [x, size],
      [x, 0]
    ]
  ];
  expect(getMultiPolygonLabelAnchor([square(0, 1), square(10, 4)])).toEqual([12, 2]);
});

test('point label runtime applies CPU collision only with anchors and a viewport size', async () => {
  const device = createNullWebGPUDevice();
  const source = preparePointLabelSource({
    fieldName: 'name',
    chunks: [{id: 'only', values: ['A', null, 'A']}],
    maxCharactersPerLabel: 10,
    maxGlyphs: 10,
    maxSourceBytes: 1024
  });
  const makeVector = <Format extends 'float32x2' | 'uint32'>(format: Format, width: number) =>
    new GPUVector({
      type: 'data',
      format,
      ownsData: true,
      data: [
        new GPUData({
          format,
          length: 3,
          ownsBuffer: true,
          buffer: device.createBuffer({
            byteLength: 3 * width,
            usage: Buffer.STORAGE | Buffer.COPY_DST
          })
        })
      ]
    });
  const positions = makeVector('float32x2', 8);
  const mask = makeVector('uint32', 4);
  const mapState = {longitude: 0, latitude: 0, zoom: 4, width: 800, height: 600};
  const makeRuntime = (anchors?: {positions: Float32Array; validity?: Uint32Array}) =>
    makePointLabelRuntime(device, {
      source,
      positions,
      workingMemoryBudgetBytes: 1 << 20,
      anchors
    });
  const collided = makeRuntime({
    positions: new Float32Array([0, 0, 0, 0, 0, 0]),
    validity: new Uint32Array([1, 1, 1])
  });
  await collided.ready;
  const resolve = (
    runtime: ReturnType<typeof makeRuntime>,
    collisionEnabled: boolean,
    state = mapState
  ) =>
    runtime.resolve({
      label: {field: {name: 'name'}, collisionEnabled},
      selectionMask: mask,
      radiusScale: 0,
      mapState: state
    });
  const withCollision = resolve(collided, true);
  if (!('layerProps' in withCollision)) throw new Error('Expected a collision plan');
  // Rows 0 and 2 share one anchor: row 0 wins; the null label has no bounds and stays hidden.
  expect([...withCollision.layerProps.visibilityWords!]).toEqual([1]);
  // Unchanged viewport reuses the cached words.
  const again = resolve(collided, true);
  if (!('layerProps' in again)) throw new Error('Expected a collision plan');
  expect(again.layerProps.visibilityWords).toBe(withCollision.layerProps.visibilityWords);
  // Without collision the validity words alone gate rows; without a viewport size nothing is filtered.
  const plain = resolve(collided, false);
  if (!('layerProps' in plain)) throw new Error('Expected a plain plan');
  expect([...plain.layerProps.visibilityWords!]).toEqual([7]);
  const unsized = resolve(collided, true, {longitude: 0, latitude: 0, zoom: 4});
  if (!('layerProps' in unsized)) throw new Error('Expected an unsized plan');
  expect([...unsized.layerProps.visibilityWords!]).toEqual([7]);
  // Without anchors collision stays an explicit diagnostic and plain labels carry no mask.
  const bare = makeRuntime();
  await bare.ready;
  expect(resolve(bare, true)).toHaveProperty('diagnostic');
  const bareLayer = resolve(bare, false);
  if (!('layerProps' in bareLayer)) throw new Error('Expected a bare plan');
  expect(bareLayer.layerProps.visibilityWords).toBeUndefined();
  for (const runtime of [collided, bare]) await runtime.destroy(Promise.resolve());
  positions.destroy();
  mask.destroy();
  device.destroy();
});
