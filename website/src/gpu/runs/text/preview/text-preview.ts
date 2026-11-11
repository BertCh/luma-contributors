// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Development page for picture text (not part of the site build, not in the run registry). It
// draws one of four labelled pictures with the run layers and the text layer, on the Chicago tract
// extract, in a theme and at a size given by the query:
//
//   text-preview.html?case=axes|bars|legend|matrix&theme=dark|light&size=card|hero|poster
//
// `card` and `hero` are the measured CSS sizes of a catalog card and the function-page hero at a
// 1440 px wide window; `poster` is the harness capture (960 px wide, device pixel ratio 1) with
// labels sized for display on a card, as snapshot-runs.mjs would capture it after the wiring.
// ./capture-text-preview.mjs photographs every combination.

import type {Buffer, Device} from '@luma.gl/core';
import {createGpuDevice} from '../../../device';
import {
  createStorageBuffer,
  type DemoLayer,
  type DemoTheme,
  type DemoView,
  drawLayers,
  getDemoTheme,
  makeRandom,
  SegmentListLayer
} from '../../../demos/kit';
import {createMatrixMesh} from '../../entries/density-interpolation-regions/support/dynamics';
import {createTracts} from '../../inputs';
import {ChartLayer, MeshLayer} from '../../layers';
import {
  CARD_DISPLAY_WIDTH,
  createBarValueLabels,
  createChartAxes,
  createLegend,
  createMatrixTicks,
  formatTick,
  getChartRectangle,
  loadFontPixels,
  mergeText,
  type RunLabel,
  type RunText,
  TextLayer
} from '../index';

type PreviewCase = 'axes' | 'bars' | 'legend' | 'matrix';
type PreviewSize = 'card' | 'card1' | 'hero' | 'poster';

/** CSS sizes measured on the running site at a 1440 x 900 window (catalog card, fn hero). */
const SIZES: Record<PreviewSize, {width: number; height: number; displayWidth?: number}> = {
  card: {width: 265, height: 166},
  // The same card on a display with a device pixel ratio of 1 (the capture sets the ratio).
  card1: {width: 265, height: 166},
  hero: {width: 887, height: 499},
  // The poster is captured large, at the card's aspect ratio here so it can be compared 1:1.
  poster: {width: 960, height: 600, displayWidth: CARD_DISPLAY_WIDTH}
};

const CLASS_NAMES = ['under 15%', '15–30%', '30–45%', '45% and over'];
const CLASS_LIMITS = [0.15, 0.3, 0.45];

function getClass(share: number): number {
  return CLASS_LIMITS.filter(limit => share >= limit).length;
}

type Scene = {layers: DemoLayer[]; text: RunText; under?: RunText; owned: Buffer[]};

function buildScene(
  device: Device,
  previewCase: PreviewCase,
  theme: DemoTheme,
  view: DemoView
): Scene {
  const world = {width: view.worldWidth, height: view.worldHeight};
  const tracts = createTracts(makeRandom(1), world, 14);
  const owned: Buffer[] = [];
  const buffer = (values: Float32Array | Uint32Array) => {
    const created = createStorageBuffer(device, values);
    owned.push(created);
    return created;
  };
  const percent = Array.from(tracts.povertyShare, share => share * 100);
  const ink = [theme.ink[0], theme.ink[1], theme.ink[2], 0.85] as const;
  const muted = [theme.muted[0], theme.muted[1], theme.muted[2], 0.55] as const;

  if (previewCase === 'axes') {
    // Poverty rate of every tract, ranked: a line chart with value axes and a median reference.
    const sorted = Float32Array.from([...percent].sort((first, second) => first - second));
    const rectangle = getChartRectangle(world);
    const line = new ChartLayer(
      device,
      'line',
      buffer(sorted),
      buffer(new Float32Array(1)),
      sorted.length,
      'float32'
    );
    line.setPlot(rectangle, [0, 1]);
    const maximum = 80;
    line.update(view, theme, {mode: -1, range: [0, maximum], radius: 1.6, solid: ink});
    const median = sorted[Math.floor(sorted.length / 2)];
    const axes = createChartAxes({
      rectangle,
      x: {range: [0, sorted.length], title: 'tracts, ranked'},
      y: {range: [0, maximum], title: 'poverty (%)', ticks: [0, 20, 40, 60, 80]},
      references: [{value: median, label: `median ${formatTick(median, 0.1)}%`}]
    });
    return {layers: [line], owned, under: axes.grid, text: axes};
  }

  if (previewCase === 'bars') {
    // Histogram of tract poverty rates in 10-point bins, with counts over the bars.
    const binCount = 8;
    const counts = new Float32Array(binCount);
    for (const value of percent) counts[Math.min(binCount - 1, Math.floor(value / 10))]++;
    const rectangle = getChartRectangle(world);
    const maximum = Math.ceil(Math.max(...counts) / 10) * 10;
    const bars = new ChartLayer(
      device,
      'bars',
      buffer(counts),
      buffer(new Float32Array(1)),
      binCount,
      'float32'
    );
    bars.setPlot(rectangle, [0, 1]);
    bars.update(view, theme, {mode: 0, range: [0, maximum], radius: 1.6});
    const categories = Array.from({length: binCount}, (_, bin) =>
      bin === binCount - 1 ? `${bin * 10}+` : `${bin * 10}–${bin * 10 + 10}`
    );
    const axes = createChartAxes({
      rectangle,
      x: {categories, title: '% in poverty'},
      y: {range: [0, maximum], title: 'tracts', ticks: 3}
    });
    return {
      layers: [bars],
      owned,
      under: axes.grid,
      text: mergeText(axes, createBarValueLabels({rectangle, values: counts, range: [0, maximum]}))
    };
  }

  if (previewCase === 'legend') {
    // Tracts by poverty class with a four-class legend.
    const classes = Uint32Array.from(tracts.povertyShare, share => getClass(share));
    const mesh = new MeshLayer(
      device,
      buffer(tracts.vertices),
      buffer(tracts.triangleIndices),
      buffer(tracts.triangleOwners),
      buffer(classes),
      tracts.triangleCount,
      'uint32'
    );
    mesh.update(view, theme, {mode: 2, solid: muted});
    const outline = new SegmentListLayer(
      device,
      buffer(tracts.outline),
      buffer(Uint32Array.of(tracts.outlineCount)),
      tracts.outlineCount
    );
    outline.update(view, theme, {
      mode: -1,
      radius: 0.5,
      solid: [theme.background[0], theme.background[1], theme.background[2], 0.6]
    });
    return {
      layers: [mesh, outline],
      owned,
      text: createLegend({
        title: 'below poverty line',
        entries: CLASS_NAMES.map((label, index) => ({label, swatch: index}))
      })
    };
  }

  // Join counts between the poverty classes of contiguous tracts: a 4 x 4 matrix with ticks and
  // a count in every cell.
  const counts = new Float32Array(16);
  const {offsets, neighbors} = tracts.contiguity;
  for (let row = 0; row < tracts.count; row++) {
    for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
      counts[
        getClass(tracts.povertyShare[row]) * 4 + getClass(tracts.povertyShare[neighbors[slot]])
      ]++;
    }
  }
  const side = 0.66;
  const left = world.width / 2 - side / 2 + 0.06;
  const rectangle = [left, 0.08, left + side, 0.08 + side] as const;
  const matrix = createMatrixMesh(4, 4, rectangle);
  const mesh = new MeshLayer(
    device,
    buffer(matrix.vertices),
    buffer(matrix.indices),
    buffer(matrix.owners),
    buffer(counts),
    matrix.triangleCount,
    'float32'
  );
  mesh.update(view, theme, {mode: 0, range: [0, Math.max(...counts)]});
  const cellLabels: RunLabel[] = [];
  for (let row = 0; row < 4; row++) {
    for (let column = 0; column < 4; column++) {
      cellLabels.push({
        text: String(counts[row * 4 + column]),
        at: [left + ((column + 0.5) / 4) * side, 0.08 + side - ((row + 0.5) / 4) * side],
        role: 'value'
      });
    }
  }
  return {
    layers: [mesh],
    owned,
    text: mergeText(
      createMatrixTicks({
        rectangle,
        rows: CLASS_NAMES,
        columns: ['<15', '15–30', '30–45', '45+'],
        rowTitle: 'tract',
        columnTitle: 'contiguous neighbour'
      }),
      {labels: cellLabels}
    )
  };
}

async function main(): Promise<void> {
  const query = new URLSearchParams(location.search);
  const previewCase = (query.get('case') ?? 'axes') as PreviewCase;
  const themeName = query.get('theme') === 'light' ? 'light' : 'dark';
  const sizeName = (query.get('size') ?? 'card') as PreviewSize;
  const size = SIZES[sizeName];
  document.documentElement.dataset.theme = themeName;
  const canvas = document.querySelector('canvas') as HTMLCanvasElement;
  canvas.style.width = `${size.width}px`;
  canvas.style.height = `${size.height}px`;
  const device = await createGpuDevice(canvas);
  if (!device || device.type !== 'webgpu') throw new Error('WebGPU required');
  const gpuDevice = device.handle as GPUDevice;
  const errors: string[] = [];
  gpuDevice.addEventListener('uncapturederror', event => {
    errors.push((event as GPUUncapturedErrorEvent).error.message);
  });
  const context = device.getDefaultCanvasContext();
  await context.initialized;
  const ratio = window.devicePixelRatio || 1;
  const width = Math.round(size.width * ratio);
  const height = Math.round(size.height * ratio);
  context.setDrawingBufferSize(width, height);
  const theme = getDemoTheme(themeName);
  const view: DemoView = {worldWidth: width / height, worldHeight: 1, width, height};
  const startTime = performance.now();
  const scene = buildScene(device, previewCase, theme, view);
  // As the wired framework does: a scene that draws text awaits the atlas before its first frame.
  // With ?lazy=1 the layers are created first and draw again when the atlas arrives.
  const isLazy = query.get('lazy') === '1';
  const textStart = performance.now();
  if (!isLazy) await loadFontPixels();
  const atlasMilliseconds = performance.now() - textStart;
  let readyCount = 0;
  let resolveReady: () => void = () => undefined;
  const allReady = new Promise<void>(resolve => {
    resolveReady = resolve;
  });
  const onReady = () => {
    readyCount++;
    if (readyCount === 2) resolveReady();
  };
  // Gridlines under the data, labels over it: two text layers, as a run would list them in draw.
  const underLayer = new TextLayer(device, onReady);
  underLayer.setText(scene.under ?? {}, {displayWidth: size.displayWidth});
  underLayer.update(view, theme);
  const textLayer = new TextLayer(device, onReady);
  textLayer.setText(scene.text, {displayWidth: size.displayWidth});
  textLayer.update(view, theme);
  const layoutMilliseconds = performance.now() - startTime;
  const textMilliseconds = performance.now() - textStart;
  const drawAll = () => drawLayers(device, theme, [underLayer, ...scene.layers, textLayer]);
  drawAll();
  if (isLazy) {
    await allReady;
    drawAll();
  }
  await gpuDevice.queue.onSubmittedWorkDone();
  const firstFrameMilliseconds = performance.now() - textStart;
  const layout = textLayer.getLayout();
  Object.assign(window, {
    textPreview: {
      ready: true,
      errors,
      width,
      height,
      placed: layout?.placed ?? [],
      dropped: layout?.dropped ?? [],
      instanceCount: layout?.instanceCount ?? 0,
      layoutMilliseconds,
      textMilliseconds,
      atlasMilliseconds,
      firstFrameMilliseconds
    }
  });
}

main().catch(error => {
  Object.assign(window, {textPreview: {ready: true, errors: [String(error)]}});
});
