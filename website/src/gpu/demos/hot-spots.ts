// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Getis-Ord Gi* at rain gauges around the Carolinas during Hurricane Helene.
// Changing day changes the observed totals; changing radius recomputes the spatial weights.

import {
  getGPUNeighborSearchParameterValues,
  getGPUSpatialAutocorrelationParameterValues,
  GPUHotSpotAnalysis,
  GPUNeighborSearch,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getRainStations, getRainNeighborCapacity} from '../data/point-observations';
import type {RunGlyph} from '../runs/interaction';
import {chooseSticky, createPointIndex} from '../runs/pick';
import {createPointerDriver} from '../runs/pointer-driver';
import type {GpuScene} from '../types';
import {HOT_SPOTS_PARAMETERS, readParameterValues} from './parameters';
import {
  assertWebGPU,
  createEmptyStorageBuffer,
  createStorageBuffer,
  DotLayer,
  DrawScheduler,
  drawLayers,
  getDemoTheme,
  GraphRunner,
  importBufferView,
  makeRandom,
  type DemoView
} from './kit';

export const parameters = HOT_SPOTS_PARAMETERS;

const POINT_COUNT = 720;

const scene: GpuScene = async context => {
  const {device} = context;
  assertWebGPU(device);
  const theme = getDemoTheme(context.theme);
  const settings = readParameterValues(parameters, context.parameters);
  const worldWidth = context.width / context.height;
  const view: DemoView = {worldWidth, worldHeight: 1, width: context.width, height: context.height};
  const random = makeRandom(context.seed);
  const bounds = [0, 0, worldWidth, 1] as const;

  const {positions, rain, previousRain} = getRainStations(
    random,
    {width: worldWidth, height: 1},
    POINT_COUNT,
    true
  );
  const pointIndex = createPointIndex(positions);
  let selected = -1;
  let zValues = new Float32Array(POINT_COUNT);
  let hasZValues = false;
  let readToken = 0;
  let isDestroyed = false;

  const slotCount = getRainNeighborCapacity(positions, parameters[0].max / 100);
  const positionBuffer = createStorageBuffer(device, positions);
  const valueBuffer = createEmptyStorageBuffer(device, POINT_COUNT);
  const neighborParameters = createEmptyStorageBuffer(device, GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH);
  const statisticParameters = createEmptyStorageBuffer(
    device,
    GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
  );
  const offsets = createEmptyStorageBuffer(device, POINT_COUNT + 1);
  const neighbors = createEmptyStorageBuffer(device, slotCount);
  const weights = createEmptyStorageBuffer(device, slotCount);
  const overflow = createEmptyStorageBuffer(device, 1);
  const zScores = createEmptyStorageBuffer(device, POINT_COUNT);

  const runner = await GraphRunner.create(device, 'demo-hot-spots', graph => {
    const slots = {
      offsets: importBufferView(graph, 'offsets', offsets, 'uint32', POINT_COUNT + 1),
      neighbors: importBufferView(graph, 'neighbors', neighbors, 'uint32', slotCount),
      weights: importBufferView(graph, 'weights', weights, 'float32', slotCount)
    };
    graph.add(
      new GPUNeighborSearch({
        mode: 'radius',
        gridSize: [32, 32],
        positions: importBufferView(graph, 'positions', positionBuffer, 'float32x2', POINT_COUNT),
        parameters: importBufferView(
          graph,
          'neighbor-parameters',
          neighborParameters,
          'float32',
          GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
        ),
        output: {
          ...slots,
          status: {overflow: importBufferView(graph, 'overflow', overflow, 'uint32', 1)}
        }
      })
    );
    graph.add(
      new GPUHotSpotAnalysis({
        weights: slots,
        values: importBufferView(graph, 'values', valueBuffer, 'float32', POINT_COUNT),
        parameters: importBufferView(
          graph,
          'statistic-parameters',
          statisticParameters,
          'float32',
          GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
        ),
        output: {zScores: importBufferView(graph, 'z-scores', zScores, 'float32', POINT_COUNT)}
      })
    );
  });

  const halo = new DotLayer(device, positionBuffer, valueBuffer, POINT_COUNT);
  const hotSpots = new DotLayer(device, positionBuffer, zScores, POINT_COUNT);
  let isDirty = true;

  const draw = () => {
    if (isDirty) {
      isDirty = false;
      valueBuffer.write(settings.day === 0 ? previousRain : rain);
      neighborParameters.write(
        getGPUNeighborSearchParameterValues({
          bounds,
          radius: settings.radius / 100,
          weightKind: 'binary'
        })
      );
      statisticParameters.write(getGPUSpatialAutocorrelationParameterValues({}));
      runner.run();
      const token = ++readToken;
      zScores
        .readAsync()
        .then(bytes => {
          if (token !== readToken || isDestroyed) return;
          zValues = new Float32Array(bytes.slice().buffer);
          hasZValues = true;
          driver.refresh();
        })
        .catch(() => {});
    }
    halo.update(view, theme, {
      mode: 1,
      range: [0, 180],
      radius: 10,
      alpha: 0.12,
      ramp: theme.diverging
    });
    hotSpots.update(view, theme, {
      mode: 1,
      range: [-4, 4],
      radius: 4,
      ramp: theme.diverging,
      solid: [...theme.muted, 0.4]
    });
    drawLayers(device, theme, [halo, hotSpots, ...driver.getLayers()]);
  };
  const scheduler = new DrawScheduler(draw);

  const driver = createPointerDriver({
    device,
    view,
    theme,
    reducedMotion: context.reducedMotion,
    kind: 'probe',
    hint: 'hover a gauge',
    rest: [0.5, 0.5],
    reportReadout: context.reportReadout,
    requestDraw: () => scheduler.request(),
    resolve(pointer) {
      selected = chooseSticky(pointIndex, pointer.x, pointer.y, selected);
      if (selected < 0) return {};
      const at = [positions[selected * 2], positions[selected * 2 + 1]] as const;
      const radius = settings.radius / 100;
      const glyphs: RunGlyph[] = [
        {shape: 'ring', at, radius, dashed: true},
        {shape: 'dot', at}
      ];
      const inside = pointIndex.within(at[0], at[1], radius).filter(index => index !== selected);
      for (const index of inside) {
        glyphs.push({
          shape: 'line',
          from: at,
          to: [positions[index * 2], positions[index * 2 + 1]]
        });
      }
      const rainfall = (settings.day === 0 ? previousRain : rain)[selected];
      const z = zValues[selected];
      const statistic =
        hasZValues && Number.isFinite(z)
          ? ` · Gi* z ${z >= 0 ? '+' : '\u2212'}${Math.abs(z).toFixed(1)}`
          : '';
      return {
        glyphs,
        readout: `${rainfall.toFixed(0)} mm · ${inside.length} neighbors${statistic}`
      };
    }
  });

  return {
    gesture: driver.gesture,
    render(time) {
      driver.advance(time);
      scheduler.flush();
    },
    resize(width, height) {
      view.width = width;
      view.height = height;
      driver.refresh();
      scheduler.request();
    },
    setPointer: driver.setPointer,
    resetPointer: driver.resetPointer,
    setParameters(update) {
      Object.assign(settings, update);
      isDirty = true;
      driver.refresh();
      scheduler.request();
    },
    destroy() {
      isDestroyed = true;
      scheduler.cancel();
      driver.destroy();
      halo.destroy();
      hotSpots.destroy();
      runner.destroy();
      for (const buffer of [
        positionBuffer,
        valueBuffer,
        neighborParameters,
        statisticParameters,
        offsets,
        neighbors,
        weights,
        overflow,
        zScores
      ]) {
        buffer.destroy();
      }
    }
  };
};

export default scene;
