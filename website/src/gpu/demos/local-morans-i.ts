// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Local Moran clusters of measured rainfall at southeastern US gauges during Hurricane Helene.

import {
  getGPUNeighborSearchParameterValues,
  getGPUSpatialAutocorrelationParameterValues,
  GPULocalMoran,
  GPUNeighborSearch,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getRainStations, getRainNeighborCapacity} from '../data/point-observations';
import type {RunGlyph} from '../runs/interaction';
import {chooseSticky, createPointIndex} from '../runs/pick';
import {createPointerDriver} from '../runs/pointer-driver';
import type {GpuScene} from '../types';
import {LOCAL_MORANS_I_PARAMETERS, readParameterValues} from './parameters';
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

export const parameters = LOCAL_MORANS_I_PARAMETERS;

const POINT_COUNT = 720;

const scene: GpuScene = async context => {
  const {device} = context;
  assertWebGPU(device);
  const theme = getDemoTheme(context.theme);
  const values = readParameterValues(parameters, context.parameters);
  const worldWidth = context.width / context.height;
  const view: DemoView = {worldWidth, worldHeight: 1, width: context.width, height: context.height};
  const random = makeRandom(context.seed);
  const bounds = [0, 0, worldWidth, 1] as const;

  const {positions, rain} = getRainStations(
    random,
    {width: worldWidth, height: 1},
    POINT_COUNT,
    true
  );
  const pointIndex = createPointIndex(positions);
  let selected = -1;
  let currentValues = Float32Array.from(rain);
  let clusterClasses: Uint32Array | null = null;
  let readToken = 0;
  let isDestroyed = false;
  const noise = Float32Array.from({length: POINT_COUNT}, () => random() - 0.5);
  const slotCount = getRainNeighborCapacity(positions, parameters[0].max / 100);
  const positionBuffer = createStorageBuffer(device, positions);
  const valueBuffer = createEmptyStorageBuffer(device, POINT_COUNT);
  const neighborParameters = createEmptyStorageBuffer(device, GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH);
  const moranParameters = createEmptyStorageBuffer(
    device,
    GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
  );
  const offsets = createEmptyStorageBuffer(device, POINT_COUNT + 1);
  const neighbors = createEmptyStorageBuffer(device, slotCount);
  const weights = createEmptyStorageBuffer(device, slotCount);
  const overflow = createEmptyStorageBuffer(device, 1);
  const zScores = createEmptyStorageBuffer(device, POINT_COUNT);
  const quadrants = createEmptyStorageBuffer(device, POINT_COUNT);
  const globalStatistics = createEmptyStorageBuffer(device, 5);

  const runner = await GraphRunner.create(device, 'demo-local-morans-i', graph => {
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
      new GPULocalMoran({
        weights: slots,
        values: importBufferView(graph, 'values', valueBuffer, 'float32', POINT_COUNT),
        parameters: importBufferView(
          graph,
          'moran-parameters',
          moranParameters,
          'float32',
          GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
        ),
        minimumNeighbors: 3,
        output: {
          zScores: importBufferView(graph, 'z-scores', zScores, 'float32', POINT_COUNT),
          quadrants: importBufferView(graph, 'quadrants', quadrants, 'uint32', POINT_COUNT),
          globalStatistics: importBufferView(
            graph,
            'global-statistics',
            globalStatistics,
            'float32',
            5
          )
        }
      })
    );
  });

  // Quadrant colors: not significant, HH, LH, LL, HL.
  const quadrantColors: (readonly [number, number, number, number])[] =
    context.theme === 'dark'
      ? [
          [...theme.muted, 0.45],
          [1, 0.36, 0.34, 1],
          [0.55, 0.76, 1, 1],
          [0.25, 0.5, 1, 1],
          [1, 0.72, 0.3, 1]
        ]
      : [
          [...theme.muted, 0.5],
          [0.82, 0.16, 0.2, 1],
          [0.45, 0.65, 0.95, 1],
          [0.12, 0.33, 0.8, 1],
          [0.9, 0.5, 0.05, 1]
        ];

  const field = new DotLayer(device, positionBuffer, valueBuffer, POINT_COUNT);
  const clusters = new DotLayer(device, positionBuffer, quadrants, POINT_COUNT, 'uint32');
  let isDirty = true;

  const draw = () => {
    if (isDirty) {
      isDirty = false;
      currentValues = Float32Array.from(rain, (value, row) =>
        Math.max(0, value + noise[row] * values.noise)
      );
      valueBuffer.write(currentValues);
      neighborParameters.write(
        getGPUNeighborSearchParameterValues({
          bounds,
          radius: values.radius / 100,
          weightKind: 'binary'
        })
      );
      moranParameters.write(
        getGPUSpatialAutocorrelationParameterValues({significanceLevel: values.significance})
      );
      runner.run();
      const token = ++readToken;
      quadrants
        .readAsync()
        .then(bytes => {
          if (token !== readToken || isDestroyed) return;
          clusterClasses = new Uint32Array(bytes.slice().buffer);
          driver.refresh();
        })
        .catch(() => {});
    }
    field.update(view, theme, {
      mode: 1,
      range: [0, 180],
      radius: 11,
      alpha: 0.2,
      ramp: theme.diverging
    });
    clusters.update(view, theme, {
      mode: 2,
      categories: quadrantColors,
      radius: 4,
      solid: [...theme.muted, 0.4]
    });
    drawLayers(device, theme, [field, clusters, ...driver.getLayers()]);
  };
  const scheduler = new DrawScheduler(draw);

  const clusterNames = ['not significant', 'high-high', 'low-high', 'low-low', 'high-low'];
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
      const radius = values.radius / 100;
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
      const label = clusterClasses ? (clusterNames[clusterClasses[selected]] ?? '') : '';
      return {
        glyphs,
        readout: `${currentValues[selected].toFixed(0)} mm · ${label || `${inside.length} neighbors`}`
      };
    }
  });

  return {
    gesture: driver.gesture,
    render(time) {
      driver.advance(time);
      scheduler.flush();
    },
    setParameters(update) {
      Object.assign(values, update);
      isDirty = true;
      driver.refresh();
      scheduler.request();
    },
    resize(width, height) {
      view.width = width;
      view.height = height;
      driver.refresh();
      scheduler.request();
    },
    setPointer: driver.setPointer,
    resetPointer: driver.resetPointer,
    destroy() {
      isDestroyed = true;
      scheduler.cancel();
      driver.destroy();
      field.destroy();
      clusters.destroy();
      runner.destroy();
      for (const buffer of [
        positionBuffer,
        valueBuffer,
        neighborParameters,
        moranParameters,
        offsets,
        neighbors,
        weights,
        overflow,
        zScores,
        quadrants,
        globalStatistics
      ]) {
        buffer.destroy();
      }
    }
  };
};

export default scene;
