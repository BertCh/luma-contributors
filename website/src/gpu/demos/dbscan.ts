// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Demo: GPUDBSCAN (DBSCAN) of Japanese earthquake epicenters. The epsilon and
// minimum-samples sliders relabel every point on the GPU, so clusters merge, split and shed noise.

import {
  getGPUDBSCANParameterValues,
  GPUDBSCAN,
  GPU_DBSCAN_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getEarthquakes} from '../data/point-observations';
import type {RunGlyph} from '../runs/interaction';
import {chooseSticky, createPointIndex} from '../runs/pick';
import {createPointerDriver} from '../runs/pointer-driver';
import type {GpuScene} from '../types';
import {createJapanCoast} from './japan-coast';
import {DBSCAN_PARAMETERS, readParameterValues, validateParameterRenames} from './parameters';
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

export const parameters = DBSCAN_PARAMETERS;

const scene: GpuScene = async context => {
  const {device} = context;
  assertWebGPU(device);
  const theme = getDemoTheme(context.theme);
  const values = readParameterValues(parameters, context.parameters);
  const worldWidth = context.width / context.height;
  const view: DemoView = {worldWidth, worldHeight: 1, width: context.width, height: context.height};
  const random = makeRandom(context.seed);
  const bounds = [0, 0, worldWidth, 1] as const;

  const {positions, count: pointCount} = getEarthquakes(
    random,
    {width: worldWidth, height: 1},
    820
  );

  const pointIndex = createPointIndex(positions, pointCount);
  let selected = -1;
  const positionBuffer = createStorageBuffer(device, positions);
  const parameterBuffer = createEmptyStorageBuffer(device, GPU_DBSCAN_PARAMETER_LENGTH);
  const labels = createEmptyStorageBuffer(device, pointCount);
  const clusterCount = createEmptyStorageBuffer(device, 1);

  const runner = await GraphRunner.create(device, 'demo-dbscan', graph => {
    graph.add(
      new GPUDBSCAN({
        positions: importBufferView(graph, 'positions', positionBuffer, 'float32x2', pointCount),
        parameters: importBufferView(
          graph,
          'parameters',
          parameterBuffer,
          'float32',
          GPU_DBSCAN_PARAMETER_LENGTH
        ),
        gridSize: [64, 36],
        output: {
          labels: importBufferView(graph, 'labels', labels, 'uint32', pointCount),
          clusterCount: importBufferView(graph, 'cluster-count', clusterCount, 'uint32', 1)
        }
      })
    );
  });

  const coast = createJapanCoast(device, worldWidth);
  const dots = new DotLayer(device, positionBuffer, labels, pointCount, 'uint32');
  let isDirty = true;

  const draw = () => {
    if (isDirty) {
      isDirty = false;
      parameterBuffer.write(
        getGPUDBSCANParameterValues({
          bounds,
          epsilon: values.epsilon / 100,
          minimumSamples: Math.round(values.minimumSamples)
        })
      );
      runner.run();
    }
    dots.update(view, theme, {
      mode: 2,
      radius: 4.2,
      solid: [...theme.muted, 0.55],
      hillshade: 0.4
    });
    coast.layer.update(view, theme, {radius: 0.7, solid: [...theme.muted, 0.8]});
    drawLayers(device, theme, [coast.layer, dots, ...driver.getLayers()]);
  };
  const scheduler = new DrawScheduler(draw);

  const driver = createPointerDriver({
    device,
    view,
    theme,
    reducedMotion: context.reducedMotion,
    kind: 'probe',
    hint: 'hover an epicenter',
    rest: [0.5, 0.5],
    reportReadout: context.reportReadout,
    requestDraw: () => scheduler.request(),
    resolve(pointer) {
      selected = chooseSticky(pointIndex, pointer.x, pointer.y, selected);
      if (selected < 0) return {};
      const at = [positions[selected * 2], positions[selected * 2 + 1]] as const;
      const epsilon = values.epsilon / 100;
      const inside = pointIndex.within(at[0], at[1], epsilon).filter(index => index !== selected);
      const glyphs: RunGlyph[] = [
        {shape: 'ring', at, radius: epsilon},
        {shape: 'dot', at}
      ];
      for (const index of inside) {
        glyphs.push({
          shape: 'line',
          from: at,
          to: [positions[index * 2], positions[index * 2 + 1]]
        });
      }
      // A point counts itself toward minimumSamples; a border point is within epsilon of a core point.
      const minimumSamples = Math.round(values.minimumSamples);
      const isCore = (index: number) =>
        pointIndex.within(positions[index * 2], positions[index * 2 + 1], epsilon).length >=
        minimumSamples;
      const role =
        inside.length + 1 >= minimumSamples ? 'core' : inside.some(isCore) ? 'border' : 'noise';
      return {glyphs, readout: `${inside.length} within epsilon \u00b7 ${role}`};
    }
  });

  return {
    gesture: driver.gesture,
    render(time) {
      driver.advance(time);
      scheduler.flush();
    },
    setParameters(update) {
      validateParameterRenames(parameters, update);
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
      scheduler.cancel();
      driver.destroy();
      coast.destroy();
      dots.destroy();
      runner.destroy();
      for (const buffer of [positionBuffer, parameterBuffer, labels, clusterCount]) {
        buffer.destroy();
      }
    }
  };
};

export default scene;
