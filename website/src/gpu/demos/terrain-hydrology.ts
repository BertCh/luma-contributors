// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Demo: GPUTerrainDerivatives and GPUTerrainFlow. Grand Canyon elevations, sliders for the sun
// (Horn hillshade, recomputed on the GPU), slope exaggeration, and the stream threshold of a network
// extracted by depression filling, D8 routing and flow accumulation.

import {
  getGPUTerrainDerivativesParameterValues,
  getGPUTerrainFlowParameterValues,
  GPUTerrainDerivatives,
  GPUTerrainFlow
} from '@luma-contributors/analysis/gpu-terrain';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {sampleElevation} from '../data/elevation';
import type {RunGlyph} from '../runs/interaction';
import {createPointerDriver} from '../runs/pointer-driver';
import type {GpuScene} from '../types';
import {readParameterValues, TERRAIN_HYDROLOGY_PARAMETERS} from './parameters';
import {
  assertWebGPU,
  createEmptyStorageBuffer,
  createStorageBuffer,
  DrawScheduler,
  drawLayers,
  getDemoTheme,
  GraphRunner,
  importBufferView,
  RasterLayer,
  type Color,
  type DemoView
} from './kit';

export const parameters = TERRAIN_HYDROLOGY_PARAMETERS;

const ROWS = 96;
const MINIMUM_ALTITUDE = 5;
const MAXIMUM_ALTITUDE = 85;
const LAND_RAMP: Color[] = ['#17323f', '#2c6556', '#6f9b55', '#c4b868', '#a98457', '#f1eee6'].map(
  value => {
    const number = Number.parseInt(value.slice(1), 16);
    return [
      ((number >> 16) & 255) / 255,
      ((number >> 8) & 255) / 255,
      (number & 255) / 255
    ] as Color;
  }
);

const scene: GpuScene = async context => {
  const {device} = context;
  assertWebGPU(device);
  const theme = getDemoTheme(context.theme);
  const values = readParameterValues(parameters, context.parameters);
  const columns = Math.max(48, Math.min(256, Math.round((ROWS * context.width) / context.height)));
  const worldWidth = columns / ROWS;
  const view: DemoView = {worldWidth, worldHeight: 1, width: context.width, height: context.height};
  const {elevation, cellSize} = sampleElevation(columns, ROWS);
  const grid = makeGPURasterMetadata({
    width: columns,
    height: ROWS,
    cellSize,
    origin: [0, ROWS * cellSize]
  });
  const cellCount = columns * ROWS;

  let low = Infinity;
  let high = -Infinity;
  for (const value of elevation) {
    low = Math.min(low, value);
    high = Math.max(high, value);
  }

  const elevationBuffer = createStorageBuffer(device, elevation);
  const derivativeParameters = createEmptyStorageBuffer(device, 8);
  const flowParameters = createEmptyStorageBuffer(device, 8);
  const hillshade = createEmptyStorageBuffer(device, cellCount);
  const streams = createEmptyStorageBuffer(device, cellCount);

  const runner = await GraphRunner.create(device, 'demo-terrain-hydrology', graph => {
    const band = {
      id: 'elevation',
      format: 'float32',
      storage: {
        kind: 'buffer',
        values: importBufferView(graph, 'elevation', elevationBuffer, 'float32', cellCount)
      }
    } as const;
    graph.add(
      new GPUTerrainDerivatives({
        grid,
        elevation: band,
        parameters: importBufferView(
          graph,
          'derivative-parameters',
          derivativeParameters,
          'float32',
          8
        ),
        output: {hillshade: importBufferView(graph, 'hillshade', hillshade, 'float32', cellCount)}
      })
    );
    graph.add(
      new GPUTerrainFlow({
        grid,
        elevation: band,
        parameters: importBufferView(graph, 'flow-parameters', flowParameters, 'float32', 8),
        fillDepressions: true,
        output: {streams: importBufferView(graph, 'streams', streams, 'uint32', cellCount)}
      })
    );
  });

  const land = new RasterLayer(device, elevationBuffer, columns, ROWS);
  const shade = new RasterLayer(device, hillshade, columns, ROWS);
  const rivers = new RasterLayer(device, streams, columns, ROWS, 'uint32');
  const water = theme.categories[0];
  const riverColors = [
    [water[0], water[1], water[2], 0],
    [0.35, 0.8, 1, 1]
  ] as const;
  let isDirty = true;

  const draw = () => {
    if (isDirty) {
      isDirty = false;
      derivativeParameters.write(
        getGPUTerrainDerivativesParameterValues(
          {
            azimuthDegrees: values.azimuthDegrees,
            altitudeDegrees: values.altitudeDegrees,
            zFactor: values.zFactor
          },
          grid
        )
      );
      flowParameters.write(
        getGPUTerrainFlowParameterValues(
          {streamThreshold: Math.round(values.streamThreshold), fillEpsilon: 0.02},
          grid
        )
      );
      runner.run();
    }
    land.update(view, theme, {range: [low, high], ramp: LAND_RAMP, smoothing: true});
    // Hillshade 0..1 inverted into black with alpha: dark where the slope faces away from the sun.
    shade.update(view, theme, {
      range: [0.95, 0.3],
      ramp: [[0, 0, 0]],
      fade: true,
      alpha: 0.55,
      smoothing: true
    });
    rivers.update(view, theme, {mode: 2, categories: riverColors});
    drawLayers(device, theme, [land, shade, rivers, ...driver.getLayers()]);
  };
  const scheduler = new DrawScheduler(draw);

  // Compass angle around the picture center is the sun azimuth; distance from it is the altitude
  // (far is low). The pointer writes the sliders only once the reader has moved it.
  const getReach = () => Math.min(view.width, view.height) / 2;
  let isPointerOwner = false;
  const driver = createPointerDriver({
    device,
    view,
    theme,
    reducedMotion: context.reducedMotion,
    kind: 'aim',
    hint: 'aim the sun',
    rest() {
      const azimuth = (values.azimuthDegrees * Math.PI) / 180;
      const reach =
        ((MAXIMUM_ALTITUDE - values.altitudeDegrees) / (MAXIMUM_ALTITUDE - MINIMUM_ALTITUDE)) *
        getReach();
      return [
        0.5 + (Math.sin(azimuth) * reach) / view.width,
        0.5 - (Math.cos(azimuth) * reach) / view.height
      ];
    },
    reportReadout: context.reportReadout,
    requestDraw: () => scheduler.request(),
    resolve(pointer) {
      const offsetX = (pointer.unitX - 0.5) * view.width;
      const offsetY = (pointer.unitY - 0.5) * view.height;
      if (isPointerOwner) {
        const azimuthDegrees =
          Math.round((Math.atan2(offsetX, -offsetY) * 180) / Math.PI + 360) % 360;
        const nearness = 1 - Math.min(1, Math.hypot(offsetX, offsetY) / getReach());
        const altitudeDegrees = Math.round(
          MINIMUM_ALTITUDE + nearness * (MAXIMUM_ALTITUDE - MINIMUM_ALTITUDE)
        );
        if (
          azimuthDegrees !== values.azimuthDegrees ||
          altitudeDegrees !== values.altitudeDegrees
        ) {
          values.azimuthDegrees = azimuthDegrees;
          values.altitudeDegrees = altitudeDegrees;
          isDirty = true;
          scheduler.request();
          context.reportParameters?.({azimuthDegrees, altitudeDegrees});
        }
      }
      const center: [number, number] = [view.worldWidth / 2, view.worldHeight / 2];
      const glyphs: RunGlyph[] = [
        {shape: 'arrow', from: center, to: [pointer.x, pointer.y]},
        {shape: 'dot', at: center}
      ];
      return {
        glyphs,
        readout: `sun ${values.azimuthDegrees}° · ${values.altitudeDegrees}° up`
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
      isPointerOwner = false;
      driver.release();
      isDirty = true;
      scheduler.request();
    },
    resize(width, height) {
      view.width = width;
      view.height = height;
      driver.refresh();
      scheduler.request();
    },
    setPointer(position) {
      if (position) isPointerOwner = true;
      driver.setPointer(position);
    },
    resetPointer: driver.resetPointer,
    destroy() {
      scheduler.cancel();
      driver.destroy();
      land.destroy();
      shade.destroy();
      rivers.destroy();
      runner.destroy();
      for (const buffer of [
        elevationBuffer,
        derivativeParameters,
        flowParameters,
        hillshade,
        streams
      ]) {
        buffer.destroy();
      }
    }
  };
};

export default scene;
