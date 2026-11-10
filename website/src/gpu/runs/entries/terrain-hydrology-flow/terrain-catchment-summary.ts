// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainCatchmentSummary over the basins GPUTerrainWatersheds delineates from pour points along
// the valley: per basin elevation statistics and a hypsometric histogram (the bars, one group of
// histogramBins per basin, colored by cell count). The basins are drawn faintly behind.

import {
  GPU_TERRAIN_FLOW_NONE,
  GPUTerrainCatchmentSummary,
  GPUTerrainWatersheds
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {getChartPosition} from '../../pick';
import {parameters} from './terrain-catchment-summary.parameters';
import {
  addTerrainFlow,
  createTerrainInputs,
  findValleyCell,
  getTerrainCell,
  getTerrainCellCenter,
  getTerrainContextLayer
} from './shared/terrain';

const BASIN_CAPACITY = 8;
const CHART_RECTANGLE = [0.05, 0.06, 0.95, 0.5] as const;
/** One color per basin, shared by the map and its bar group. */
const BASIN_COLORS = [
  [0.95, 0.55, 0.2],
  [0.3, 0.7, 0.95],
  [0.55, 0.85, 0.4],
  [0.9, 0.4, 0.65],
  [0.75, 0.65, 0.95],
  [0.95, 0.85, 0.3],
  [0.3, 0.85, 0.75],
  [0.85, 0.45, 0.4]
] as const;
const REST = [0.84, 0.28] as const;
const HIGHLIGHT = [
  [0, 0, 0, 0],
  [1, 1, 1, 0.9]
] as const;

function getMedian(values: number[]): number {
  const sorted = [...values].sort((first, second) => first - second);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
}

export default defineOperationRun({
  parameters,
  caption: 'Hypsometric histogram of every basin: how much of its area lies at each elevation.',
  createInputs: context => {
    const terrain = createTerrainInputs(context);
    const pourPoints = new Uint32Array(BASIN_CAPACITY);
    for (let index = 0; index < BASIN_CAPACITY; index++) {
      pourPoints[index] = findValleyCell(terrain, 0.05 + 0.125 * index);
    }
    return {...terrain, world: context.world, pourPoints, focus: {basin: -1, key: ''}};
  },
  build(builder, terrain, values) {
    const flow = addTerrainFlow(builder, terrain);
    flow.writeStreamThreshold(25);
    const histogramBins = Math.round(values.histogramBins);
    const pourPoints = builder.allocate('pour-points', 'uint32', BASIN_CAPACITY);
    const basins = builder.allocate('basins', 'uint32', terrain.cellCount);
    builder.graph.add(
      new GPUTerrainWatersheds({
        grid: terrain.grid,
        flowDirections: flow.flowDirections.view,
        pourPoints: pourPoints.view,
        output: {labels: basins.view}
      })
    );
    const hypsometry = builder.allocate('hypsometry', 'float32', BASIN_CAPACITY * histogramBins);
    builder.graph.add(
      new GPUTerrainCatchmentSummary({
        grid: terrain.grid,
        elevation: flow.filledElevation.view,
        basinIds: basins.view,
        basinCount: BASIN_CAPACITY,
        streams: flow.streams.view,
        flowDirections: flow.flowDirections.view,
        histogramBins,
        output: {
          statistics: builder.allocate('statistics', 'float32', BASIN_CAPACITY * 6).view,
          hypsometry: hypsometry.view,
          profiles: builder.allocate('profiles', 'float32', terrain.cellCount * 3).view,
          profileStatus: builder.allocate('profile-status', 'uint32', terrain.cellCount).view
        }
      })
    );
    // The focused basin, for the map: 1 on its cells.
    const focusMask = builder.allocate('focus-mask', 'uint32', terrain.cellCount);
    const baseline = builder.upload(
      'baseline',
      'float32x4',
      Float32Array.of(
        CHART_RECTANGLE[0] * terrain.world.width,
        CHART_RECTANGLE[1] * terrain.world.height,
        CHART_RECTANGLE[2] * terrain.world.width,
        CHART_RECTANGLE[1] * terrain.world.height
      )
    );
    return {
      outputs: {filledElevation: flow.filledElevation, basins, hypsometry, focusMask, baseline},
      update: current => {
        const active = new Uint32Array(BASIN_CAPACITY).fill(GPU_TERRAIN_FLOW_NONE);
        active.set(terrain.pourPoints.subarray(0, Math.round(current.pourPointCount)));
        pourPoints.write(active);
      }
    };
  },
  rebuildOn: ['histogramBins'],
  async afterRun({outputs, values, upload}) {
    // One bar buffer per basin: its group of bins, zero elsewhere, so each can take its basin's color.
    const bytes = await outputs.hypsometry.buffer.readAsync();
    const histogram = new Float32Array(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
    );
    const binCount = Math.round(values.histogramBins);
    const activeCount = Math.round(values.pourPointCount);
    let maximum = 1;
    for (let index = 0; index < activeCount * binCount; index++) {
      if (Number.isFinite(histogram[index])) maximum = Math.max(maximum, histogram[index]);
    }
    const groups = Array.from({length: activeCount}, (_, basin) => {
      const bars = new Float32Array(activeCount * binCount);
      for (let bin = 0; bin < binCount; bin++) {
        const value = histogram[basin * binCount + bin];
        bars[basin * binCount + bin] = Number.isFinite(value) ? value : 0;
      }
      return upload(`basin-${basin}-bars`, 'float32', bars);
    });
    return {groups, maximum};
  },
  // The pointer picks a basin, on the map or by its bar group, and that basin is lit in both.
  interaction: {
    kind: 'probe',
    hint: 'hover a basin',
    rest: REST,
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const labels = peek(outputs.basins);
      if (!labels) return {};
      const activeCount = Math.round(values.pourPointCount);
      const chart = getChartPosition(CHART_RECTANGLE, world, pointer.x, pointer.y);
      // At rest nothing is hovered, so the poster is the plain default picture.
      const isRest = Math.hypot(pointer.unitX - REST[0], pointer.unitY - REST[1]) < 0.003;
      let basin = -1;
      if (isRest) {
        basin = -1;
      } else if (chart.isInside) {
        basin = Math.min(activeCount - 1, Math.floor(chart.u * activeCount));
      } else {
        const label = labels[getTerrainCell(inputs, world, pointer.x, pointer.y).index];
        if (label < activeCount) basin = label;
      }
      const key = `${basin}:${activeCount}:${labels.length}`;
      if (key !== inputs.focus.key) {
        inputs.focus = {basin, key};
        const mask = new Uint32Array(inputs.cellCount);
        for (let cell = 0; cell < inputs.cellCount; cell++) {
          if (basin >= 0 && labels[cell] === basin) mask[cell] = 1;
        }
        outputs.focusMask.write(mask);
      }
      if (basin < 0) return {focus: -1, spotlight: null, readout: `${activeCount} basins`};
      const elevations: number[] = [];
      let cells = 0;
      let center: [number, number] = [0, 0];
      for (let cell = 0; cell < inputs.cellCount; cell++) {
        if (labels[cell] !== basin) continue;
        cells++;
        elevations.push(inputs.elevation[cell]);
        const position = getTerrainCellCenter(inputs, inputs.world, cell);
        center = [center[0] + position[0], center[1] + position[1]];
      }
      return {
        focus: basin,
        spotlight: null,
        readout: `basin ${basin + 1} · ${cells.toLocaleString('en-US')} cells · median elevation ${getMedian(elevations).toFixed(2)}`
      };
    }
  },
  draw: ({inputs, outputs, extra, interaction}) => [
    getTerrainContextLayer(inputs, outputs.filledElevation),
    {
      layer: 'raster',
      values: outputs.basins,
      columns: inputs.columns,
      rows: inputs.rows,
      categories: BASIN_COLORS,
      alpha: interaction.focus === undefined || interaction.focus === -1 ? 0.8 : 0.5
    },
    {
      layer: 'raster',
      values: outputs.focusMask,
      columns: inputs.columns,
      rows: inputs.rows,
      categories: HIGHLIGHT,
      alpha: 0.55
    },
    {
      layer: 'lines',
      context: true,
      segments: outputs.baseline,
      radius: 1,
      color: [1, 1, 1, 0.5]
    },
    ...extra.groups.map(
      (bars, basin) =>
        ({
          layer: 'chart',
          kind: 'bars',
          values: bars,
          range: [0, extra.maximum * 1.05],
          color: [
            ...BASIN_COLORS[basin],
            interaction.focus === undefined ||
            interaction.focus === -1 ||
            interaction.focus === basin
              ? 1
              : 0.35
          ],
          rectangle: CHART_RECTANGLE
        }) as const
    )
  ]
});
