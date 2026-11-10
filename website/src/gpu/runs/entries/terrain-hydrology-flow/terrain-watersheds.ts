// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainWatersheds on the D8 directions of GPUTerrainFlow: every cell is labelled with the
// pour point its flow path reaches. The slider activates more of the pour points in the view;
// unused rows hold the "none" marker and cells that reach no pour point stay unlabelled.

import {GPU_TERRAIN_FLOW_NONE, GPUTerrainWatersheds} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {parameters} from './terrain-watersheds.parameters';
import {
  addTerrainFlow,
  createTerrainInputs,
  findValleyCell,
  getTerrainCell,
  getTerrainCellCenter,
  snapToChannelCell
} from './shared/terrain';

const POUR_POINT_CAPACITY = 8;
/** One more row than the sliders reach, for the pour point the pointer places. */
const ROW_CAPACITY = POUR_POINT_CAPACITY + 1;
const REST: readonly [number, number] = [0.62, 0.5];

export default defineOperationRun({
  parameters,
  caption: 'Each color is the area draining to one pour point; the pointer adds one.',
  createInputs: context => {
    const terrain = createTerrainInputs(context);
    // Pour points along the trunk valley, which drains toward the west edge: the area upstream of each
    // is its basin, and the westernmost takes what is left.
    const pourPoints = new Uint32Array(POUR_POINT_CAPACITY);
    for (let index = 0; index < POUR_POINT_CAPACITY; index++) {
      pourPoints[index] = findValleyCell(terrain, 0.05 + 0.125 * index);
    }
    return {...terrain, world: context.world, pourPoints, placed: null as number | null};
  },
  build(builder, terrain) {
    const flow = addTerrainFlow(builder, terrain);
    flow.writeStreamThreshold(25);
    const pourPoints = builder.allocate('pour-points', 'uint32', ROW_CAPACITY);
    const labels = builder.allocate('labels', 'uint32', terrain.cellCount);
    const pourDots = builder.allocate('pour-dots', 'float32x2', ROW_CAPACITY);
    const pourIds = builder.upload(
      'pour-ids',
      'uint32',
      Uint32Array.from({length: ROW_CAPACITY}, (_, index) => index)
    );
    builder.graph.add(
      new GPUTerrainWatersheds({
        grid: terrain.grid,
        flowDirections: flow.flowDirections.view,
        pourPoints: pourPoints.view,
        output: {labels: labels.view}
      })
    );
    return {
      outputs: {
        filledElevation: flow.filledElevation,
        accumulation: flow.accumulation,
        labels,
        pourDots,
        pourIds
      },
      update: values => {
        const count = Math.round(values.pourPointCount);
        const active = new Uint32Array(ROW_CAPACITY).fill(GPU_TERRAIN_FLOW_NONE);
        active.set(terrain.pourPoints.subarray(0, count));
        // The placed pour point follows the fixed ones, so it takes the next color.
        if (terrain.placed !== null) active[count] = terrain.placed;
        pourPoints.write(active);
        const positions = new Float32Array(ROW_CAPACITY * 2).fill(Number.NaN);
        active.forEach((cell, row) => {
          if (cell === GPU_TERRAIN_FLOW_NONE) return;
          positions.set(getTerrainCellCenter(terrain, terrain.world, cell), row * 2);
        });
        pourDots.write(positions);
      }
    };
  },
  // The pointer places a pour point: the cell under it snaps to the strongest channel within three
  // cells, and the area draining to it is delineated as one more basin. Resting removes it.
  interaction: {
    kind: 'place',
    hint: 'place a pour point',
    rest: REST,
    resolve(pointer, {inputs, outputs, world, peek}) {
      const accumulation = peek(outputs.accumulation);
      const labels = peek(outputs.labels);
      const isRest = Math.hypot(pointer.unitX - REST[0], pointer.unitY - REST[1]) < 0.003;
      if (isRest || !accumulation) {
        const changed = inputs.placed !== null;
        inputs.placed = null;
        return {rerun: changed};
      }
      const under = getTerrainCell(inputs, world, pointer.x, pointer.y).index;
      const snapped = snapToChannelCell(inputs, accumulation, under, 3);
      const changed = inputs.placed !== snapped;
      inputs.placed = snapped;
      const at = getTerrainCellCenter(inputs, world, snapped);
      const column = snapped % inputs.columns;
      const row = Math.floor(snapped / inputs.columns);
      let basinCells = 0;
      if (labels) {
        const label = labels[snapped];
        for (const value of labels) if (value === label) basinCells++;
      }
      return {
        rerun: changed,
        glyphs: [
          {shape: 'line', from: [pointer.x, pointer.y], to: at, dashed: true},
          {shape: 'crosshair', at}
        ],
        readout: `pour point [${column}, ${row}]${basinCells > 0 ? ` · basin ${basinCells.toLocaleString('en-US')} cells` : ''}`
      };
    }
  },
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.filledElevation,
      columns: inputs.columns,
      rows: inputs.rows,
      hillshade: 0.8,
      elevation: 30
    },
    {
      layer: 'raster',
      values: outputs.labels,
      columns: inputs.columns,
      rows: inputs.rows,
      alpha: 0.7
    },
    {
      layer: 'dots',
      positions: outputs.pourDots,
      values: outputs.pourIds,
      count: Math.round(values.pourPointCount) + (inputs.placed === null ? 0 : 1),
      radius: 4.5
    }
  ]
});
