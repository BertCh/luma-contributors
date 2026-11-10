// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCellLevelSelection publishes the active pyramid level's cell count (and indirect draw
// arguments) from a one-word `activeLevel`. The markers inside each cell are drawn exactly as many
// times as the selection wrote, so the picture follows the GPU-written count.

import {
  GPUCellLevelSelection,
  GPUCellPyramid
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findParameter} from '../../interaction';
import type {RunGlyph} from '../../interaction';
import {getHonshuEvents, HONSHU_WINDOW} from '../../../data/point-observations';
import {
  addCellFill,
  addPointKeys,
  allocateCellTable,
  getCellFamily,
  getCellTableViews,
  H3_BOUNDARY_STRIDE
} from './support/cells';
import {parameters} from './cell-level-selection.parameters';

const LEVEL_CAPACITY = 2048;
const LEVEL_COLORS = [
  [0.35, 0.7, 1, 1],
  [0.4, 0.85, 0.5, 1],
  [1, 0.7, 0.3, 1],
  [0.95, 0.45, 0.6, 1]
] as const;
const LEVEL_PARAMETER = findParameter(parameters, 'activeLevel');

/** One inscribed triangle per cell: its wireframe is a marker that is drawn once per counted cell. */
function createCellMarkers(cellCount: number, stride: number, isHexagon: boolean): Uint32Array {
  const indices = new Uint32Array(cellCount * 3);
  for (let cell = 0; cell < cellCount; cell++) {
    const offsets = isHexagon ? [0, 2, 4] : [0, 1, 2];
    offsets.forEach((offset, corner) => (indices[cell * 3 + corner] = cell * stride + offset));
  }
  return indices;
}

export default defineOperationRun({
  parameters,
  geographicWindow: HONSHU_WINDOW,
  caption: 'The active level and its cell count are chosen on the GPU from one word.',
  createInputs: ({random}) => getHonshuEvents(random, 900),
  build(builder, points, values) {
    const family = getCellFamily(values.family);
    const resolutions = family === 'h3' ? [6, 5, 4, 3] : [11, 10, 9, 8];
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const pointValues = builder.upload('values', 'float32', points.values);
    const keys = addPointKeys(builder, family, resolutions[0], positions);
    const levels = resolutions.map((_, level) =>
      allocateCellTable(builder, `level-${level}`, LEVEL_CAPACITY)
    );
    const levelCounts = builder.allocate('level-counts', 'uint32', resolutions.length);
    builder.graph.add(
      new GPUCellPyramid({
        family,
        cells: keys.view,
        values: pointValues.view,
        sumScale: 65536,
        resolutions,
        output: {levels: levels.map(getCellTableViews), levelCounts: levelCounts.view}
      })
    );
    const activeLevel = builder.upload('active-level', 'uint32', Uint32Array.of(0));
    const selectedCount = builder.allocate('selected-count', 'uint32', 1);
    const drawArguments = builder.allocate('draw-arguments', 'uint32', 4);
    builder.graph.add(
      new GPUCellLevelSelection({
        levelCounts: levelCounts.view,
        activeLevel: activeLevel.view,
        output: {status: {count: selectedCount.view}, drawArguments: drawArguments.view}
      })
    );
    const stride = family === 'h3' ? H3_BOUNDARY_STRIDE : 4;
    const markers = builder.upload(
      'markers',
      'uint32',
      createCellMarkers(LEVEL_CAPACITY, stride, family === 'h3')
    );
    return {
      outputs: {
        levels,
        selectedCount,
        markers,
        levelFills: levels.map((level, index) =>
          addCellFill(builder, family, level.cells, `fill-${index}`)
        )
      },
      update: current => activeLevel.write(Uint32Array.of(current.activeLevel))
    };
  },
  rebuildOn: ['family'],
  // The pointer's x position along a four-step level ruler is the per-frame activeLevel.
  interaction: {
    kind: 'level',
    hint: 'slide along the levels',
    rest: () => [(LEVEL_PARAMETER.default + 0.5) / 4, 0.5],
    resolve(pointer, {outputs, peek}) {
      // The whole canvas width is four equal bands, one per level.
      const activeLevel = Math.min(3, Math.max(0, Math.floor(pointer.unitX * 4)));
      const [west, south, east, north] = HONSHU_WINDOW;
      const bandWidth = (east - west) / 4;
      const bottom = south + (north - south) * 0.02;
      const top = south + (north - south) * 0.1;
      const glyphs: RunGlyph[] = [];
      for (let level = 0; level < 4; level++) {
        const left = west + level * bandWidth;
        glyphs.push({
          shape: 'box',
          minimum: [left + bandWidth * 0.02, bottom],
          maximum: [left + bandWidth * 0.98, top]
        });
      }
      // The active band is crossed out with an inner box and a diagonal.
      const left = west + activeLevel * bandWidth;
      glyphs.push(
        {
          shape: 'box',
          minimum: [left + bandWidth * 0.06, bottom + (top - bottom) * 0.2],
          maximum: [left + bandWidth * 0.94, top - (top - bottom) * 0.2]
        },
        {
          shape: 'line',
          from: [left + bandWidth * 0.02, bottom],
          to: [left + bandWidth * 0.98, top]
        },
        {shape: 'line', from: [left + bandWidth * 0.02, top], to: [left + bandWidth * 0.98, bottom]}
      );
      const count = peek(outputs.selectedCount);
      return {
        values: {activeLevel},
        glyphs,
        readout: `activeLevel ${activeLevel} · ${count ? count[0] : '…'} cells drawn (GPU count)`
      };
    }
  },
  draw: ({outputs, values}) => {
    const fill = outputs.levelFills[values.activeLevel];
    return [
      {
        layer: 'polygons',
        positions: fill.boundaries,
        indices: fill.indices,
        owners: fill.owners,
        values: fill.owners,
        triangleCount: fill.triangleCount,
        // One flat color per level, so the four pictures cannot be mistaken for each other.
        color: LEVEL_COLORS[values.activeLevel],
        alpha: 0.6
      },
      {
        layer: 'triangles',
        positions: fill.boundaries,
        triangles: outputs.markers,
        count: outputs.selectedCount,
        color: [1, 1, 1, 0.9]
      }
    ];
  }
});
