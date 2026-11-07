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
import {createClusteredPoints} from '../../inputs';
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
  caption: 'The active level and its cell count are chosen on the GPU from one word.',
  createInputs: ({random, world}) => createClusteredPoints(random, world, 900, 4, 0.12),
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
  draw: ({outputs, values}) => {
    const fill = outputs.levelFills[values.activeLevel];
    return [
      {
        layer: 'polygons',
        positions: fill.boundaries,
        indices: fill.indices,
        owners: fill.owners,
        values: outputs.levels[values.activeLevel].sumValues,
        triangleCount: fill.triangleCount,
        range: [0, 8 * Math.pow(3, values.activeLevel)],
        alpha: 1
      },
      {
        layer: 'triangles',
        positions: fill.boundaries,
        triangles: outputs.markers,
        count: outputs.selectedCount,
        color: [1, 1, 1, 0.4]
      }
    ];
  }
});
