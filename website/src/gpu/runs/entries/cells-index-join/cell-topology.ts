// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCellTopology expands three seed cells into their k-ring disk (colored by grid distance) or
// their ring at distance k. The neighbor keys are decoded with GPUCellGeometry only to draw them.

import {GPUCellTopology} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {chooseSticky, createPointIndex} from '../../pick';
import {getHonshuEventSites, HONSHU_WINDOW} from '../../../data/point-observations';
import {addCellFill, addPointKeys, getCellFamily} from './support/cells';
import {formatCellKey} from './support/pointer';
import {parameters} from './cell-topology.parameters';

export default defineOperationRun({
  parameters,
  geographicWindow: HONSHU_WINDOW,
  caption: 'Cell disks and rings around three observed Honshu earthquake epicenters.',
  createInputs: () => {
    const anchors = Float32Array.from(
      getHonshuEventSites([
        [141.2, 36.6],
        [142.3, 37.5],
        [141.6, 38.1]
      ]).flat()
    );
    // `seeds` moves with the pointer; `anchors` stay put and decide which seed the pointer holds.
    return {anchors, seeds: Float32Array.from(anchors), index: createPointIndex(anchors), held: -1};
  },
  build(builder, inputs, values) {
    const family = getCellFamily(values.family);
    const isDisk = values.operation === 0;
    const k = values.k;
    const stride = family === 'h3' ? 3 * k * (k + 1) + 1 : (2 * k + 1) ** 2;
    const ringStride = (family === 'h3' ? 6 : 8) * k;
    const seedCount = inputs.seeds.length / 2;
    const rowStride = isDisk ? stride : ringStride;
    const seeds = builder.allocate('seeds', 'float32x2', seedCount);
    const seedCells = addPointKeys(builder, family, family === 'h3' ? 5 : 11, seeds);
    const neighbors = builder.allocate('neighbors', 'uint32x2', seedCount * rowStride);
    const distances = builder.allocate('distances', 'uint32', seedCount * rowStride);
    builder.graph.add(
      new GPUCellTopology({
        family,
        operation: {type: isDisk ? 'disk' : 'ring', k},
        cells: seedCells.view,
        output: isDisk
          ? {cells: neighbors.view, distances: distances.view}
          : {cells: neighbors.view}
      })
    );
    return {
      outputs: {seeds, seedCells, distances, fill: addCellFill(builder, family, neighbors)},
      update: () => seeds.write(inputs.seeds)
    };
  },
  rebuildOn: ['family', 'operation', 'k'],
  // The pointer carries the nearest seed: the disk or ring around it is keyed again every frame.
  interaction: {
    kind: 'place',
    hint: 'move a seed',
    rest: ({toUnit, inputs}) => toUnit([inputs.anchors[0], inputs.anchors[1]]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      inputs.held = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.held);
      inputs.seeds.set(inputs.anchors);
      inputs.seeds[inputs.held * 2] = pointer.x;
      inputs.seeds[inputs.held * 2 + 1] = pointer.y;
      const keys = peek(outputs.seedCells);
      const k = values.k;
      const isDisk = values.operation === 0;
      const family = getCellFamily(values.family);
      const cellCount = isDisk
        ? family === 'h3'
          ? 3 * k * (k + 1) + 1
          : (2 * k + 1) ** 2
        : (family === 'h3' ? 6 : 8) * k;
      return {
        rerun: true,
        glyphs: [{shape: 'crosshair', at: [pointer.x, pointer.y]}],
        readout: `${cellCount} cells · k ${k} · seed ${formatCellKey(keys, inputs.held)}`
      };
    }
  },
  draw: ({outputs, values}) => [
    {
      layer: 'polygons',
      positions: outputs.fill.boundaries,
      indices: outputs.fill.indices,
      owners: outputs.fill.owners,
      values: outputs.distances,
      triangleCount: outputs.fill.triangleCount,
      color: values.operation === 0 ? undefined : [0.95, 0.6, 0.25, 0.9]
    },
    {layer: 'dots', positions: outputs.seeds, radius: 4, color: [1, 1, 1, 1]}
  ]
});
