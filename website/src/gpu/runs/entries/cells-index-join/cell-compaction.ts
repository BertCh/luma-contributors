// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCellCompaction (compact) merges complete sets of sibling H3 cells into their parent,
// recursively. The input is the resolution-6 descendants of seven resolution-4 cells (GPUCellTopology
// disk + children); the mask drops some of them, and only complete families merge.

import {GPUCellCompaction, GPUCellTopology} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {
  addCellGeometry,
  addPointKeys,
  createCellFanMesh,
  createCellWireframe,
  H3_BOUNDARY_STRIDE
} from './support/cells';
import {parameters} from './cell-compaction.parameters';

const DISK_CELLS = 7;
const CHILDREN_PER_CELL = 49;
const INPUT_COUNT = DISK_CELLS * CHILDREN_PER_CELL;

function getRowHash(row: number): number {
  const value = Math.sin(row * 12.9898 + 4.1) * 43758.5453;
  return value - Math.floor(value);
}

export default defineOperationRun({
  parameters,
  caption: 'Complete families of cells merge into their parent; broken families stay fine.',
  createInputs: ({world}) => ({center: Float32Array.of(world.width * 0.5, 0.5)}),
  build(builder, inputs, values) {
    const center = builder.upload('center', 'float32x2', inputs.center);
    const centerCell = addPointKeys(builder, 'h3', 4, center);
    const disk = builder.allocate('disk', 'uint32x2', DISK_CELLS);
    builder.graph.add(
      new GPUCellTopology({
        id: 'disk-topology',
        family: 'h3',
        operation: {type: 'disk', k: 1},
        cells: centerCell.view,
        output: {cells: disk.view}
      })
    );
    const children = builder.allocate('children', 'uint32x2', INPUT_COUNT);
    builder.graph.add(
      new GPUCellTopology({
        id: 'children-topology',
        family: 'h3',
        operation: {type: 'children', resolution: 6, inputResolution: 4},
        cells: disk.view,
        output: {cells: children.view}
      })
    );
    const mask = builder.allocate('mask', 'uint32', INPUT_COUNT);
    const compacted = builder.allocate('compacted', 'uint32x2', INPUT_COUNT);
    const count = builder.allocate('count', 'uint32', 1);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPUCellCompaction({
        family: 'h3',
        operation: {type: 'compact', resolution: 6, minimumResolution: values.minimumResolution},
        cells: children.view,
        mask: mask.view,
        output: {cells: compacted.view, status: {count: count.view, overflow: overflow.view}}
      })
    );
    const inputGeometry = addCellGeometry(builder, 'h3', children, 'input-geometry');
    const outputGeometry = addCellGeometry(builder, 'h3', compacted, 'output-geometry');
    const mesh = createCellFanMesh(INPUT_COUNT, H3_BOUNDARY_STRIDE, 6);
    const wireframe = createCellWireframe(INPUT_COUNT, H3_BOUNDARY_STRIDE, 6);
    return {
      outputs: {
        inputBoundaries: inputGeometry.boundaries,
        boundaries: outputGeometry.boundaries,
        indices: builder.upload('fill-indices', 'uint32', mesh.indices),
        owners: builder.upload('fill-owners', 'uint32', mesh.owners),
        triangleCount: mesh.triangleCount,
        wireframe: builder.upload('wireframe', 'uint32', wireframe),
        wireframeCount: wireframe.length / 3,
        // Per-cell values of the output fill: one family color per coarse cell.
        cellValues: builder.upload(
          'cell-values',
          'float32',
          Float32Array.from({length: INPUT_COUNT}, (_, row) => (row % 7) / 6)
        )
      },
      update: current =>
        mask.write(
          Uint32Array.from({length: INPUT_COUNT}, (_, row) =>
            getRowHash(row) < current.keep ? 1 : 0
          )
        )
    };
  },
  rebuildOn: ['minimumResolution'],
  draw: ({outputs}) => [
    {
      layer: 'triangles',
      context: true,
      positions: outputs.inputBoundaries,
      triangles: outputs.wireframe,
      count: outputs.wireframeCount
    },
    {
      layer: 'polygons',
      positions: outputs.boundaries,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.cellValues,
      triangleCount: outputs.triangleCount,
      alpha: 0.8
    },
    {
      layer: 'triangles',
      positions: outputs.boundaries,
      triangles: outputs.wireframe,
      count: outputs.wireframeCount,
      radius: 1.6
    }
  ]
});
