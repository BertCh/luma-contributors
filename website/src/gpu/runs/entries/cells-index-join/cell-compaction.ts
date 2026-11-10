// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCellCompaction (compact) merges complete sets of sibling H3 cells into their parent,
// recursively. The input is the resolution-6 descendants of seven resolution-4 cells (GPUCellTopology
// disk + children); the mask drops some of them, and only complete families merge. The pointer is an
// eraser: a small kernel clears the mask of every input cell whose center is under the cursor disk.

import {GPUCellCompaction, GPUCellTopology} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {getLargestHonshuEvent} from '../../../data/point-observations';
import {
  addCellGeometry,
  addPointKeys,
  createCellFanMesh,
  createCellWireframe,
  H3_BOUNDARY_STRIDE
} from './support/cells';
import {addHighWordField, addKernel} from './support/kernels';
import {parameters} from './cell-compaction.parameters';

const WINDOW = [140.6, 37.1, 142.6, 38.3] as const;
const DISK_CELLS = 7;
const CHILDREN_PER_CELL = 49;
const INPUT_COUNT = DISK_CELLS * CHILDREN_PER_CELL;
/** Eraser radius in degrees of latitude; longitude is foreshortened by the cosine of the latitude. */
const ERASER_RADIUS = 0.1;
const LONGITUDE_SCALE = Math.cos((((WINDOW[1] + WINDOW[3]) / 2) * Math.PI) / 180);

function getRowHash(row: number): number {
  const value = Math.sin(row * 12.9898 + 4.1) * 43758.5453;
  return value - Math.floor(value);
}

export default defineOperationRun({
  parameters,
  geographicWindow: WINDOW,
  caption: 'Compact a masked H3 disk around the M7.3 Honshu earthquake in March 2022.',
  createInputs: () => ({
    center: Float32Array.from(getLargestHonshuEvent()),
    pointer: [(WINDOW[0] + WINDOW[2]) / 2, (WINDOW[1] + WINDOW[3]) / 2] as [number, number]
  }),
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
    const inputGeometry = addCellGeometry(builder, 'h3', children, 'input-geometry');
    // The mask is written on the GPU: a cell stays when its row hash is under `keep` and its
    // center is outside the eraser disk at the pointer.
    const hashes = builder.upload(
      'row-hashes',
      'float32',
      Float32Array.from({length: INPUT_COUNT}, (_, row) => getRowHash(row))
    );
    const control = builder.allocate('mask-control', 'float32', 4);
    const mask = builder.allocate('mask', 'uint32', INPUT_COUNT);
    addKernel(builder, {
      id: 'mask-eraser',
      invocationCount: INPUT_COUNT,
      bindings: [
        {name: 'centers', buffer: inputGeometry.centers, access: 'read'},
        {name: 'hashes', buffer: hashes, access: 'read'},
        {name: 'control', buffer: control, access: 'read'},
        {name: 'mask', buffer: mask, access: 'read_write'}
      ],
      body: `let base = centersOffset + index * 2u;
  let offset = vec2<f32>(
    (centers[base] - control[controlOffset + 1u]) * ${LONGITUDE_SCALE},
    centers[base + 1u] - control[controlOffset + 2u]
  );
  let isErased = length(offset) < control[controlOffset + 3u];
  let isKept = hashes[hashesOffset + index] < control[controlOffset];
  mask[maskOffset + index] = select(0u, 1u, isKept && !isErased);`
    });
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
    const outputGeometry = addCellGeometry(builder, 'h3', compacted, 'output-geometry');
    const mesh = createCellFanMesh(INPUT_COUNT, H3_BOUNDARY_STRIDE, 6);
    const wireframe = createCellWireframe(INPUT_COUNT, H3_BOUNDARY_STRIDE, 6);
    return {
      outputs: {
        inputBoundaries: inputGeometry.boundaries,
        boundaries: outputGeometry.boundaries,
        mask,
        count,
        indices: builder.upload('fill-indices', 'uint32', mesh.indices),
        owners: builder.upload('fill-owners', 'uint32', mesh.owners),
        triangleCount: mesh.triangleCount,
        wireframe: builder.upload('wireframe', 'uint32', wireframe),
        wireframeCount: wireframe.length / 3,
        // The resolution (4, 5 or 6) stored in each compacted key colors the output cells.
        resolutions: addHighWordField(builder, 'compacted-resolution', compacted, 20)
      },
      update: current =>
        control.write(
          Float32Array.of(current.keep, inputs.pointer[0], inputs.pointer[1], ERASER_RADIUS)
        )
    };
  },
  rebuildOn: ['minimumResolution'],
  // The pointer erases the input cells under it; a family with a hole can no longer merge.
  interaction: {
    kind: 'window',
    hint: 'erase cells',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, peek}) {
      inputs.pointer = [pointer.x, pointer.y];
      const mask = peek(outputs.mask);
      const count = peek(outputs.count);
      let masked = 0;
      if (mask) for (const bit of mask) if (bit === 0) masked++;
      return {
        rerun: true,
        glyphs: [{shape: 'ring', at: inputs.pointer, radius: ERASER_RADIUS, dashed: true}],
        readout: `${masked} of ${INPUT_COUNT} inputs masked · ${count ? count[0] : '…'} cells out`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'triangles',
      context: true,
      positions: outputs.inputBoundaries,
      triangles: outputs.wireframe,
      count: outputs.wireframeCount,
      radius: 0.8,
      alpha: 0.5
    },
    {
      layer: 'polygons',
      positions: outputs.boundaries,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.resolutions,
      triangleCount: outputs.triangleCount,
      ramp: 'categorical',
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
