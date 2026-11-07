// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUS2CapCovering covers spherical caps (a center on the unit sphere and an angular radius) with
// uniform-level S2 cells: all cells that touch the cap (exterior) or lie inside it (interior).

import {GPUS2CapCovering} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addCellFill} from './support/cells';
import {createCircleOutline, getUnitVector} from './support/polygons';
import {parameters} from './s2-cap-covering.parameters';

const CAPACITY = 256;
const DEGREES_TO_RADIANS = Math.PI / 180;

export default defineOperationRun({
  parameters,
  caption: 'S2 cells covering three caps; the circles are the caps themselves.',
  createInputs: ({world}) => ({
    centers: [
      [world.width * 0.25, 0.55],
      [world.width * 0.55, 0.4],
      [world.width * 0.8, 0.62]
    ] as [number, number][]
  }),
  build(builder, inputs, values) {
    const capCount = inputs.centers.length;
    const caps = builder.allocate('caps', 'float32x4', capCount);
    const outline = builder.allocate('cap-outline', 'float32x4', capCount * 64);
    const cells = builder.allocate('cells', 'uint32x2', capCount * CAPACITY);
    const counts = builder.allocate('counts', 'uint32x2', capCount);
    const status = builder.allocate('cap-status', 'uint32', capCount);
    builder.graph.add(
      new GPUS2CapCovering({
        caps: caps.view,
        level: values.level,
        mode: values.mode === 0 ? 'exterior' : 'interior',
        capacity: CAPACITY,
        output: {cells: cells.view, counts: counts.view, status: status.view}
      })
    );
    return {
      outputs: {
        outline,
        sources: builder.upload(
          'sources',
          'uint32',
          Uint32Array.from({length: capCount * CAPACITY}, (_, row) => Math.floor(row / CAPACITY))
        ),
        fill: addCellFill(builder, 's2', cells)
      },
      update: current => {
        caps.write(
          Float32Array.from(
            inputs.centers.flatMap(([longitude, latitude]) => [
              ...getUnitVector(longitude, latitude),
              current.radius * DEGREES_TO_RADIANS
            ])
          )
        );
        outline.write(
          Float32Array.from(
            inputs.centers.flatMap(center => [...createCircleOutline(center, current.radius)])
          )
        );
      }
    };
  },
  rebuildOn: ['level', 'mode'],
  draw: ({outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.fill.boundaries,
      indices: outputs.fill.indices,
      owners: outputs.fill.owners,
      values: outputs.sources,
      triangleCount: outputs.fill.triangleCount,
      alpha: 0.7
    },
    {
      layer: 'triangles',
      positions: outputs.fill.boundaries,
      triangles: outputs.fill.wireframe,
      count: outputs.fill.wireframeCount,
      radius: 1,
      color: [1, 1, 1, 0.5]
    },
    {layer: 'lines', context: true, segments: outputs.outline, radius: 1.8}
  ]
});
