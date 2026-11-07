// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUS2CellHierarchy walks the S2 cell tree with exact integer arithmetic: parent at a level, the
// four children, or the four edge neighbors. The seed cells are outlined; the result cells are
// filled, one color per seed.

import {GPUS2CellHierarchy} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints} from '../../inputs';
import {addCellFill, addCellGeometry, addPointKeys, createCellWireframe} from './support/cells';
import {parameters} from './s2-cell-hierarchy.parameters';

const OPERATIONS = ['parent', 'children', 'neighbors'] as const;
const STRIDES = [1, 4, 4];

export default defineOperationRun({
  parameters,
  caption: 'S2 parents, children or neighbors of the seed cells (outlined).',
  createInputs: ({random, world}) => createClusteredPoints(random, world, 48, 5, 0.15),
  build(builder, points, values) {
    const operation = OPERATIONS[values.operation];
    const stride = STRIDES[values.operation];
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const seeds = addPointKeys(builder, 's2', values.inputLevel, positions);
    const results = builder.allocate('results', 'uint32x2', points.count * stride);
    const status = builder.allocate('hierarchy-status', 'uint32', points.count);
    builder.graph.add(
      new GPUS2CellHierarchy({
        cells: seeds.view,
        operation,
        level: values.level,
        output: {cells: results.view, status: status.view}
      })
    );
    const seedGeometry = addCellGeometry(builder, 's2', seeds, 'seed-geometry');
    const wireframe = createCellWireframe(points.count, seedGeometry.stride, 4);
    return {
      outputs: {
        seedBoundaries: seedGeometry.boundaries,
        wireframe: builder.upload('wireframe', 'uint32', wireframe),
        wireframeCount: wireframe.length / 3,
        sources: builder.upload(
          'sources',
          'uint32',
          Uint32Array.from({length: points.count * stride}, (_, row) => Math.floor(row / stride))
        ),
        fill: addCellFill(builder, 's2', results)
      }
    };
  },
  draw: ({outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.fill.boundaries,
      indices: outputs.fill.indices,
      owners: outputs.fill.owners,
      values: outputs.sources,
      triangleCount: outputs.fill.triangleCount,
      alpha: 0.6
    },
    {
      layer: 'triangles',
      positions: outputs.seedBoundaries,
      triangles: outputs.wireframe,
      count: outputs.wireframeCount,
      radius: 1.6,
      color: [1, 1, 1, 0.95]
    }
  ]
});
