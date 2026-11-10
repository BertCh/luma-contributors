// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUS2CellHierarchy walks the S2 cell tree with exact integer arithmetic: parent at a level, the
// four children, or the four edge neighbors. The seed cells are outlined; the result cells are
// filled, one color per seed.

import {GPUS2CellHierarchy} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {chooseSticky, createPointIndex} from '../../pick';
import {getHonshuEventSites, HONSHU_WINDOW} from '../../../data/point-observations';
import {addCellFill, addCellGeometry, addPointKeys, createCellWireframe} from './support/cells';
import {getBoundaryGlyph} from './support/pointer';
import {parameters} from './s2-cell-hierarchy.parameters';

const OPERATIONS = ['parent', 'children', 'neighbors'] as const;
const STRIDES = [1, 4, 4];

export default defineOperationRun({
  parameters,
  geographicWindow: HONSHU_WINDOW,
  caption: 'S2 parents, children or neighbors of the seed cells (outlined).',
  // Six well-separated epicenters: each seed's result cells stay apart.
  createInputs: () => {
    const sites = getHonshuEventSites([
      [140.8, 36.5],
      [142.2, 36.4],
      [143.5, 36.9],
      [141.0, 37.9],
      [142.4, 37.8],
      [143.6, 38.3]
    ]);
    const positions = Float32Array.from(sites.flat());
    return {count: sites.length, positions, index: createPointIndex(positions), held: -1};
  },
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
  // Probe: the seed under the pointer, with the cells the hierarchy operation wrote for it.
  interaction: {
    kind: 'probe',
    hint: 'hover a seed',
    rest: ({toUnit, inputs}) => toUnit([inputs.positions[0], inputs.positions[1]]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const seed = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.held);
      inputs.held = seed;
      if (seed < 0) return {};
      const stride = STRIDES[values.operation];
      const seedBoundaries = peek(outputs.seedBoundaries);
      const resultBoundaries = peek(outputs.fill.boundaries);
      const glyphs: RunGlyph[] = [];
      const seedOutline = getBoundaryGlyph(seedBoundaries, seed, 4, 4);
      if (seedOutline) glyphs.push(seedOutline);
      for (let slot = 0; slot < stride; slot++) {
        const outline = getBoundaryGlyph(
          resultBoundaries,
          seed * stride + slot,
          outputs.fill.stride,
          4
        );
        if (outline) glyphs.push(outline);
      }
      const operation = OPERATIONS[values.operation];
      const result =
        operation === 'parent'
          ? `parent at level ${values.level}`
          : `${stride} ${operation} of level ${values.inputLevel}`;
      return {focus: seed, glyphs, readout: `seed ${seed} level ${values.inputLevel} → ${result}`};
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.fill.boundaries,
      indices: outputs.fill.indices,
      owners: outputs.fill.owners,
      values: outputs.sources,
      triangleCount: outputs.fill.triangleCount,
      alpha: 0.4
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
